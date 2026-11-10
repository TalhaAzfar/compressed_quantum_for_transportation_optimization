"""Complete-anneal MPS fitting and independent 42-qubit hardware feasibility pilot."""
import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time

from .contribution_hardware_core import ROOT, read, save, digest, now

CONFIG = ROOT / 'configs/larger_width_feasibility_2026_10.json'
EXTRA = ""


def settings():
    c = read(CONFIG)
    return c, ROOT / c['output_dir']


def runner():
    return [sys.executable, '-m', 'revision.local_runner', '--extra-site', EXTRA,
            'revision.larger_width_feasibility']


def source_paths():
    return ['configs/larger_width_feasibility_2026_10.json',
            'revision/larger_width_feasibility.py', 'revision/placement_aer_reference.py',
            'revision/flp_mps_events.py', 'revision/flp42_fidelity_parallel.py',
            'revision/full_circuit_explicit.py', 'revision/tensor_prefix_core.py',
            'revision/tensor_prefix_sabre.py', 'revision/cpu_memory_guard.py',
            'revision/contribution_hardware_core.py', 'revision/validate_tensor_prefix.py',
            'revision/audit_tensor_prefix_statistics.py', 'revision/hardware_width_figure.py',
            'revision/larger_width_cpu_diagnostic.py',
            'tests/test_larger_width_feasibility.py']


def verify_hashes(record):
    for field in ['source_sha256', 'input_sha256']:
        for name, expected in record.get(field, {}).items():
            if digest(ROOT / name) != expected:
                raise RuntimeError('Frozen file changed: ' + name)


def freeze():
    c, out = settings()
    for name in ['targets', 'fits', 'circuits', 'logs', 'checkpoints', 'preflight',
                 'raw_counts', 'source_snapshot']:
        (out / name).mkdir(parents=True, exist_ok=True)
    sources = source_paths()
    tests = (out / 'tests.log').read_text()
    assert 'passed' in tests and 'FAILED' not in tests and 'ERROR' not in tests
    record = dict(config=c, source_sha256={p: digest(ROOT / p) for p in sources},
                  input_sha256={p: digest(ROOT / p) for p in
                                [c['parent_case'], c['archived_backend_target']]})
    if c.get('threaded_target_seed'):
        record['input_sha256'][c['threaded_target_seed']] = digest(ROOT / c['threaded_target_seed'])
    path = out / 'preparation_protocol.json'
    if path.exists():
        old = read(path)
        assert all(old[k] == v for k, v in record.items()), 'Frozen preparation changed'
        verify_hashes(old)
    else:
        record.update(frozen_utc=now(), previous_42_qubit_fitting_outcomes_known=True,
                      new_hardware_outcomes_seen=False,
                      packages={p: importlib.metadata.version(p) for p in
                                ['qiskit', 'qiskit-aer', 'quimb', 'qiskit-addon-aqc-tensor',
                                 'qiskit-ibm-runtime', 'numpy', 'scipy']})
        save(path, record)
        for name in sources:
            shutil.copy2(ROOT / name, out / 'source_snapshot' / Path(name).name)
    case = read(ROOT / c['parent_case'])
    assert case['model']['n_qubits'] == 42
    save(out / 'case.json', case)
    print('Frozen complete-anneal 42-qubit pilot; no minimum fitting fidelity.', flush=True)


def preflight():
    from .nighthawk_study import connect
    from .tensor_prefix_core import path_patch
    c, out = settings()
    _, backend = connect(c)
    assert backend.status().operational
    processor = backend.configuration().processor_type
    assert str(processor.get('revision')) == '1', processor
    properties = backend.properties(refresh=True).to_dict()
    save(out / 'preflight/device.json', dict(observed_utc=now(), backend=backend.name,
         processor=processor, backend_version=backend.backend_version,
         calibration_timestamp=str(properties['last_update_date'])))
    save(out / 'preflight/properties.json', properties)
    (out / 'preflight/target.pkl').write_bytes(pickle.dumps(backend.target))
    physical, edges, diagnostic = path_patch(backend.target, 42, c['patch_seed'])
    save(out / 'preflight/patch.json', dict(physical=physical, edges=edges,
                                          selection=diagnostic))
    print('Live preflight:', backend.name, processor, '42-qubit patch', flush=True)


def aer_state(qc, bond, cutoff, threads=1):
    import quimb.tensor as qtn
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_matrix_product_state
    from .placement_aer_reference import from_aer_mps
    sim = AerSimulator(method='matrix_product_state',
                       matrix_product_state_max_bond_dimension=bond,
                       matrix_product_state_truncation_threshold=cutoff,
                       max_parallel_threads=threads, max_parallel_experiments=1,
                       mps_omp_threads=threads, mps_parallel_threshold=16)
    saved = qc.copy()
    saved.save_matrix_product_state()
    result = sim.run(saved, shots=1).result()
    if not result.success:
        raise RuntimeError(str(result.status))
    converted = from_aer_mps(result.data(0)['matrix_product_state'])
    return qtn.CircuitMPS(N=qc.num_qubits, psi0=converted.psi, max_bond=bond,
                          cutoff=cutoff)


def fuse_zz(qc):
    """Exact local rewrite for simulation; retain the original hardware source."""
    from qiskit import QuantumCircuit
    assert not qc.num_clbits
    result = QuantumCircuit(qc.num_qubits)
    result.global_phase = qc.global_phase
    i = fused = 0
    while i < len(qc.data):
        a = qc.data[i]
        if i+2 < len(qc.data):
            b, d = qc.data[i+1:i+3]
            if (a.operation.name == d.operation.name == 'cx' and
                b.operation.name == 'rz' and a.qubits == d.qubits and
                b.qubits == (a.qubits[1],)):
                result.rzz(b.operation.params[0], *[qc.find_bit(q).index for q in a.qubits])
                i += 3
                fused += 1
                continue
        result.append(a.operation, [qc.find_bit(q).index for q in a.qubits])
        i += 1
    return result, fused


def target_worker(bond):
    from .tensor_prefix_core import target_circuit
    from .flp_mps_events import probabilities
    from .nighthawk_study import qpy_write
    c, out = settings()
    model = read(out / 'case.json')['model']
    qc = target_circuit(model, c['endpoint'], c['steps'], c['duration'])
    if not (out / 'circuits/source.qpy').exists():
        qpy_write(out / 'circuits/source.qpy', qc)
    start = time.monotonic()
    print('Target bond', bond, 'started', flush=True)
    simulation_circuit, fused = fuse_zz(qc) if c['target_gate_fusion'] else (qc, 0)
    if c['target_backend'] == 'quimb':
        from .flp42_fidelity_parallel import state as quimb_state
        state = quimb_state(simulation_circuit, bond, c['cutoff'])
        construction = 'Direct threaded Quimb MPS; no dense target'
    elif c['target_backend'] == 'aer':
        state = aer_state(simulation_circuit, bond, c['cutoff'], c['threads_per_worker'])
        construction = 'Direct Aer MPS, converted to Quimb; no dense target'
    else:
        raise ValueError('Unknown target backend')
    path = out / 'targets' / f'b{bond}.pkl'
    path.write_bytes(pickle.dumps(state))
    record = dict(bond=bond, norm=float(abs(state.psi.H @ state.psi)),
                  achieved_bond=state.psi.max_bond(), state_sha256=digest(path),
                  construction=construction,
                  fused_zz_blocks=fused,
                  simulation_gate_counts=dict(simulation_circuit.count_ops()),
                  construction_seconds=time.monotonic()-start)
    record['events'] = probabilities(model, state)
    record.update(status='completed', completed_utc=now(), seconds=time.monotonic()-start)
    save(out / 'targets' / f'b{bond}.json', record)
    print('Target bond', bond, 'complete:', record['events'], flush=True)


def target_selection():
    from .tensor_prefix_core import fidelity
    c, out = settings()
    comparisons = []
    selected = None
    for a, b in zip(c['target_bonds'], c['target_bonds'][1:]):
        paths = [out / 'targets' / f'b{bond}.json' for bond in [a, b]]
        if not all(p.exists() and read(p).get('status') == 'completed' for p in paths):
            continue
        ra, rb = map(read, paths)
        for bond, record in [(a, ra), (b, rb)]:
            assert digest(out / 'targets' / f'b{bond}.pkl') == record['state_sha256']
        low = pickle.loads((out / 'targets' / f'b{a}.pkl').read_bytes())
        high = pickle.loads((out / 'targets' / f'b{b}.pkl').read_bytes())
        overlap = fidelity(low, high)
        rel = {e: abs(ra['events'][e]-rb['events'][e])/max(rb['events'][e], 1e-300)
               for e in ['p_feasible', 'p_good', 'p_optimal']}
        accepted = (overlap >= c['minimum_target_overlap'] and
                    rel['p_feasible'] <= c['maximum_feasibility_relative_change'])
        comparisons.append(dict(low_bond=a, high_bond=b, overlap=overlap,
                                relative_event_changes=rel, accepted=accepted))
        if accepted and selected is None:
            selected = b
        del low, high
    record = dict(status='converged' if selected else 'unresolved',
                  selected_bond=selected, comparisons=comparisons,
                  interpretation='Numerical MPS convergence; no exact-state certificate.',
                  completed_utc=now())
    save(out / 'targets/selection.json', record)
    print('Target selection', record, flush=True)
    return record


def fit_worker(layers, index):
    import numpy as np
    from .tensor_prefix_core import native_ansatz
    from .flp42_fidelity_parallel import checkpoint_fit
    from .steps_native_sweep import extend_parameters
    from .flp_mps_events import probabilities
    from .nighthawk_study import qpy_write
    c, out = settings()
    selection = read(out / 'targets/selection.json')
    assert selection['status'] == 'converged'
    target = pickle.loads((out / 'targets' / f"b{selection['selected_bond']}.pkl").read_bytes())
    patch = read(out / 'preflight/patch.json')
    ansatz = native_ansatz(42, patch['edges'], layers)
    x = np.random.default_rng(c['seed']+layers*10+index).normal(0, .02,
                                                             ansatz.num_parameters)
    initialization = 'fixed-seed small rotations'
    earlier = [read(p) for p in (out / 'fits').glob('*.json')]
    earlier = [r for r in earlier if r.get('verified') and r['layers'] < layers]
    if earlier:
        old = max(earlier, key=lambda r: (r['fidelity'], -r['layers'], -r['index']))
        x = extend_parameters(old['parameters'], 42, len(patch['edges']),
                              old['layers'], layers)
        if index:
            x += np.random.default_rng(c['seed']+layers*10+index).normal(0, .02, len(x))
        initialization = f"identity extension of L{old['layers']} start {old['index']}"
    name = f'l{layers}_r{index}'
    print('Fit', name, 'started', flush=True)
    fc = dict(c, fit_cutoff=c['cutoff'], check_cutoff=c['cutoff'],
              convergence_min=c['minimum_target_overlap'])
    bound, record = checkpoint_fit(target, ansatz, x, fc, out / 'checkpoints' / f'{name}.json')
    record.update(layers=layers, index=index, initialization=initialization,
                  target_bond=selection['selected_bond'], completed_utc=now())
    qpy_write(out / 'circuits' / f'{name}.qpy', bound)
    if record['verified']:
        from .flp42_fidelity_parallel import state
        high = state(bound, c['check_bond'], c['cutoff'])
        record['ideal_events'] = probabilities(read(out / 'case.json')['model'], high)
    save(out / 'fits' / f'{name}.json', record)
    print('Fit', name, 'F', record['fidelity'], 'verified', record['verified'], flush=True)


def pool(tasks):
    """Two bounded workers; one aggregate RSS watchdog also covers verification."""
    import psutil
    from .cpu_memory_guard import physical_cpu_ids
    c, out = settings()
    cpus = physical_cpu_ids()
    assert len(cpus) >= c['workers']*c['threads_per_worker']
    available = [cpus[i*c['threads_per_worker']:(i+1)*c['threads_per_worker']]
                 for i in range(c['workers'])]
    pending = list(tasks)
    active = []
    peak = 0
    try:
        while pending or active:
            for task, cpu, proc, log, started in list(active):
                if proc.poll() is None:
                    continue
                log.close()
                save(out / 'logs' / f"{task['name']}_process.json",
                     dict(returncode=proc.returncode, seconds=time.monotonic()-started,
                          cpus=cpu, threads=c['threads_per_worker']))
                if not (out / task['result']).exists():
                    save(out / task['result'], dict(status='worker_failed', verified=False,
                                                   task=task, returncode=proc.returncode))
                active.remove((task, cpu, proc, log, started))
                available.append(cpu)
            while pending and available:
                task = pending.pop(0)
                if (out / task['result']).exists():
                    continue
                cpu = available.pop(0)
                command = [sys.executable, '-m', 'revision.cpu_memory_guard',
                           '--address-gib', str(c['address_gib']), '--cpus',
                           ','.join(map(str, cpu)), '--threads', str(c['threads_per_worker']),
                           '--', *runner(), *task['args']]
                log = (out / 'logs' / f"{task['name']}.log").open('a')
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                active.append((task, cpu, proc, log, time.monotonic()))
                print('Launched', task['name'], 'CPUs', cpu, flush=True)
            total = 0
            for task, cpu, proc, log, started in active:
                try:
                    tree = [psutil.Process(proc.pid)]
                    tree += tree[0].children(recursive=True)
                    total += sum(p.memory_info().rss for p in tree if p.is_running())
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic()-started > task['seconds']:
                    raise RuntimeError('Worker exceeded time budget: '+task['name'])
            peak = max(peak, total)
            if total > c['aggregate_rss_gib']*1024**3:
                raise RuntimeError('Aggregate RAM watchdog reached')
            if psutil.virtual_memory().available < c['system_reserve_gib']*1024**3:
                raise RuntimeError('System memory reserve reached')
            save(out / 'progress.json', dict(updated_utc=now(),
                 running=[t['name'] for t, *_ in active], pending=len(pending),
                 rss_gib=total/1024**3, peak_rss_gib=peak/1024**3))
            time.sleep(1)
    finally:
        import signal
        for _, _, proc, log, _ in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            log.close()


def prepare():
    from .tensor_prefix_core import target_circuit
    from .nighthawk_study import qpy_write
    c, out = settings()
    verify_hashes(read(out / 'preparation_protocol.json'))
    assert (out / 'preflight/target.pkl').exists(), 'Live hardware preflight required'
    if not (out / 'circuits/source.qpy').exists():
        qpy_write(out / 'circuits/source.qpy', target_circuit(
            read(out / 'case.json')['model'], c['endpoint'], c['steps'], c['duration']))
    for bonds in [c['target_bonds'][:2], c['target_bonds'][2:]]:
        pool([dict(name=f'target_b{b}', result=f'targets/b{b}.json',
                   args=['target', str(b)], seconds=c['target_seconds']+120) for b in bonds])
        if target_selection()['status'] == 'converged':
            break
    if read(out / 'targets/selection.json')['status'] != 'converged':
        save(out / 'preparation_status.json', dict(status='target_unresolved', completed_utc=now()))
        return False
    for layers in c['layers']:
        pool([dict(name=f'fit_l{layers}_r{i}', result=f'fits/l{layers}_r{i}.json',
                   args=['fit', str(layers), str(i)], seconds=c['fit_seconds']+600)
              for i in range(c['starts'])])
    freeze_hardware()
    return True


def best_per_depth(records):
    accepted = [r for r in records if r.get('verified') and 'layers' in r]
    depths = sorted({r['layers'] for r in accepted})
    return [max((r for r in accepted if r['layers'] == l),
                key=lambda r: (r['fidelity'], -r['index'])) for l in depths
            ]


def freeze_hardware():
    from .nighthawk_study import qpy_read, qpy_write
    from .tensor_prefix_sabre import compile_isa_sabre
    from .contribution_hardware_core import check_isa
    c, out = settings()
    assert not (out / 'hardware_protocol.json').exists(), 'Hardware already frozen'
    records = best_per_depth([read(p) for p in (out / 'fits').glob('*.json')])
    assert {0, 2} <= {r['layers'] for r in records}, 'Product and shallow verification required'
    assert len(records) >= 3, 'At least one additional native depth required'
    selection = read(out / 'targets/selection.json')
    assert selection['status'] == 'converged'
    target = pickle.loads((out / 'preflight/target.pkl').read_bytes())
    patch = read(out / 'preflight/patch.json')
    rows = []
    items = [('source', None)] + [(('product' if r['layers']==0 else
             'shallow' if r['layers']==2 else f"aqc_l{r['layers']}"), r) for r in records]
    for method, record in items:
        logical = out / 'circuits' / ('source.qpy' if record is None else
                                      f"l{record['layers']}_r{record['index']}.qpy")
        qc = qpy_read(logical)
        isa, resource = compile_isa_sabre(qc, target, patch['physical'], patch['edges'],
                                         seed=c['transpile_seed'])
        checked = check_isa(isa, target, patch['physical'], resource['final_logical_mapping'])
        assert all(checked[k] == resource[k] for k in checked)
        if record is not None:
            assert resource['swap_count'] == 0
            assert resource['cz_depth'] <= c['maximum_native_cz_depth']
        path = out / 'circuits' / f'{method}_isa.qpy'
        qpy_write(path, isa)
        rows.append(dict(method=method, candidate_id=logical.stem,
                         layers=None if record is None else record['layers'],
                         fidelity=1. if record is None else record['fidelity'],
                         ideal_events=read(out / 'targets' / f"b{selection['selected_bond']}.json")['events']
                         if record is None else record['ideal_events'],
                         logical_qpy=str(logical.relative_to(ROOT)), logical_sha256=digest(logical),
                         isa_qpy=str(path.relative_to(ROOT)), isa_sha256=digest(path),
                         resources=resource))
    plan = dict(rows=rows, blocks=c['screen_blocks'], shots_per_circuit=c['screen_shots'],
                total_shots=len(rows)*c['screen_blocks']*c['screen_shots'])
    assert plan['total_shots']+4*c['confirmation_blocks']*c['confirmation_shots'] <= c['maximum_hardware_shots']
    save(out / 'screen_plan.json', plan)
    inputs = [out / 'case.json', out / 'targets/selection.json', out / 'preflight/target.pkl',
              out / 'preflight/device.json', out / 'preflight/patch.json',
              out / 'tests.log', out / 'preparation_protocol.json']
    for row in rows:
        inputs += [ROOT / row['logical_qpy'], ROOT / row['isa_qpy']]
    inputs += list((out / 'targets').glob('*.json'))+list((out / 'fits').glob('*.json'))
    save(out / 'hardware_protocol.json', dict(frozen_utc=now(),
         source_sha256=read(out / 'preparation_protocol.json')['source_sha256'],
         input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
         screen_plan_sha256=digest(out / 'screen_plan.json'),
         primary_inference=c['inference'], no_minimum_fitting_fidelity=True,
         confirmation_outcomes_seen=False))
    save(out / 'preparation_status.json', dict(status='hardware_ready', completed_utc=now()))
    print('Hardware frozen:',len(rows),'screen circuits;',plan['total_shots'],'screen shots',flush=True)


def choose_screen(rows, pooled):
    counts = {r['method']: r for r in pooled}
    native = [r for r in rows if r['layers'] is not None and r['layers'] > 0]
    return max(native, key=lambda r: (counts[r['method']]['feasible_count']/counts[r['method']]['shots'],
                                     r['fidelity'], -r['resources']['cz_depth'], r['candidate_id']))


def submit(phase, block):
    import numpy as np
    from .nighthawk_study import connect, qpy_read
    from .contribution_hardware_core import check_isa
    from qiskit_ibm_runtime import SamplerV2
    c, out = settings()
    assert phase in ['screen', 'confirm'] and block in [0, 1]
    verify_hashes(read(out / 'hardware_protocol.json'))
    planpath = out / f'{phase}_plan.json'
    plan = read(planpath)
    if phase == 'screen':
        assert digest(planpath) == read(out / 'hardware_protocol.json')['screen_plan_sha256']
    else:
        protocol = read(out / 'confirmation_protocol.json')
        verify_hashes(protocol)
        assert digest(planpath) == protocol['plan_sha256']
    with (out / 'submission.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        jobs = read(out / 'jobs.json') if (out / 'jobs.json').exists() else []
        if any(j['phase'] == phase and j['block'] == block for j in jobs):
            return
        stem = f'{phase}_b{block}'
        intent = out / f'intent_{stem}.json'
        if intent.exists():
            raise RuntimeError('Ambiguous submission requires receipt/tag reconciliation')
        _, backend = connect(c)
        assert backend.status().operational
        processor = backend.configuration().processor_type
        assert str(processor.get('revision')) == '1', processor
        properties = backend.properties(refresh=True).to_dict()
        ordered = [plan['rows'][i] for i in np.random.default_rng(
            c['seed']+block+(1000 if phase == 'confirm' else 0)).permutation(len(plan['rows']))]
        pubs = []
        for row in ordered:
            assert digest(ROOT / row['isa_qpy']) == row['isa_sha256']
            qc = qpy_read(ROOT / row['isa_qpy'])
            r = row['resources']
            check_isa(qc, backend.target, r['physical'], r['final_logical_mapping'])
            pubs.append((qc, None, plan['shots_per_circuit']))
        total = len(pubs)*plan['shots_per_circuit']
        assert len(jobs) < c['maximum_hardware_jobs']
        assert sum(j['total_shots'] for j in jobs)+total <= c['maximum_hardware_shots']
        sampler = SamplerV2(mode=backend)
        sampler.options.max_execution_time = c['maximum_execution_seconds_per_job']
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        sampler.options.environment.job_tags = ['aqc_larger_width_2026_10', stem, digest(planpath)[:12]]
        save(out / f'calibration_{stem}.json', properties)
        record = dict(phase=phase, block=block, rows=ordered, total_shots=total,
                      shots_per_circuit=plan['shots_per_circuit'], plan_sha256=digest(planpath),
                      backend=backend.name, backend_version=backend.backend_version,
                      processor=processor, calibration_timestamp=str(properties['last_update_date']))
        save(intent, dict(prepared_utc=now(), **record))
        job = sampler.run(pubs)
        save(out / f'receipt_{stem}.json', dict(received_utc=now(), job_id=job.job_id()))
        jobs.append(dict(submitted_utc=now(), job_id=job.job_id(), **record))
        save(out / 'jobs.json', jobs)
        print('Submitted', stem, job.job_id(), total, 'shots', flush=True)


def collect(phase):
    from .nighthawk_study import connect
    from .contribution_hardware_core import scalar_metrics
    c, out = settings()
    jobs = [j for j in read(out / 'jobs.json') if j['phase'] == phase]
    service, _ = connect(c)
    model = read(out / 'case.json')['model']
    for record in jobs:
        path = out / 'raw_counts' / f"{phase}_b{record['block']}.json"
        if path.exists():
            continue
        job = service.job(record['job_id'])
        status = str(job.status())
        print(phase, record['block'], status, flush=True)
        if status in ['ERROR', 'CANCELLED', 'JobStatus.ERROR', 'JobStatus.CANCELLED']:
            save(out / f"failure_{record['job_id']}.json", dict(status=status, error=job.error_message()))
            raise RuntimeError('Hardware failure retained; no automatic resubmission')
        if status not in ['DONE', 'JobStatus.DONE']:
            continue
        result = job.result()
        assert len(result) == len(record['rows'])
        rows = []
        for original, pub in zip(record['rows'], result):
            counts = pub.data.meas.get_counts()
            assert sum(counts.values()) == record['shots_per_circuit']
            rows.append(dict(method=original['method'], candidate_id=original['candidate_id'],
                             counts=counts, metrics=scalar_metrics(model, counts),
                             metadata=json.loads(json.dumps(pub.metadata, default=str))))
        save(path, dict(collected_utc=now(), phase=phase, block=record['block'],
                        job_id=record['job_id'], rows=rows,
                        job_metrics=json.loads(json.dumps(job.metrics(), default=str))))
    return len(jobs) == 2 and all((out / 'raw_counts' / f"{phase}_b{j['block']}.json").exists() for j in jobs)


def aggregate(phase):
    from .contribution_hardware_core import scalar_metrics
    c, out = settings()
    plan = read(out / f'{phase}_plan.json')
    model = read(out / 'case.json')['model']
    jobs = [j for j in read(out / 'jobs.json') if j['phase'] == phase]
    assert len(jobs) == 2 and {j['block'] for j in jobs} == {0, 1}
    index = {}
    hashes = {}
    for j in jobs:
        assert j['plan_sha256'] == digest(out / f'{phase}_plan.json')
        assert j['backend'] == c['backend'] and j['processor']['family'] == 'Nighthawk'
        path = out / 'raw_counts' / f"{phase}_b{j['block']}.json"
        raw = read(path)
        assert raw['job_id'] == j['job_id'] and raw['block'] == j['block']
        assert [r['method'] for r in raw['rows']] == [r['method'] for r in j['rows']]
        hashes[str(path.relative_to(ROOT))] = digest(path)
        for row, original in zip(raw['rows'], j['rows']):
            assert row['candidate_id'] == original['candidate_id']
            metric = scalar_metrics(model, row['counts'])
            assert metric == row['metrics'] and metric['shots'] == plan['shots_per_circuit']
            key = (row['method'], j['block'])
            assert key not in index
            index[key] = metric
    assert set(index) == {(r['method'], b) for r in plan['rows'] for b in [0, 1]}
    pooled = [dict(method=r['method'], shots=sum(index[r['method'], b]['shots'] for b in [0, 1]),
                   feasible_count=sum(index[r['method'], b]['feasible_count'] for b in [0, 1]),
                   optimal_count=sum(index[r['method'], b]['optimal_count'] for b in [0, 1]))
              for r in plan['rows']]
    return plan, index, pooled, hashes


def select():
    c, out = settings()
    assert not (out / 'confirmation_protocol.json').exists()
    verify_hashes(read(out / 'hardware_protocol.json'))
    plan, _, pooled, hashes = aggregate('screen')
    chosen = choose_screen(plan['rows'], pooled)
    rows = [dict(chosen, method='selected_native', screen_method=chosen['method'])]
    rows += [r for r in plan['rows'] if r['method'] in ['source', 'product', 'shallow']]
    assert len(rows) == 4
    save(out / 'screen_selection.json', dict(selected=chosen, pooled=pooled,
         selected_utc=now(), confirmation_outcomes_seen=False))
    save(out / 'confirm_plan.json', dict(rows=rows, blocks=c['confirmation_blocks'],
         shots_per_circuit=c['confirmation_shots'],
         total_shots=4*c['confirmation_blocks']*c['confirmation_shots']))
    inputs = dict(hashes)
    inputs[str((out / 'screen_selection.json').relative_to(ROOT))] = digest(out / 'screen_selection.json')
    save(out / 'confirmation_protocol.json', dict(frozen_utc=now(),
         input_sha256=inputs, plan_sha256=digest(out / 'confirm_plan.json'),
         confirmation_outcomes_seen=False, inference=c['inference']))
    print('Selected', chosen['method'], 'CZ depth', chosen['resources']['cz_depth'],
          '; independent confirmation frozen', flush=True)


def analyze():
    from .audit_tensor_prefix_statistics import conditional_two_sided, holm_adjust
    from .hardware_width_figure import block_mover_interval, block_difference_interval
    c, out = settings()
    verify_hashes(read(out / 'hardware_protocol.json'))
    verify_hashes(read(out / 'confirmation_protocol.json'))
    plan, index, pooled, hashes = aggregate('confirm')
    comparisons = []
    for event in ['feasible', 'optimal']:
        for control in ['source', 'product', 'shallow']:
            tables = [(index['selected_native', b][event+'_count'], index['selected_native', b]['shots'],
                       index[control, b][event+'_count'], index[control, b]['shots']) for b in [0, 1]]
            effect, low, high = block_mover_interval(tables)
            _, conservative_low, conservative_high = block_difference_interval(tables)
            comparisons.append(dict(event=event, reference=control,
                               difference=effect, low=low, high=high,
                               conservative_low=conservative_low, conservative_high=conservative_high,
                               p_value=conditional_two_sided(tables)))
    for row, p in zip(comparisons, holm_adjust([r['p_value'] for r in comparisons])):
        row['holm_p_value'] = p
        row['significant_gain'] = p < .05 and row['difference'] > 0
        row['significant_loss'] = p < .05 and row['difference'] < 0
    record = dict(completed_utc=now(), pooled=pooled, comparisons=comparisons,
                  methods=plan['rows'], raw_sha256=hashes,
                  scope=c['scope'], inference=c['inference'])
    save(out / 'analysis.json', record)
    lines = ['# Complete-anneal compression at 42 qubits', '',
             'FLP with six facilities and six customers. Independent Nighthawk confirmation.', '',
             '| Preparation | CZ depth | Feasible / shots | Optimal / shots |',
             '|---|---:|---:|---:|']
    bymethod = {r['method']:r for r in plan['rows']}
    for row in pooled:
        lines.append(f"| {row['method']} | {bymethod[row['method']]['resources']['cz_depth']} | "
                     f"{row['feasible_count']} / {row['shots']} | {row['optimal_count']} / {row['shots']} |")
    lines += ['', 'Six block-stratified comparisons share one Holm correction. '
              'Screening shots are excluded from inference.', '',
              'Target convergence uses direct MPS calculations and does not provide an exact-state certificate. '
              'This one-instance experiment does not establish a scaling law.']
    (out / 'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Independent confirmation analyzed:', pooled, flush=True)


def run():
    c, out = settings()
    with (out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out / 'hardware_protocol.json').exists():
            if not prepare():
                return
        for phase in ['screen', 'confirm']:
            if phase == 'confirm' and not (out / 'confirmation_protocol.json').exists():
                select()
            for block in [0, 1]:
                submit(phase, block)
            while not collect(phase):
                save(out / 'pipeline_status.json', dict(phase=phase, status='waiting_for_hardware',
                                                       updated_utc=now()))
                time.sleep(30)
        analyze()
        save(out / 'pipeline_status.json', dict(status='complete', completed_utc=now()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['freeze','preflight','target','target-selection','fit',
                                     'prepare','freeze-hardware','submit','collect','select','analyze','run'])
    p.add_argument('argument', nargs='?')
    p.add_argument('index', nargs='?', type=int)
    a = p.parse_args()
    if a.action == 'target': target_worker(int(a.argument))
    elif a.action == 'fit': fit_worker(int(a.argument), a.index)
    elif a.action == 'submit': submit(a.argument, a.index)
    elif a.action == 'collect': collect(a.argument)
    else:
        {'freeze':freeze, 'preflight':preflight, 'target-selection':target_selection,
         'prepare':prepare, 'freeze-hardware':freeze_hardware, 'select':select,
         'analyze':analyze, 'run':run}[a.action]()


if __name__ == '__main__':
    main()
