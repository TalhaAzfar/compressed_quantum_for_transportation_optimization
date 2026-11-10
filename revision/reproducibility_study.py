"""Fresh-instance preparation and one guarded queue for reproduction/diagnosis."""
import argparse
import fcntl
import importlib.metadata
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .contribution_hardware_core import ROOT, read, save, digest, now

CONFIG = ROOT / 'configs/reproducibility_2026_10.json'
OUT = ROOT / 'outputs/reproducibility_2026_10'
PARENT = ROOT / 'outputs/native_depth_refinement_2026_10'
EXTRA = ""


def settings():
    return read(CONFIG)


def preflight():
    from .nighthawk_study import connect
    from .tensor_prefix_core import path_patch
    from .native_aqc import edge_matchings
    assert not (OUT / 'preparation_protocol.json').exists()
    c = settings()
    _, backend = connect(c)
    assert backend.status().operational
    properties = backend.properties(refresh=True).to_dict()
    target = backend.target
    record = dict(observed_utc=now(), backend=backend.name,
                  processor=backend.configuration().processor_type,
                  operational=backend.status().operational,
                  backend_version=backend.backend_version,
                  calibration_timestamp=str(properties['last_update_date']))
    assert record['processor']['family'] == 'Nighthawk'
    save(OUT / 'preflight/device.json', record)
    save(OUT / 'preflight/properties.json', properties)
    (OUT / 'preflight/target.pkl').write_bytes(pickle.dumps(target))
    patches = {}
    for n in [16, 20]:
        physical, edges, diagnostic = path_patch(target, n, c['patch_seed'])
        patches[str(n)] = dict(physical=physical, edges=edges, diagnostic=diagnostic,
                              matchings=len(edge_matchings(edges)))
    save(OUT / 'preflight/patches.json', patches)
    print(record, flush=True)
    print('Physical patch matchings:', {n: p['matchings'] for n, p in patches.items()}, flush=True)


def freeze():
    c = settings()
    assert len(c['cases']) == 6 and len({s['seed'] for s in c['cases']}) == 6
    assert not {s['seed'] for s in c['cases']} & {s['seed'] for s in read(ROOT/'configs/native_depth_refinement.json')['cases']}
    names = [str(CONFIG.relative_to(ROOT)), 'revision/reproducibility_study.py',
             'revision/reproducibility_diagnostics.py', 'tests/test_reproducibility.py',
             'revision/native_depth_refinement_v2.py', 'revision/native_depth_objective_v2.py',
             'revision/native_depth_objective.py', 'revision/cpu_memory_guard.py',
             'revision/hardware_width_sweep.py', 'revision/width_sweep_models.py',
             'revision/tensor_prefix_models.py', 'revision/tensor_prefix_core.py',
             'revision/tensor_prefix_sabre.py', 'revision/native_aqc.py',
             'revision/nighthawk_study.py', 'revision/corrected_study.py',
             'revision/steps_native_sweep.py', 'revision/depth200_native_cleanup.py',
             'revision/full_circuit_compression.py', 'revision/contribution_hardware_core.py']
    inputs = [OUT/'preflight'/n for n in ['device.json', 'properties.json', 'target.pkl', 'patches.json']]
    inputs += [PARENT/'release_checksums.json', OUT/'preparation_tests.log']
    tests = (OUT/'preparation_tests.log').read_text()
    assert 'passed' in tests and 'FAILED' not in tests and 'ERROR' not in tests
    record = dict(config=c, source_sha256={n: digest(ROOT/n) for n in names},
                  input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
                  packages={n: importlib.metadata.version(n) for n in ['quimb','torch','qiskit','qiskit-aer','qiskit-addon-aqc-tensor','cotengra']})
    path = OUT/'preparation_protocol.json'
    if path.exists():
        assert all(read(path)[k] == v for k, v in record.items()), 'Frozen preparation changed'
    else:
        for name in names:
            shutil.copy2(ROOT/name, OUT/'source_snapshot'/Path(name).name)
        save(path, dict(frozen_utc=now(), new_hardware_outcomes_seen=False, **record))


def check_frozen():
    protocol = read(OUT/'preparation_protocol.json')
    for field in ['source_sha256', 'input_sha256']:
        assert all(digest(ROOT/p) == h for p, h in protocol[field].items()), 'Frozen input changed'


def depth_layers(matchings):
    assert matchings > 0
    return [('product', 0, 0), ('base', 2, 4*matchings)] + [
        (f'd{cap}', cap//(2*matchings), cap) for cap in settings()['new_native_depth_caps']]


def worker(cid):
    import numpy as np
    from .width_sweep_models import make_model, validate_model
    from .hardware_width_sweep import exact_state, dense_target, state_events
    from .nighthawk_study import evolution, qpy_write
    from .tensor_prefix_models import numpy_model
    from .tensor_prefix_core import native_ansatz
    from .tensor_prefix_sabre import compile_isa_sabre
    from .steps_native_sweep import extend_parameters
    from .native_depth_objective_v2 import make_objective
    from .native_depth_refinement_v2 import fit_one
    from .full_circuit_compression import mps_dense_qiskit
    from .depth200_native_cleanup import cleanup, cleaned_resources
    check_frozen()
    c = settings(); folder = OUT/'cases'/cid
    folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():
        return
    spec = next(s for s in c['cases'] if s['id'] == cid)
    model = make_model(spec); n = model['n_qubits']
    save(folder/'model.json', model)
    save(folder/'model_validation.json', validate_model(model))
    patch = read(OUT/'preflight/patches.json')[str(n)]
    edges, physical = patch['edges'], patch['physical']
    backend = pickle.loads((OUT/'preflight/target.pkl').read_bytes())
    source = evolution(numpy_model(model), c['annealing_time'], c['steps'])
    if (folder/'reference.npy').exists():
        reference = np.load(folder/'reference.npy')
    else:
        reference = exact_state(source); np.save(folder/'reference.npy', reference)
    assert abs(np.vdot(reference, reference).real-1) < 1e-8
    if (folder/'target.pkl').exists():
        target = pickle.loads((folder/'target.pkl').read_bytes())
    else:
        target = dense_target(reference, n, c['target_bond'])
        target._psi /= np.sqrt(float(abs(target.psi.H@target.psi)))
        (folder/'target.pkl').write_bytes(pickle.dumps(target))
    target_dense = mps_dense_qiskit(target, n)
    target_dense /= np.linalg.norm(target_dense)
    target_fidelity = float(abs(np.vdot(reference, target_dense))**2)
    assert target_fidelity >= c['minimum_target_fidelity']
    save(folder/'target_validation.json', dict(target_exact_fidelity=target_fidelity,
        reference_sha256=digest(folder/'reference.npy'), target_sha256=digest(folder/'target.pkl'),
        construction='exact-reference-assisted MPS', max_bond=c['target_bond']))

    def circuit_row(name, bound, layers, fit, events, cap=None):
        logical = OUT/'circuits'/(name+'.qpy'); isa_path = OUT/'circuits'/(name+'_isa.qpy')
        qpy_write(logical, bound)
        isa, resources = compile_isa_sabre(bound, backend, physical, edges)
        if layers is not None:
            assert resources['swap_count'] == 0
        isa, certificate = cleanup(isa)
        resources = cleaned_resources(isa, resources, backend, certificate)
        if cap is not None:
            assert resources['cz_depth'] <= cap
        qpy_write(isa_path, isa)
        return dict(id=name, candidate_id=name, case_id=cid, family=model['problem'].upper(),
                    n_qubits=n, layers=layers, exact_fidelity=fit['exact_fidelity'] if fit else 1.,
                    ideal_events=events, fit=fit, resources=resources, cz_depth_cap=cap,
                    logical_qpy=str(logical.relative_to(ROOT)), logical_sha256=digest(logical),
                    isa_qpy=str(isa_path.relative_to(ROOT)), isa_sha256=digest(isa_path))

    path = folder/'source_result.json'
    if not path.exists():
        save(path, circuit_row(cid+'_source', source, None, None, state_events(reference, model)))
    source_row = read(path)
    rows = []
    for label, layers, cap in depth_layers(patch['matchings']):
        ansatz = native_ansatz(n, edges, layers); objective = None
        for start in range(3):
            name = f'{cid}_{label}_r{start}'; path = folder/(name+'.json')
            if path.exists():
                rows.append(read(path)); continue
            if objective is None:
                warm_path = folder/(label+'_contraction.json')
                objective, contraction = make_objective(target, ansatz,
                    repeats=c['contraction_repeats'], max_tensor_bytes=c['max_tensor_bytes'],
                    max_slice_write_bytes=c['max_slice_write_bytes'], max_slices=c['max_slices'],
                    seed=c['parameter_seed']+layers, warm_path_file=warm_path)
                save(warm_path, contraction)
                print(cid, label, 'slices', contraction['slices'], flush=True)
            available = [r for r in rows if r['layers'] <= layers and (start != 2 or r['layers'] == 0)]
            old = max(available, key=lambda r: r['exact_fidelity']) if available else None
            x = (extend_parameters(old['fit']['parameters'], n, len(edges), old['layers'], layers)
                 if old else np.zeros(ansatz.num_parameters))
            perturbation = c['start_perturbations'][start]
            if perturbation:
                x += np.random.default_rng(c['parameter_seed']+spec['seed']+100*layers+start).normal(0, perturbation, len(x))
            initialization = dict(source_candidate=old['id'] if old else 'all_plus',
                                  perturbation=perturbation, identity_extension=perturbation == 0)
            bound, fit = fit_one(objective, ansatz, x, c, folder/(name+'_checkpoint.json'),
                                reference=reference, target_dense=target_dense, model=model)
            row = circuit_row(name, bound, layers, fit, fit['ideal_events'], cap)
            row.update(label=label, start=start, initialization=initialization, completed_utc=now())
            save(path, row); rows.append(row)
            print(name, 'F', fit['exact_fidelity'], 'feas', fit['ideal_events']['p_feasible'],
                  'evaluations', fit['evaluations'], 'adequate', fit['adequate_evaluation_budget'], flush=True)
        del objective
    assert len(rows) == 12 and all(r['fit']['verified'] for r in rows)
    save(folder/'complete.json', dict(completed_utc=now(), case_id=cid, model=model, patch=patch,
         source=source_row, rows=rows, all_fits_verified=True,
         adequate_fits=sum(r['fit']['adequate_evaluation_budget'] for r in rows),
         protocol_sha256=digest(OUT/'preparation_protocol.json')))


def run():
    import psutil
    from .cpu_memory_guard import physical_cpu_ids
    lock = (OUT/'preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze(); c = settings(); cpus = physical_cpu_ids()
    assert len(cpus) >= c['parallel_workers']*c['physical_cores_per_worker']
    tasks = []
    # One diagnostic begins alongside fitting; all workers share this watchdog.
    diagnostics = [f'{family}_r{i}' for family in ['tsp25','vrp20'] for i in range(1,4)]
    tasks.append(('diagnostic', diagnostics.pop(0)))
    tasks += [('fit', s['id']) for s in c['cases']]
    tasks += [('diagnostic', cid) for cid in diagnostics]
    def result_path(kind, cid):
        return OUT/('cases' if kind == 'fit' else 'diagnostics')/cid/'complete.json'
    tasks = [(kind, cid) for kind, cid in tasks if not result_path(kind, cid).exists()]
    active = {}; finished = {}; peak = 0; started = time.monotonic()
    try:
        while tasks or active:
            for slot in range(c['parallel_workers']):
                if slot in active or not tasks: continue
                if psutil.virtual_memory().available < c['system_reserve_bytes']+10**9: break
                kind, cid = tasks.pop(0)
                cores = cpus[slot*c['physical_cores_per_worker']:(slot+1)*c['physical_cores_per_worker']]
                module = 'revision.reproducibility_study' if kind == 'fit' else 'revision.reproducibility_diagnostics'
                command = [sys.executable, '-m', 'revision.cpu_memory_guard', '--address-gib', str(c['worker_address_gib']),
                    '--cpus', ','.join(map(str, cores)), '--threads', str(c['numerical_threads_per_worker']), '--',
                    sys.executable, '-m', 'revision.local_runner', '--extra-site', EXTRA, module, 'worker', cid]
                log = (OUT/'logs'/f'{kind}_{cid}.log').open('a')
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active[slot] = dict(child=child, log=log, kind=kind, case_id=cid, cpus=cores, peak_rss=0, started=time.monotonic())
                print('Started', kind, cid, 'PID', child.pid, 'CPUs', cores, flush=True)
            total = psutil.Process().memory_info().rss
            for slot, item in list(active.items()):
                child = item['child']
                if child.poll() is not None:
                    item['log'].close()
                    ok = child.returncode == 0 and result_path(item['kind'], item['case_id']).exists()
                    record = dict(status='complete' if ok else 'failed', returncode=child.returncode,
                        seconds=time.monotonic()-item['started'], cpu_ids=item['cpus'], peak_worker_rss_bytes=item['peak_rss'])
                    key = item['kind']+'_'+item['case_id']; finished[key] = record
                    save(OUT/'logs'/(key+'_process.json'), record)
                    del active[slot]
                    if not ok: raise RuntimeError(f'{key} failed; inspect retained logs before continuing')
                    continue
                rss = 0
                try:
                    proc = psutil.Process(child.pid)
                    for member in [proc]+proc.children(recursive=True):
                        try: rss += member.memory_info().rss
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                total += rss; item['peak_rss'] = max(item['peak_rss'], rss)
                assert rss <= c['worker_rss_limit_bytes'], 'Worker RSS guard'
                assert time.monotonic()-item['started'] < 12*3600, 'Worker wall-time guard'
            peak = max(peak, total)
            assert total <= c['aggregate_rss_limit_bytes'], 'Aggregate 35 GB RSS guard'
            assert psutil.virtual_memory().available >= c['system_reserve_bytes'], 'System memory reserve'
            save(OUT/'preparation_progress.json', dict(updated_utc=now(), finished=finished,
                active=[dict(kind=i['kind'], case_id=i['case_id'], pid=i['child'].pid, cpu_ids=i['cpus'],
                             seconds=time.monotonic()-i['started'], peak_rss_bytes=i['peak_rss']) for i in active.values()],
                pending=tasks, peak_total_rss_bytes=peak, elapsed_seconds=time.monotonic()-started))
            time.sleep(1)
        cases = [read(OUT/'cases'/s['id']/'complete.json') for s in c['cases']]
        assert all(r['adequate_fits'] == 12 and r['all_fits_verified'] for r in cases)
        save(OUT/'preparation_complete.json', dict(completed_utc=now(), cases=6, verified_fits=72,
            larger_diagnostic_cases=6, peak_total_rss_bytes=peak, protocol_sha256=digest(OUT/'preparation_protocol.json')))
    finally:
        for item in active.values():
            try: os.killpg(item['child'].pid, signal.SIGKILL)
            except ProcessLookupError: pass
            item['child'].wait(); item['log'].close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['preflight','freeze','run','worker'])
    p.add_argument('case', nargs='?'); a = p.parse_args()
    try:
        worker(a.case) if a.action == 'worker' else globals()[a.action]()
    except BaseException as error:
        if a.action == 'run':
            save(OUT/'preparation_stop.json', dict(stopped_utc=now(), type=type(error).__name__, reason=str(error)))
        raise
