"""Bounded FLP42 native-depth extension with block coordinate AQC-Tensor fits.

The existing pilot's audited hardware/inference routines are reused in memory;
neither the original program nor its frozen experiment files are modified.
"""
import argparse
import fcntl
import importlib.metadata
import os
import pickle
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from .contribution_hardware_core import ROOT, read, save, digest, now

CONFIG = ROOT / 'configs/flp42_depth_extension_2026_10.json'
EXTRA = ""


def settings():
    c = read(CONFIG)
    return c, ROOT / c['output_dir']


def runner():
    return [sys.executable, '-m', 'revision.local_runner', '--extra-site', EXTRA,
            'revision.flp42_depth_extension']


def pilot():
    from . import larger_width_feasibility as base
    base.settings = settings
    base.runner = runner
    return base


def split_block(ansatz, n, edge_count, first, last):
    """Factor H R0 (E0 R1) ... into fixed prefix, variable block, suffix."""
    from qiskit import QuantumCircuit
    if not 0 <= first < last:
        raise ValueError('Invalid layer block')
    leading = 3*n
    stride = 3*edge_count+2*n
    left, right = leading+first*stride, leading+last*stride
    if right > len(ansatz.data):
        raise ValueError('Block exceeds ansatz')
    parts = []
    for data in [ansatz.data[:left], ansatz.data[left:right], ansatz.data[right:]]:
        q = QuantumCircuit(n)
        for inst in data:
            q.append(inst.operation, [ansatz.find_bit(b).index for b in inst.qubits])
        parts.append(q)
    parts[0].global_phase = ansatz.global_phase
    return parts


def bind_subset(qc, full_ansatz, values):
    mapping = dict(zip(full_ansatz.parameters, values))
    return qc.assign_parameters({p: mapping[p] for p in qc.parameters})


def block_settings(input_state, bond, cutoff):
    """Use the cached input in AQC-Tensor's explicit-gradient context.

    The library overrides psi0 when applying a circuit to a supplied state,
    so inverse-target propagation still begins at that supplied target.
    """
    from functools import partial
    import quimb.tensor as qtn
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    return QuimbSimulator(partial(qtn.CircuitMPS, psi0=input_state.psi,
                                  max_bond=bond, cutoff=cutoff),
                          autodiff_backend='explicit')


def validate_resources(c):
    import psutil
    from .cpu_memory_guard import physical_cpu_ids
    if c['workers']*c['threads_per_worker'] > len(physical_cpu_ids()):
        raise RuntimeError('Requested more physical cores than available')
    if c['aggregate_rss_gib'] > 35:
        raise RuntimeError('Combined workers exceed the user RAM budget')
    return dict(observed_utc=now(), total_gib=psutil.virtual_memory().total/2**30,
                available_gib=psutil.virtual_memory().available/2**30,
                physical_cores=len(physical_cpu_ids()), logical_cpus=psutil.cpu_count(),
                workers=c['workers'], threads_per_worker=c['threads_per_worker'],
                aggregate_rss_limit_gib=c['aggregate_rss_gib'],
                system_reserve_gib=c['system_reserve_gib'])


def freeze():
    c, out = settings()
    parent = ROOT/c['parent_dir']
    for name in ['fits', 'circuits', 'targets', 'checkpoints', 'logs', 'preflight',
                 'raw_counts', 'source_snapshot']:
        (out/name).mkdir(parents=True, exist_ok=True)
    sources = ['configs/flp42_depth_extension_2026_10.json',
               'revision/flp42_depth_extension.py', 'revision/larger_width_feasibility.py',
               'revision/flp42_fidelity_parallel.py', 'revision/tensor_prefix_core.py',
               'revision/steps_native_sweep.py', 'revision/cpu_memory_guard.py',
               'revision/full_circuit_explicit.py', 'revision/flp_mps_events.py',
               'revision/tensor_prefix_sabre.py', 'revision/contribution_hardware_core.py',
               'revision/nighthawk_study.py', 'revision/local_runner.py',
               'revision/audit_tensor_prefix_statistics.py', 'revision/hardware_width_figure.py',
               'tests/test_flp42_depth_extension.py']
    names = ['case.json', 'targets/selection.json', 'targets/b512.pkl',
             'targets/b512.json', 'preflight/patch.json', 'circuits/source.qpy']
    names += [str(p.relative_to(parent)) for p in sorted((parent/'fits').glob('*.json'))]
    names += [str(p.relative_to(parent)) for p in sorted((parent/'circuits').glob('l*_r*.qpy'))]
    inputs = [parent/name for name in names]
    inputs += [out/'tests.log']
    tests = (out/'tests.log').read_text()
    assert 'passed' in tests and 'FAILED' not in tests and 'ERROR' not in tests
    record = dict(config=c, source_sha256={p:digest(ROOT/p) for p in sources},
                  input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs})
    path = out/'preparation_protocol.json'
    if path.exists():
        old = read(path)
        assert all(old[k] == v for k,v in record.items()), 'Frozen preparation changed'
        pilot().verify_hashes(old)
        return
    assert read(parent/'targets/selection.json')['status'] == 'converged'
    record.update(frozen_utc=now(), prior_hardware_outcomes_known=True,
                  new_hardware_outcomes_seen=False,
                  packages={p:importlib.metadata.version(p) for p in
                            ['qiskit', 'quimb', 'qiskit-addon-aqc-tensor',
                             'qiskit-ibm-runtime', 'numpy', 'scipy']})
    for name in names:
        destination = out/name
        if destination.exists():
            assert digest(destination) == digest(parent/name)
        else:
            shutil.copy2(parent/name, destination)
    for source in sources:
        shutil.copy2(ROOT/source, out/'source_snapshot'/Path(source).name)
    save(out/'resources_at_start.json', validate_resources(c))
    save(path, record)
    print('Frozen FLP42 extension; original data retained; maximum CZ depth 200.', flush=True)


def preflight():
    from .nighthawk_study import connect
    c, out = settings()
    _, backend = connect(c)
    assert backend.status().operational
    processor = backend.configuration().processor_type
    assert str(processor.get('revision')) == '1'
    patch = read(out/'preflight/patch.json')
    live = {tuple(sorted(e)) for e in backend.target['cz']}
    assert all(tuple(sorted((patch['physical'][a], patch['physical'][b]))) in live
               for a,b in patch['edges']), 'Original patch connectivity changed'
    properties = backend.properties(refresh=True).to_dict()
    save(out/'preflight/device.json', dict(observed_utc=now(), backend=backend.name,
         processor=processor, backend_version=backend.backend_version,
         calibration_timestamp=str(properties['last_update_date'])))
    save(out/'preflight/properties.json', properties)
    (out/'preflight/target.pkl').write_bytes(pickle.dumps(backend.target))
    print('Live original-patch preflight:', backend.name, processor, flush=True)


def fit_worker(layers, index):
    import numpy as np
    from scipy.optimize import minimize
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation import apply_circuit_to_state
    from .tensor_prefix_core import native_ansatz, fidelity
    from .flp42_fidelity_parallel import state
    from .full_circuit_explicit import explicit_settings
    from .steps_native_sweep import extend_parameters
    from .flp_mps_events import probabilities
    from .nighthawk_study import qpy_write
    c, out = settings()
    name = f'l{layers}_r{index}'
    started, cpu_started = time.monotonic(), time.process_time()
    assert index in [2,3]
    patch = read(out/'preflight/patch.json')
    n, edges = 42, patch['edges']
    target = pickle.loads((out/'targets/b512.pkl').read_bytes())
    target._psi /= np.sqrt(float(abs(target.psi.H@target.psi)))
    ansatz = native_ansatz(n, edges, layers)
    candidates = [read(p) for p in (out/'fits').glob('*.json')]
    candidates = [r for r in candidates if r.get('verified') and
                  (r['layers'] < layers or (r['layers'] == layers and r['index'] < 2))]
    seed = max(candidates, key=lambda r:(r['fidelity'], -r['layers'], -r['index']))
    x = extend_parameters(seed['parameters'], n, len(edges), seed['layers'], layers)
    if index == 3:
        x += np.random.default_rng(c['seed']+layers).normal(0,.005,len(x))
    parameter_indices = {p:i for i,p in enumerate(ansatz.parameters)}
    low = state(ansatz.assign_parameters(x), c['fit_bond'], c['cutoff'])
    high = state(ansatz.assign_parameters(x), c['check_bond'], c['cutoff'])
    best_fidelity = fidelity(target, high)
    initial_fidelity = best_fidelity
    best_x = x.copy()
    history, updates = [], []
    width = c['block_layers']
    blocks = [(a,min(layers,a+width)) for a in range(0,layers,width)]
    # Train added layers first, then sweep back through the existing circuit.
    schedule = list(reversed(blocks))
    for sweep in range(1,c['sweeps']):
        schedule += blocks if sweep%2 else list(reversed(blocks))
    class BudgetStop(Exception):
        pass
    status = 'completed_block_sweeps'
    print(name, 'initial F', best_fidelity, 'seed L', seed['layers'], flush=True)
    for step,(first,last) in enumerate(schedule):
        if time.monotonic()-started >= c['fit_seconds']:
            status = 'wall_budget_best_verified_update'
            break
        prefix, block, suffix = split_block(ansatz,n,len(edges),first,last)
        block_input = state(bind_subset(prefix,ansatz,x), c['fit_bond'], c['cutoff'])
        block_input._psi /= np.sqrt(float(abs(block_input.psi.H@block_input.psi)))
        effective_target = apply_circuit_to_state(bind_subset(suffix,ansatz,x).inverse(),
                             target, explicit_settings(c['fit_bond'],c['cutoff']))
        effective_target._psi /= np.sqrt(float(abs(effective_target.psi.H@effective_target.psi)))
        objective = MaximizeStateFidelity(effective_target,block,
                      block_settings(block_input,c['fit_bond'],c['cutoff']))
        indices = [parameter_indices[p] for p in block.parameters]
        local_best = dict(loss=float('inf'), parameters=x[indices].copy())
        evaluations = 0
        def fun(theta):
            nonlocal evaluations
            if evaluations and (evaluations >= c['fit_max_evaluations_per_block'] or
                                time.monotonic()-started >= c['fit_seconds']):
                raise BudgetStop()
            loss, gradient = objective.loss_function(theta)
            loss, gradient = float(loss), np.asarray(gradient,float)
            if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
                raise ValueError('Nonfinite block objective or gradient')
            evaluations += 1
            if loss < local_best['loss']:
                local_best.update(loss=loss, parameters=np.array(theta))
            history.append(dict(step=step,block=[first,last],evaluation=evaluations,
                                seconds=time.monotonic()-started,loss=loss))
            save(out/'checkpoints'/f'{name}.json',dict(updated_utc=now(),
                 parameters=best_x.tolist(),fidelity=best_fidelity,step=step,
                 active_block=[first,last],history=history,
                 active_best_parameters=local_best['parameters'].tolist(),
                 active_best_loss=local_best['loss']))
            print(name,'block',first,last,'eval',evaluations,'F_block',1-loss,flush=True)
            return loss,gradient
        try:
            result = minimize(fun,x[indices],jac=True,method='L-BFGS-B',
                        options=dict(maxiter=c['fit_maxiter_per_block'],maxls=8,
                                     ftol=1e-10,gtol=1e-7))
            block_status = str(result.message)
        except BudgetStop:
            block_status = 'block_evaluation_or_wall_budget'
        candidate = x.copy()
        candidate[indices] = local_best['parameters']
        candidate_state = state(ansatz.assign_parameters(candidate),c['check_bond'],c['cutoff'])
        f = fidelity(target,candidate_state)
        accepted = f > best_fidelity
        if accepted:
            x, best_x, best_fidelity = candidate, candidate.copy(), f
        updates.append(dict(step=step,block=[first,last],evaluations=evaluations,
                            status=block_status,fidelity=f,accepted=accepted,
                            retained_fidelity=best_fidelity))
        save(out/'checkpoints'/f'{name}_updates.json',dict(updated_utc=now(),
             parameters=best_x.tolist(),fidelity=best_fidelity,updates=updates))
        print(name,'completed block',first,last,'full F',f,'accepted',accepted,flush=True)
        del objective, effective_target, block_input, candidate_state
    bound = ansatz.assign_parameters(best_x)
    low = state(bound,c['fit_bond'],c['cutoff'])
    high = state(bound,c['check_bond'],c['cutoff'])
    convergence = fidelity(low,high)
    f_low, f_high = fidelity(target,low), fidelity(target,high)
    difference = abs(f_low-f_high)
    verified = convergence >= c['minimum_ansatz_overlap'] and difference < c['objective_tolerance']
    record = dict(layers=layers,index=index,parameters=best_x.tolist(),fidelity=f_high,
                  initial_fidelity=initial_fidelity,objective_fidelity=f_low,
                  ansatz_bond_convergence=convergence,objective_difference=difference,
                  verified=bool(verified),status=status,evaluations=len(history),
                  updates=updates,history=history,initialization=dict(layers=seed['layers'],
                     index=seed['index'],fidelity=seed['fidelity'],perturbed=index==3),
                  target_bond=512,fit_bond=c['fit_bond'],check_bond=c['check_bond'],
                  fitting_method='AQC-Tensor block coordinate explicit MPS gradient',
                  fitting_seconds=time.monotonic()-started,seconds=time.monotonic()-started,
                  cpu_seconds=time.process_time()-cpu_started,completed_utc=now())
    qpy_write(out/'circuits'/f'{name}.qpy',bound)
    if verified:
        record['ideal_events'] = probabilities(read(out/'case.json')['model'],high)
    save(out/'fits'/f'{name}.json',record)
    print(name,'finished F',f_high,'verified',verified,'evaluations',len(history),flush=True)


def prepare():
    c,out = settings()
    base = pilot()
    base.verify_hashes(read(out/'preparation_protocol.json'))
    assert (out/'preflight/target.pkl').exists(), 'Live original-patch preflight required'
    for layers in c['layers']:
        worker_pool([dict(name=f'fit_l{layers}_r{i}',result=f'fits/l{layers}_r{i}.json',
                         args=['fit',str(layers),str(i)],seconds=c['fit_seconds']+1800)
                     for i in [2,3]])
    freeze_hardware()


def worker_pool(tasks):
    """Two workers on balanced physical-core groups, with one RAM watchdog."""
    import psutil
    from .cpu_memory_guard import physical_cpu_ids
    c,out = settings()
    cpus = physical_cpu_ids()[:c['workers']*c['threads_per_worker']]
    available = [cpus[w::c['workers']] for w in range(c['workers'])]
    assert all(len(group) == c['threads_per_worker'] for group in available)
    pending,active,peak = list(tasks),[],0
    try:
        while pending or active:
            for task,cpu,proc,log,started in list(active):
                if proc.poll() is None:
                    continue
                log.close()
                save(out/'logs'/f"{task['name']}_process.json",dict(
                     returncode=proc.returncode,seconds=time.monotonic()-started,
                     cpus=cpu,threads=c['threads_per_worker']))
                if not (out/task['result']).exists():
                    save(out/task['result'],dict(status='worker_failed',verified=False,
                         task=task,returncode=proc.returncode))
                print('Finished',task['name'],'return code',proc.returncode,flush=True)
                active.remove((task,cpu,proc,log,started))
                available.append(cpu)
            while pending and available:
                task = pending.pop(0)
                if (out/task['result']).exists():
                    continue
                cpu = available.pop(0)
                command = [sys.executable,'-m','revision.cpu_memory_guard',
                           '--address-gib',str(c['address_gib']),'--cpus',
                           ','.join(map(str,cpu)),'--threads',str(c['threads_per_worker']),
                           '--',*runner(),*task['args']]
                log = (out/'logs'/f"{task['name']}.log").open('a')
                proc = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                        start_new_session=True)
                active.append((task,cpu,proc,log,time.monotonic()))
                print('Launched',task['name'],'CPUs',cpu,flush=True)
            rss,observations = 0,[]
            for task,cpu,proc,log,started in active:
                if time.monotonic()-started > task['seconds']:
                    raise RuntimeError('Worker exceeded time budget: '+task['name'])
                try:
                    tree = [psutil.Process(proc.pid)]
                    tree += tree[0].children(recursive=True)
                    for p in tree:
                        with p.oneshot():
                            resident = p.memory_info().rss
                            rss += resident
                            observations.append(dict(pid=p.pid,rss_gib=resident/2**30,
                                threads=p.num_threads(),cpu_seconds=sum(p.cpu_times()[:2]),
                                affinity=p.cpu_affinity()))
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak,rss)
            memory = psutil.virtual_memory()
            save(out/'progress.json',dict(updated_utc=now(),
                 running=[t['name'] for t,*_ in active],pending=len(pending),
                 rss_gib=rss/2**30,peak_rss_gib=peak/2**30,
                 available_system_gib=memory.available/2**30,processes=observations))
            if rss > c['aggregate_rss_gib']*2**30:
                raise RuntimeError('Aggregate RAM watchdog reached')
            if memory.available < c['system_reserve_gib']*2**30:
                raise RuntimeError('System memory reserve reached')
            time.sleep(.5)
    finally:
        for _,_,proc,log,_ in active:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            log.close()


def freeze_hardware():
    from .nighthawk_study import qpy_read,qpy_write
    from .tensor_prefix_sabre import compile_isa_sabre
    from .contribution_hardware_core import check_isa
    c,out = settings()
    base = pilot()
    parent = ROOT/c['parent_dir']
    assert not (out/'hardware_protocol.json').exists()
    records = base.best_per_depth([read(p) for p in (out/'fits').glob('*.json')])
    assert set(c['layers']) <= {r['layers'] for r in records}, 'Missing verified depth'
    original = {r['method']:r for r in read(parent/'screen_plan.json')['rows']}
    target = pickle.loads((out/'preflight/target.pkl').read_bytes())
    patch = read(out/'preflight/patch.json')
    items = [('source',None,out/'circuits/source.qpy')]
    for control in ['product','shallow']:
        row = original[control]
        record = read(parent/'fits'/f"{row['candidate_id']}.json")
        items.append((control,record,ROOT/row['logical_qpy']))
    for r in records:
        if r['layers']:
            items.append((f"aqc_l{r['layers']}",r,
                          out/'circuits'/f"l{r['layers']}_r{r['index']}.qpy"))
    rows = []
    for method,record,logical in items:
        isa,resource = compile_isa_sabre(qpy_read(logical),target,patch['physical'],
                                       patch['edges'],seed=c['transpile_seed'])
        checked = check_isa(isa,target,patch['physical'],resource['final_logical_mapping'])
        assert all(checked[k] == resource[k] for k in checked)
        if record is not None:
            assert resource['swap_count'] == 0 and resource['cz_depth'] <= 200
        destination = out/'circuits'/f'{method}_isa.qpy'
        qpy_write(destination,isa)
        rows.append(dict(method=method,candidate_id=logical.stem,
             layers=None if record is None else record['layers'],
             fidelity=1. if record is None else record['fidelity'],
             ideal_events=read(out/'targets/b512.json')['events'] if record is None
                          else record['ideal_events'],
             logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=digest(logical),
             isa_qpy=str(destination.relative_to(ROOT)),isa_sha256=digest(destination),
             resources=resource))
    plan = dict(rows=rows,blocks=c['screen_blocks'],shots_per_circuit=c['screen_shots'],
                total_shots=len(rows)*c['screen_blocks']*c['screen_shots'])
    assert plan['total_shots']+4*c['confirmation_blocks']*c['confirmation_shots'] <= c['maximum_hardware_shots']
    save(out/'screen_plan.json',plan)
    inputs = [out/'case.json',out/'preparation_protocol.json',out/'preflight/target.pkl',
              out/'preflight/patch.json',out/'preflight/device.json',out/'tests.log']
    inputs += list((out/'targets').glob('*.json'))+list((out/'fits').glob('*.json'))
    for row in rows:
        inputs += [ROOT/row['logical_qpy'],ROOT/row['isa_qpy']]
    save(out/'hardware_protocol.json',dict(frozen_utc=now(),
         source_sha256=read(out/'preparation_protocol.json')['source_sha256'],
         input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
         screen_plan_sha256=digest(out/'screen_plan.json'),
         primary_inference=c['inference'],confirmation_outcomes_seen=False))
    save(out/'preparation_status.json',dict(status='hardware_ready',completed_utc=now()))
    print('Frozen',len(rows),'screening circuits;',plan['total_shots'],'screen shots',flush=True)


def analyze():
    from .audit_tensor_prefix_statistics import holm_adjust
    c,out = settings()
    pilot().analyze()
    record = read(out/'analysis.json')
    rows = {r['method']:r for r in record['methods']}
    for contrast in record['comparisons']:
        ref = rows[contrast['reference']]
        same = rows['selected_native']['logical_sha256'] == ref['logical_sha256']
        # Matching candidate IDs also catches copied original preparations.
        same = same or rows['selected_native']['candidate_id'] == ref['candidate_id']
        contrast['identical_circuit_self_control'] = same
        if same:
            contrast['p_value'] = 1.
    corrected = holm_adjust([r['p_value'] for r in record['comparisons']])
    for row,p in zip(record['comparisons'],corrected):
        row.update(holm_p_value=p,significant_gain=p < .05 and row['difference'] > 0,
                   significant_loss=p < .05 and row['difference'] < 0)
    save(out/'analysis.json',record)


def run():
    c,out = settings()
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (out/'hardware_protocol.json').exists():
            prepare()
        base = pilot()
        for phase in ['screen','confirm']:
            if phase == 'confirm' and not (out/'confirmation_protocol.json').exists():
                base.select()
            for block in [0,1]:
                base.submit(phase,block)
            while not base.collect(phase):
                save(out/'pipeline_status.json',dict(status='waiting_for_hardware',
                     phase=phase,updated_utc=now()))
                time.sleep(30)
        analyze()
        save(out/'pipeline_status.json',dict(status='complete',completed_utc=now()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','preflight','prepare','fit','run','submit',
                                   'collect','select','analyze'])
    p.add_argument('argument',nargs='?')
    p.add_argument('index',nargs='?',type=int)
    a = p.parse_args()
    if a.action == 'fit':
        fit_worker(int(a.argument),a.index)
    elif a.action == 'submit':
        pilot().submit(a.argument,a.index)
    elif a.action == 'collect':
        pilot().collect(a.argument)
    elif a.action == 'select':
        pilot().select()
    else:
        globals()[a.action]()


if __name__ == '__main__':
    main()
