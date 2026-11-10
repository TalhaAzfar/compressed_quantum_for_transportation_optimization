"""Prospective full-anneal compression sweep, with bounded parallel workers."""
import argparse
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .contribution_hardware_core import ROOT, now, save, read, digest

CONFIG = ROOT/'configs/hardware_width_sweep.json'
OUT = ROOT/'outputs/hardware_width_sweep_2026'
EXTRA = ""
SOURCE_HASH = digest(Path(__file__))


def caused_by(exc, kind):
    """Numba may wrap a Python alarm exception in one or more SystemErrors."""
    pending = [exc]; seen = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen: continue
        if isinstance(current, kind): return True
        seen.add(id(current)); pending.extend([current.__cause__, current.__context__])
    return False


def settings():
    return read(CONFIG)


def freeze():
    import importlib.metadata
    c = settings(); device = read(OUT/'preflight/device.json')
    assert device['operational'] and device['backend'] == c['backend']
    assert device['processor']['family'] == 'Nighthawk'
    for name in ['cases', 'logs', 'source_snapshot', 'raw_counts']:
        (OUT/name).mkdir(exist_ok=True)
    names = ['configs/hardware_width_sweep.json', 'revision/hardware_width_sweep.py',
             'revision/width_sweep_models.py', 'revision/tensor_prefix_models.py',
             'revision/tensor_prefix_core.py', 'revision/tensor_prefix_sabre.py',
             'revision/native_aqc.py', 'revision/nighthawk_study.py', 'revision/corrected_study.py',
             'revision/full_circuit_explicit.py', 'revision/steps_native_sweep.py',
             'revision/contribution_hardware_core.py', 'tests/test_width_sweep.py']
    hashes = {name: digest(ROOT/name) for name in names}
    protocol = OUT/'preparation_protocol.json'
    if protocol.exists():
        old = read(protocol)
        effective = dict(old['source_sha256'])
        if (OUT/'preparation_amendments.json').exists():
            effective.update(read(OUT/'preparation_amendments.json')['effective_source_sha256'])
        assert effective == hashes and old['target_sha256'] == digest(OUT/'preflight/target.pkl')
        return
    for name in names:
        shutil.copy2(ROOT/name, OUT/'source_snapshot'/Path(name).name)
    save(protocol, dict(frozen_utc=now(), config=c, device=device, source_sha256=hashes,
         target_sha256=digest(OUT/'preflight/target.pkl'), new_hardware_outcomes_seen=False,
         packages={p: importlib.metadata.version(p) for p in ['qiskit', 'qiskit-aer', 'qiskit-addon-aqc-tensor', 'quimb', 'numpy', 'scipy']}))


def exact_state(qc):
    import numpy as np
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method='statevector', max_parallel_threads=2, max_parallel_experiments=1)
    work = qc.copy(); work.save_statevector()
    result = sim.run(work, shots=1).result(); assert result.success
    return np.asarray(result.get_statevector(), dtype=complex)


def dense_target(reference, n, bond):
    """Validation-assisted target preparation, explicitly not a scaling claim."""
    import numpy as np
    import quimb.tensor as qtn
    dense = np.array(reference.reshape((2,)*n).transpose(tuple(reversed(range(n)))), copy=True)
    mps = qtn.MatrixProductState.from_dense(dense, dims=2, max_bond=bond, cutoff=1e-10)
    return qtn.CircuitMPS(n, psi0=mps)


def verify_fitted_objective(target, bound, actual, training_fidelity, exact_target_fidelity, fit_bond, tolerance):
    """Preserve training error; certify the selected state at higher bond if needed."""
    import numpy as np
    from .tensor_prefix_core import mps_state, fidelity
    from .full_circuit_compression import mps_dense_qiskit
    training_error = abs(training_fidelity-exact_target_fidelity)
    report = dict(training_objective_difference=training_error, objective_difference=training_error,
                  verification_bond=fit_bond, verified_tensor_fidelity=training_fidelity,
                  higher_bond_verification=False)
    if training_error < tolerance: return report
    attempts = []
    for bond in [2*fit_bond, 4*fit_bond]:
        checked = mps_state(bound, bond)
        checked_fidelity = fidelity(target, checked)
        dense = mps_dense_qiskit(checked, bound.num_qubits); dense /= np.linalg.norm(dense)
        state_check = float(abs(np.vdot(actual, dense))**2); del dense
        difference = abs(checked_fidelity-exact_target_fidelity)
        attempts.append(dict(bond=bond, objective_difference=difference, fidelity_to_exact_output=state_check))
        if difference < tolerance and state_check > 1-1e-6:
            return dict(report, objective_difference=difference, verification_bond=bond,
                        verified_tensor_fidelity=checked_fidelity, higher_bond_verification=True,
                        verification_attempts=attempts)
    raise AssertionError(dict(training_error=training_error, verification_attempts=attempts))


def state_events(state, model):
    import numpy as np
    from .width_sweep_models import feasible_costs
    costs = feasible_costs(model); keys = np.array(list(costs), dtype=int)
    cost = np.array(list(costs.values())); probabilities = abs(state[keys])**2
    return dict(p_feasible=float(probabilities.sum()),
                p_good=float(probabilities[cost <= 1.05*model['optimum']+1e-9].sum()),
                p_optimal=float(probabilities[abs(cost-model['optimum']) <= 1e-9].sum()))


def worker(cid):
    import numpy as np
    from scipy.optimize import minimize
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from .width_sweep_models import make_model, validate_model
    from .tensor_prefix_core import native_ansatz, mps_state, fidelity, path_patch
    from .full_circuit_explicit import explicit_settings
    from .full_circuit_compression import mps_dense_qiskit
    from .steps_native_sweep import extend_parameters
    from .tensor_prefix_sabre import compile_isa_sabre
    from .nighthawk_study import evolution, qpy_write
    from .tensor_prefix_models import numpy_model
    c = settings(); spec = next(s for s in c['cases'] if s['id'] == cid)
    folder = OUT/'cases'/cid; folder.mkdir(exist_ok=True)
    assert not (folder/'complete.json').exists()
    model = make_model(spec); save(folder/'model.json', model)
    validation = validate_model(model); save(folder/'model_validation.json', validation)
    n = model['n_qubits']; target = pickle.loads((OUT/'preflight/target.pkl').read_bytes())
    physical, edges, patch = path_patch(target, n, c['patch_seed'])
    source = evolution(numpy_model(model), c['annealing_time'], c['steps'])
    print(cid, 'checking exact source state', flush=True)
    if n >= c.get('dense_target_min_qubits', 100):
        import hashlib
        fingerprint = hashlib.sha256(json.dumps(dict(model=model, steps=c['steps'], duration=c['annealing_time']), sort_keys=True).encode()).hexdigest()
        cache = Path('/tmp/aqc_width_sweep_state_cache')/fingerprint; cache.mkdir(parents=True, exist_ok=True)
        if (cache/'reference.npy').exists(): reference = np.load(cache/'reference.npy')
        else:
            reference = exact_state(source); np.save(cache/'reference.npy', reference)
        target_method = 'exact_statevector_to_MPS_SVD'; attempts = []; previous_state = None
        for bond in c['dense_target_check_bonds']:
            print(cid, 'converting exact target to MPS bond', bond, flush=True)
            path = cache/f'target{bond}.pkl'
            if path.exists(): high = pickle.loads(path.read_bytes())
            else:
                high = dense_target(reference, n, bond); path.write_bytes(pickle.dumps(high))
            dense = mps_dense_qiskit(high, n); dense /= np.linalg.norm(dense)
            exact_check = float(abs(np.vdot(reference, dense))**2); del dense
            convergence = fidelity(previous_state, high) if previous_state is not None else None
            attempts.append(dict(bond=bond, exact_fidelity=exact_check, preceding_bond_overlap=convergence))
            print(cid, 'target fidelity', exact_check, flush=True)
            if bond >= c['target_bonds'][1] and exact_check >= c['minimum_target_fidelity']: break
            previous_state = high
        assert exact_check >= c['minimum_target_fidelity'], attempts
        # An exact state comparison supersedes the lower-bond proxy at this size.
        del previous_state
    else:
        reference = exact_state(source)
        low = mps_state(source, c['target_bonds'][0]); high = mps_state(source, c['target_bonds'][1])
        convergence = fidelity(low, high); del low
        dense = mps_dense_qiskit(high, n); dense /= np.linalg.norm(dense)
        exact_check = float(abs(np.vdot(reference, dense))**2); del dense
        assert min(convergence, exact_check) >= c['minimum_target_fidelity'], (convergence, exact_check)
        target_method = 'gatewise_Quimb_MPS'; attempts = []
    assert abs(float(np.vdot(reference, reference).real)-1) < 1e-8
    high._psi /= np.sqrt(float(abs(high.psi.H@high.psi)))
    save(folder/'target_validation.json', dict(bond_convergence=convergence, exact_fidelity=exact_check,
         exact_state_qubits=n, target_construction=target_method, construction_attempts=attempts))
    rows = []
    def record(method, qc, events, fit_record=None):
        logical = folder/(method+'.qpy'); isa_path = folder/(method+'_isa.qpy')
        qpy_write(logical, qc)
        isa, resources = compile_isa_sabre(qc, target, physical, edges, seed=c['transpiler_seed'])
        if method != 'source':
            assert resources['swap_count'] == 0
        assert resources['conservative_accumulated_matrix_error_bound'] < 1e-8
        qpy_write(isa_path, isa)
        row = dict(case_id=cid, family=model['problem'].upper(), n_qubits=n, method=method,
                   logical_qpy=str(logical.relative_to(ROOT)), logical_sha256=digest(logical),
                   isa_qpy=str(isa_path.relative_to(ROOT)), isa_sha256=digest(isa_path),
                   resources=resources, ideal_events=events, fit=fit_record)
        rows.append(row); save(folder/(method+'_result.json'), row)
        print(cid, method, 'CZ depth', resources['cz_depth'], 'events', events,
              'F', fit_record['exact_fidelity'] if fit_record else 1., flush=True)
    record('source', source, state_events(reference, model))
    old_parameters = None; previous = 0
    for layers in c['layers']:
        name = 'product' if layers == 0 else f'aqc_l{layers}'
        ansatz = native_ansatz(n, edges, layers)
        x0 = np.random.default_rng(c['fit_seed']+layers).normal(0, .01, ansatz.num_parameters)
        if old_parameters is not None:
            x0 += extend_parameters(old_parameters, n, len(edges), previous, layers)
        objective = MaximizeStateFidelity(high, ansatz, explicit_settings(c['fit_bond']))
        best = dict(loss=float('inf'), parameters=x0); history = []; started = time.monotonic()
        def fun(x):
            loss, gradient = objective.loss_function(x)
            loss = float(loss); gradient = np.asarray(gradient, float)
            assert np.isfinite(loss) and np.all(np.isfinite(gradient))
            history.append(loss)
            if loss < best['loss']:
                best.update(loss=loss, parameters=np.array(x))
                save(folder/(name+'_fit_checkpoint.json'), dict(parameters=best['parameters'].tolist(),
                     loss=loss, evaluations=len(history), seconds=time.monotonic()-started,
                     fit_bond=c['fit_bond'], execution_source_sha256=SOURCE_HASH))
            return loss, gradient
        class Deadline(Exception): pass
        def stop(*_): raise Deadline()
        fit_cap = c.get('large_native_fit_seconds', c['fit_seconds']) if n >= 25 and layers > 0 else c['fit_seconds']
        signal.signal(signal.SIGALRM, stop); signal.alarm(fit_cap)
        try:
            result = minimize(fun, x0, jac=True, method='L-BFGS-B',
                              options=dict(maxiter=c['fit_maxiter'], maxls=12, ftol=1e-10))
            status = str(result.message)
        except (Deadline, SystemError) as exc:
            if not caused_by(exc, Deadline): raise
            status = 'time_cap_best_completed_evaluation'
        finally:
            signal.alarm(0)
        assert history, 'No fitting evaluation completed'
        elapsed = time.monotonic()-started
        bound = ansatz.assign_parameters(best['parameters']); actual = exact_state(bound)
        exact_f = float(abs(np.vdot(reference, actual))**2)
        dense = mps_dense_qiskit(high, n); dense /= np.linalg.norm(dense)
        target_overlap = float(abs(np.vdot(dense, actual))**2); del dense
        verified_objective = verify_fitted_objective(high, bound, actual, 1-best['loss'], target_overlap,
                                                    c['fit_bond'], c['objective_check_tolerance'])
        fit_record = dict(layers=layers, parameters=best['parameters'].tolist(), exact_fidelity=exact_f,
                          tensor_fidelity=1-best['loss'], **verified_objective,
                          status=status, evaluations=len(history), seconds=elapsed, loss_history=history,
                          verified=True, fitting_bond=c['fit_bond'], fit_seconds_cap=fit_cap)
        record(name, bound, state_events(actual, model), fit_record)
        old_parameters = best['parameters']; previous = layers
        del actual, objective
    save(folder/'complete.json', dict(completed_utc=now(), model=model, physical=physical, edges=edges,
         patch=patch, target_validation=read(folder/'target_validation.json'), rows=rows,
         preparation_protocol_sha256=digest(OUT/'preparation_protocol.json'), execution_source_sha256=SOURCE_HASH))


def run():
    import psutil
    from .cpu_memory_guard import physical_cpu_ids
    freeze(); c = settings(); cpus = physical_cpu_ids(); assert len(cpus) >= 6
    pending = [s['id'] for s in c['cases'] if not (OUT/'cases'/s['id']/'complete.json').exists()]
    running = {}; results = {}; peak = 0.; started = time.monotonic()
    try:
        while pending or running:
            free_slots = [i for i in range(c['parallel_workers']) if i not in running]
            for slot in free_slots:
                if not pending: break
                if psutil.virtual_memory().available < c['reserve_gib']*2**30:
                    raise RuntimeError('System memory reserve reached')
                cid = pending.pop(0); logfile = (OUT/'logs'/(cid+'.log')).open('a')
                command = [sys.executable, '-m', 'revision.cpu_memory_guard', '--address-gib', str(c['worker_address_gib']),
                           '--cpus', ','.join(map(str, cpus[2*slot:2*slot+2])), '--threads', '2', '--',
                           sys.executable, '-m', 'revision.local_runner', '--extra-site', EXTRA,
                           'revision.hardware_width_sweep', 'worker', cid]
                process = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
                running[slot] = dict(process=process, logfile=logfile, started=time.monotonic(), case_id=cid)
                print('Started', cid, 'PID', process.pid, flush=True)
            rss = 0
            for slot, item in list(running.items()):
                process = item['process']
                if process.poll() is not None:
                    item['logfile'].close(); ok = process.returncode == 0 and (OUT/'cases'/item['case_id']/'complete.json').exists()
                    results[item['case_id']] = dict(status='complete' if ok else 'failed', returncode=process.returncode,
                                                   seconds=time.monotonic()-item['started'])
                    print(item['case_id'], results[item['case_id']]['status'], flush=True)
                    del running[slot]; continue
                try:
                    p = psutil.Process(process.pid)
                    for child in [p]+p.children(recursive=True):
                        try: rss += child.memory_info().rss
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                if time.monotonic()-item['started'] > c['worker_seconds']:
                    os.killpg(process.pid, signal.SIGKILL)
            peak = max(peak, rss/2**30)
            if rss > c['aggregate_rss_gib']*2**30 or psutil.virtual_memory().available < c['reserve_gib']*2**30:
                raise RuntimeError('Aggregate memory guard reached; stopping worker tree')
            save(OUT/'preparation_progress.json', dict(updated_utc=now(), completed=results,
                 active=[r['case_id'] for r in running.values()], pending=pending, peak_worker_rss_gib=peak,
                 elapsed_seconds=time.monotonic()-started))
            time.sleep(.5)
    finally:
        for item in running.values():
            try: os.killpg(item['process'].pid, signal.SIGKILL)
            except ProcessLookupError: pass
            item['process'].wait(); item['logfile'].close()
    if not all((OUT/'cases'/s['id']/'complete.json').exists() for s in c['cases']):
        raise RuntimeError('Preparation incomplete; all failures retained for diagnosis, no hardware selection')
    save(OUT/'preparation_complete.json', dict(completed_utc=now(), cases=len(c['cases']), peak_worker_rss_gib=peak))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('action'); p.add_argument('case', nargs='?'); a = p.parse_args()
    if a.action == 'run': run()
    elif a.action == 'worker': worker(a.case)
    elif a.action == 'freeze': freeze()
    else: raise ValueError(a.action)
