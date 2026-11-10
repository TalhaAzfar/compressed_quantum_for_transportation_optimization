"""Development audit for a new, complete-circuit depth-200 optimum study.

Earlier studies are read-only inputs. Exact simulation is limited to 25 qubits;
larger references require explicit MPS convergence, including rare events.
"""
import argparse
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/depth200_optimum_study'
EXTRA = ""


def save(path, value):
    from .flp42_fidelity_parallel import save as atomic_save
    atomic_save(path, value)


def case_data(case_id):
    return json.loads((ROOT / 'outputs/aqc_tensor_qaoa_v2/cases' / (case_id + '.json')).read_text())


def feasible_values(model):
    """Enumerate feasible labels, independently of the quantum state."""
    if model['problem'] == 'tsp':
        m = model['nodes'] - 1
        return [sum(1 << (t*m+j) for t, j in enumerate(tour))
                for tour in itertools.permutations(range(m))]
    f, c = model['facilities'], model['customers']
    values = []
    for mask in range(1, 1 << f):
        opened = [j for j in range(f) if mask >> j & 1]
        for assignment in itertools.product(opened, repeat=c):
            values.append(mask + sum(1 << (f+i*f+j) for i, j in enumerate(assignment)))
    return values


def metrics_from_amplitudes(model, values, amplitudes, norm=1.):
    import numpy as np
    from .tensor_prefix_models import decode
    labels = decode(model, values)
    assert labels['feasible'].all()
    p = np.abs(np.asarray(amplitudes))**2 / norm
    result = {f'p_{event}': float(p[labels[event]].sum())
              for event in ['feasible', 'good', 'optimal']}
    result['conditional_feasible_mean_cost'] = float(p @ labels['cost'] / p.sum()) if p.sum() else None
    result['feasible_strings'] = len(values)
    result['optimal_strings'] = int(labels['optimal'].sum())
    return result


def mps_metrics(model, state):
    if model['problem'] == 'flp':
        from .flp_mps_events import probabilities
        return probabilities(model, state)
    from .flp_mps_events import site_arrays
    import numpy as np
    arrays, norm = site_arrays(state)
    values = feasible_values(model)
    amplitudes = []
    for value in values:
        v = np.ones(1, dtype=complex)
        for q, a in enumerate(arrays):
            v = v @ a[:, (value >> q) & 1, :]
        amplitudes.append(v[0])
    return metrics_from_amplitudes(model, values, amplitudes, norm)


def worker(case_id, method, mode, bond):
    from .nighthawk_study import qpy_read
    from .flp42_fidelity_parallel import digest, now
    started = time.monotonic()
    case = case_data(case_id)
    model = case['model']
    source = ROOT / case['methods'][method]['logical_qpy']
    qc = qpy_read(source)
    name = f'{case_id}_{method}_{mode}' + (f'_b{bond}' if mode == 'mps' else '')
    record = dict(case_id=case_id, method=method, mode=mode, bond=bond if mode == 'mps' else None,
                  n_qubits=model['n_qubits'], source_qpy=str(source.relative_to(ROOT)),
                  source_sha256=digest(source), classical_optimum=model['optimum'])
    if mode == 'exact':
        assert qc.num_qubits <= 25, 'Dense simulator hard cap'
        from qiskit_aer import AerSimulator
        from qiskit_aer.library import save_amplitudes
        values = feasible_values(model)
        saved = qc.copy()
        saved.save_amplitudes(values)
        sim = AerSimulator(method='statevector', max_parallel_threads=1,
                           max_parallel_experiments=1, max_memory_mb=4096)
        result = sim.run(saved, shots=1).result()
        assert result.success
        record['events'] = metrics_from_amplitudes(model, values, result.data(0)['amplitudes'])
        record['verified'] = True
    else:
        from .placement_aer_reference import reference
        state, energy = reference(qc, model, bond, 1e-13)
        state_path = OUT / 'references' / (name + '.pkl')
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_bytes(pickle.dumps(state))
        record.update(events=mps_metrics(model, state), energy=energy,
                      state_pickle=str(state_path.relative_to(ROOT)), state_sha256=digest(state_path),
                      achieved_bond=state.psi.max_bond(), verified=False,
                      verification_note='Single-bond result; convergence is not yet established.')
    p = record['events']['p_optimal']
    import math
    record.update(expected_optima_32768=32768*p,
                  ideal_probability_at_least_one_32768=-math.expm1(32768*math.log1p(-p)) if p < 1 else 1.,
                  seconds=time.monotonic()-started, completed_utc=now(), affinity=sorted(os.sched_getaffinity(0)))
    save(OUT / 'screen' / (name + '.json'), record)
    print(name, record['events'], 'seconds', record['seconds'], flush=True)


def preflight():
    from .nighthawk_study import connect
    from .flp42_fidelity_parallel import now, digest
    config = json.loads((ROOT / 'configs/aqc_tensor_qaoa_v2.json').read_text())
    _, backend = connect(config)
    folder = OUT / 'preflight'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'target_snapshot.pkl'
    if target.exists():
        raise RuntimeError('Preflight snapshot already exists; use a separately named acquisition calibration.')
    target.write_bytes(pickle.dumps(backend.target))
    save(folder / 'device.json', dict(checked_utc=now(), backend=backend.name,
         processor=backend.configuration().processor_type, status=backend.status().to_dict(),
         version=backend.backend_version, target_sha256=digest(target)))
    save(folder / 'calibration.json', json.loads(json.dumps(backend.properties(refresh=True).to_dict(), default=str)))
    print(json.dumps(json.loads((folder / 'device.json').read_text())), flush=True)


def pool():
    """Small, explicit development screen; all attempted candidates retained."""
    from .flp42_fidelity_parallel import now, digest
    jobs = [('tsp25', m, 'exact', 0) for m in
            ['standard_p1', 'standard_p2', 'aqc_quarter', 'shallow_quarter', 'refined_quarter']]
    jobs += [('flp20', m, 'exact', 0) for m in ['aqc_half', 'standard_p2']]
    jobs += [('flp30', m, 'mps', b) for m in ['standard_p1', 'aqc_half', 'standard_p2'] for b in [64, 128]]
    protocol = OUT / 'development_screen_protocol.json'
    frozen = dict(jobs=jobs, exact_qubit_cap=25, single_worker_deadline_seconds=900,
        source_sha256=digest(Path(__file__)), purpose='Audit ideal optimum and feasibility probabilities before fitting and hardware selection.',
        selection_note='Development using previously evaluated instances. Not independent confirmatory evidence; no optimum bitstring used in circuit preparation or AQC loss.',
        prior_results_known=True)
    if protocol.exists():
        prior=json.loads(protocol.read_text())
        assert prior['source_sha256'] == frozen['source_sha256']
    else:
        save(protocol, dict(created_utc=now(), **frozen))
    cpus=sorted(os.sched_getaffinity(0)); assert len(cpus) == 4
    active=[]
    pending=list(jobs)
    while pending or active:
        free=[cpu for cpu in cpus if cpu not in [x['cpu'] for x in active]]
        while pending and free:
            cid, method, mode, bond=pending.pop(0)
            name=f'{cid}_{method}_{mode}' + (f'_b{bond}' if mode == 'mps' else '')
            if (OUT / 'screen' / (name+'.json')).exists():continue
            cpu=free.pop(0)
            path=OUT / 'logs' / (name+'.log');path.parent.mkdir(parents=True,exist_ok=True)
            log=path.open('a')
            cmd=[sys.executable,'-m','revision.local_runner','--extra-site',EXTRA,
                 'revision.depth200_optimum_screen','worker',cid,method,mode,str(bond),'--cpu',str(cpu)]
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(child=child,log=log,cpu=cpu,name=name,started=time.monotonic()))
            print('Started',name,'CPU',cpu,flush=True)
        for item in active[:]:
            expired=time.monotonic()-item['started']>900
            if expired and item['child'].poll() is None:item['child'].kill()
            if item['child'].poll() is None:continue
            code=item['child'].wait();item['log'].close()
            save(OUT/'logs'/(item['name']+'_process.json'),dict(returncode=code,seconds=time.monotonic()-item['started'],
                 cpu=item['cpu'],status='wall_time_limit' if expired else ('completed' if code==0 else 'worker_failed')))
            print('Finished',item['name'],code,flush=True);active.remove(item)
        time.sleep(.2)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['worker','pool','preflight'])
    p.add_argument('args',nargs='*');p.add_argument('--cpu',type=int)
    a=p.parse_args()
    if a.cpu is not None:os.sched_setaffinity(0,{a.cpu})
    if a.action=='worker':worker(*a.args[:3],int(a.args[3]))
    elif a.action=='pool':pool()
    else:preflight()
