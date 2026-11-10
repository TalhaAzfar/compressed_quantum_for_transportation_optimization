"""Diagnose larger fixed instances without adding hardware or changing targets."""
import argparse
import csv
import math
import time
from .contribution_hardware_core import ROOT, read, save, digest, now
from .reproducibility_study import OUT, PARENT, check_frozen

WIDTH = ROOT/'outputs/hardware_width_sweep_2026'


def shots_for_hit(p, confidence=.95):
    if not 0 <= p <= 1 or not 0 < confidence < 1:
        raise ValueError('Invalid probability')
    if p == 0: return None
    if p == 1: return 1
    return math.ceil(math.log1p(-confidence)/math.log1p(-p))


def probability_of_hit(p, shots):
    if not 0 <= p <= 1 or shots < 0:
        raise ValueError('Invalid probability or shots')
    if not shots or p == 0: return 0.
    if p == 1: return 1.
    return -math.expm1(shots*math.log1p(-p))


def existing():
    """Derived diagnosis from frozen exact events and fresh confirmation counts."""
    with (PARENT/'confirm_pooled.csv').open() as stream:
        measured = list(csv.DictReader(stream))
    with (PARENT/'classical_best_by_budget.csv').open() as stream:
        budgets = list(csv.DictReader(stream))
    source = {r['case_id']: r for r in measured if r['method'] == 'source'}
    rows = []
    for r in measured:
        if int(r['n_qubits']) < 16: continue
        for event in ['feasible', 'optimal']:
            ideal = float(r['ideal_p_'+event]); reference = float(source[r['case_id']]['ideal_p_'+event])
            count = int(r[event+'_count']); shots = int(r['shots'])
            rows.append(dict(case_id=r['case_id'], family=r['family'], n_qubits=int(r['n_qubits']),
                method=r['method'], event=event, cz_depth=int(r['cz_depth']), exact_fidelity=float(r['exact_fidelity']),
                source_ideal_probability=reference, circuit_ideal_probability=ideal,
                ideal_retention_relative_to_source=ideal/reference if reference else None,
                hardware_count=count, shots=shots, hardware_probability=count/shots,
                hardware_interval=[float(r['p_'+event+'_low']), float(r['p_'+event+'_high'])],
                hardware_to_ideal_ratio=(count/shots)/ideal if ideal else None,
                ideal_expected_count=shots*ideal,
                ideal_probability_at_least_one=probability_of_hit(ideal, shots),
                ideal_shots_for_95_percent_hit=shots_for_hit(ideal)))
    result = dict(created_utc=now(), rows=rows,
        best_fidelity_depth_rows=[r for r in budgets if int(r['n_qubits']) >= 16 and r['criterion'] == 'fidelity'],
        inputs_sha256={str((PARENT/n).relative_to(ROOT)):digest(PARENT/n) for n in ['confirm_pooled.csv','classical_best_by_budget.csv','confirmation_comparisons.csv']},
        interpretation='Ratios diagnose changes in probability, not an identified noise mechanism. Observed zero counts do not imply zero probability. Shot projections use ideal or point estimates, not guarantees. The anneal is a fixed finite-time source, not an assumed optimizer.')
    save(OUT/'diagnostics/existing_large_cases.json', result)
    with (OUT/'diagnostics/existing_large_cases.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for cid in source:
        if not cid.startswith(('tsp25','vrp20','flp20')): continue
        rs = [r for r in rows if r['case_id'] == cid and r['method'] == 'source']
        print(cid, {r['event']:dict(ideal=r['circuit_ideal_probability'], expected=r['ideal_expected_count'], observed=r['hardware_count']) for r in rs}, flush=True)
    return result


def worker(cid):
    import numpy as np
    from .nighthawk_study import evolution, qpy_write
    from .tensor_prefix_models import numpy_model
    from .hardware_width_sweep import exact_state, state_events
    from .width_sweep_models import feasible_costs, validate_model
    check_frozen()
    assert cid in [f'{f}_r{i}' for f in ['tsp25','vrp20'] for i in range(1,4)]
    folder = OUT/'diagnostics'/cid; folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists(): return
    case = read(WIDTH/'cases'/cid/'complete.json'); model = case['model']
    validation = validate_model(model)
    save(folder/'model_validation.json', validation)
    costs = feasible_costs(model); uniform = len(costs)/(1 << model['n_qubits'])
    rows = []
    # Every specified setting is retained. No new grid setting is sent to hardware.
    for duration in [2.,4.,8.]:
        previous = None
        for steps in [12,24]:
            name = f't{int(duration)}_s{steps}'; started = time.monotonic()
            circuit = evolution(numpy_model(model), duration, steps)
            path = folder/(name+'.qpy'); qpy_write(path, circuit)
            state = (np.load(PARENT/'targets'/cid/'reference.npy', mmap_mode='r')
                     if duration == 2. and steps == 12 else exact_state(circuit))
            norm = float(np.vdot(state,state).real); assert abs(norm-1) < 1e-8
            events = state_events(state, model)
            if duration == 2. and steps == 12:
                saved = next(r for r in case['rows'] if r['method'] == 'source')['ideal_events']
                assert all(abs(events[k]-saved[k]) < 1e-8 for k in events)
            row = dict(case_id=cid, n_qubits=model['n_qubits'], family=model['problem'].upper(),
                duration=duration, steps=steps, **events, exact_norm=norm,
                same_duration_12_step_fidelity=float(abs(np.vdot(previous,state))**2) if previous is not None else None,
                uniform_bitstring_p_feasible=uniform,
                feasible_enrichment=events['p_feasible']/uniform,
                p_optimal_conditional_feasible=events['p_optimal']/events['p_feasible'] if events['p_feasible'] else None,
                ideal_optimum_expected_per_8192=8192*events['p_optimal'],
                ideal_optimum_hit_probability_per_8192=probability_of_hit(events['p_optimal'],8192),
                ideal_shots_for_95_percent_optimum=shots_for_hit(events['p_optimal']),
                logical_qpy=str(path.relative_to(ROOT)), logical_sha256=digest(path),
                seconds=time.monotonic()-started)
            save(folder/(name+'.json'), row); rows.append(row)
            print(cid, name, events, '12/24 fidelity', row['same_duration_12_step_fidelity'], flush=True)
            previous = state if steps == 12 else None
            del state
    save(folder/'complete.json', dict(completed_utc=now(), case_id=cid, rows=rows,
        model_sha256=digest(WIDTH/'cases'/cid/'model.json'),
        original_reference_sha256=digest(PARENT/'targets'/cid/'reference.npy'),
        scope='Exploratory ideal source diagnosis only; all T=2/4/8 and 12/24-step settings retained. No hardware selection or asymptotic digitization-convergence claim. VRP encoding remains within its certified four-customer limit.'))


def summarize():
    existing()
    rows = []
    for folder in sorted((OUT/'diagnostics').glob('*_r*')):
        if (folder/'complete.json').exists(): rows.extend(read(folder/'complete.json')['rows'])
    if rows:
        save(OUT/'diagnostics/source_duration_grid.json',dict(created_utc=now(),rows=rows,
            complete=len(rows)==36,hardware_results=False))
        with (OUT/'diagnostics/source_duration_grid.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['existing','worker','summarize']);p.add_argument('case',nargs='?');a=p.parse_args()
    worker(a.case) if a.action=='worker' else globals()[a.action]()
