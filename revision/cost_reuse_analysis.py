"""Rescore fixed hardware candidate pools under new costs, without new QPU runs.

This is a secondary analysis of existing confirmation samples. Reusing these
samples does not add independent hardware observations or adapt the circuits to
the new objective. All three original instances at each chosen setting are used.
"""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import resource

import numpy as np

from .width_sweep_models import feasible_costs, make_model, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cost_reuse_analysis_2026_10_11'
STUDIES = [
    dict(family='FLP', prefix='flp20', study='native_depth_refinement_2026_10',
         models='hardware_width_sweep_2026', problem='flp', customers=4, facilities=4),
    dict(family='TSP', prefix='tsp25', study='long_anneal_hardware_2026_10',
         models='long_anneal_hardware_2026_10', problem='tsp', nodes=6),
    dict(family='VRP', prefix='vrp20', study='long_anneal_hardware_2026_10',
         models='long_anneal_hardware_2026_10', problem='vrp', nodes=5, vehicles=2),
]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'summary.json').exists(), 'Completed analysis is preserved.'
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        studies=STUDIES, new_cost_instances_per_family=20, first_seed=2026101100,
        circuit_selection='All three previously selected preparations and their original controls at each setting.',
        cost_generation='Existing cost generators with new independent seeds; the same 20 costs are scored for all three fixed preparations in each family.',
        inference='Descriptive retrospective rescoring; no significance tests across reused pools and no new hardware observations.',
        outcomes=['feasible and unique feasible counts', 'observed optimum and 5%-gap hits',
                  'best observed relative gap', 'conditional mean relative gap'],
        classical_reference='Uniform feasible sampling at the same number of feasible draws; expected optimum-hit probability and expected best gap.',
        additional_qpu_shots=0, parameter_refitting=False,
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT/'revision/width_sweep_models.py', ROOT/'revision/tensor_prefix_models.py']})
    save(OUT/'protocol.json', protocol)
    scenarios, inputs, records, pool_rows = [], {}, [], []
    for group_index, group in enumerate(STUDIES):
        folder = ROOT/'outputs'/group['study']
        plan_path = folder/'confirm_plan.json'
        plan = read(plan_path)
        inputs[str(plan_path.relative_to(ROOT))] = digest(plan_path)
        ids = [group['prefix'] + '_r' + str(i) for i in [1, 2, 3]]
        expected = {(r['case_id'], r['method']): r for r in plan['rows'] if r['case_id'] in ids}
        assert len(expected) == 12
        models = {}
        for cid in ids:
            path = ROOT/'outputs'/group['models']/'cases'/cid/'model.json'
            models[cid] = read(path)
            inputs[str(path.relative_to(ROOT))] = digest(path)
        feasible_set = set(feasible_costs(models[ids[0]]))
        assert all(set(feasible_costs(m)) == feasible_set for m in models.values())

        new_models = []
        for i in range(20):
            spec = {k: group[k] for k in ['problem', 'customers', 'facilities', 'nodes', 'vehicles'] if k in group}
            spec.update(id=group['prefix']+'_newcost_'+str(i+1), seed=2026101100+100*group_index+i)
            assert spec['seed'] not in {m['seed'] for m in models.values()}
            m = make_model(spec)
            costs = feasible_costs(m)
            assert set(costs) == feasible_set
            assert abs(min(costs.values()) - m['optimum']) < 1e-9
            new_models.append((m, costs))
            scenarios.append(m)
        # Save every cost realization before scoring it; none is selected on outcomes.
        save(OUT/'cost_scenarios.json', scenarios)

        pooled = {key: Counter() for key in expected}
        blocks = set()
        for path in sorted((folder/'raw_counts').glob('confirm*'+group['family']+'*.json')):
            raw = read(path)
            inputs[str(path.relative_to(ROOT))] = digest(path)
            for row in raw['rows']:
                key = row['case_id'], row['method']
                if key not in expected:
                    continue
                assert expected[key]['candidate_id'] == row['candidate_id']
                assert metrics(models[key[0]], row['counts']) == row['metrics']
                block_key = (*key, raw['block'])
                assert block_key not in blocks
                blocks.add(block_key)
                pooled[key].update(row['counts'])
        assert blocks == {(*key, b) for key in expected for b in [0, 1]}

        for key, counts in pooled.items():
            cid, method = key
            shots = sum(counts.values())
            assert shots == 2*plan['shots_per_circuit']
            feasible = {int(s, 2): n for s, n in counts.items() if int(s, 2) in feasible_set}
            nfeasible = sum(feasible.values())
            pool = dict(family=group['family'], source_study=group['study'],
                training_instance=cid, method=method, candidate_id=expected[key]['candidate_id'],
                n_qubits=models[cid]['n_qubits'], shots=shots,
                cz_depth=expected[key]['resources']['cz_depth'], feasible_count=nfeasible,
                p_feasible=nfeasible/shots, unique_feasible=len(feasible),
                feasible_space_size=len(feasible_set), feasible_space_coverage=len(feasible)/len(feasible_set))
            pool_rows.append(pool)
            for model, costs in new_models:
                optimum = model['optimum']
                observed = [(costs[s], n) for s, n in feasible.items()]
                optimum_count = sum(n for cost, n in observed if abs(cost-optimum) <= 1e-9)
                good_count = sum(n for cost, n in observed if cost <= 1.05*optimum+1e-9)
                best = min((cost for cost, n in observed), default=None)
                gaps = np.sort([(cost-optimum)/optimum for cost in costs.values()])
                nopt = sum(abs(cost-optimum) <= 1e-9 for cost in costs.values())
                uf_hit = 1-(1-nopt/len(costs))**nfeasible
                # Exact order-statistic expectation for independent uniform
                # feasible draws, with duplicates allowed and equal sample count.
                survival = (1-np.arange(len(costs)+1)/len(costs))**nfeasible if nfeasible else None
                uf_gap = float(np.dot(gaps, survival[:-1]-survival[1:])) if nfeasible else None
                records.append(dict(**pool, test_instance=model['instance_id'], test_seed=model['seed'],
                    optimum_cost=optimum, optimum_count=optimum_count, near_optimal_count=good_count,
                    optimum_found=bool(optimum_count), near_optimal_found=bool(good_count),
                    p_optimal=optimum_count/shots, p_near_optimal=good_count/shots,
                    best_cost=best, best_relative_gap=(best-optimum)/optimum if best is not None else None,
                    conditional_mean_relative_gap=(sum((cost-optimum)/optimum*n for cost, n in observed)/nfeasible) if nfeasible else None,
                    uniform_feasible_expected_optimum_hit=uf_hit,
                    uniform_feasible_expected_best_gap=uf_gap))

    write_csv(OUT/'candidate_pools.csv', pool_rows)
    write_csv(OUT/'cost_rescoring.csv', records)
    save(OUT/'input_sha256.json', inputs)
    summaries = []
    for group in STUDIES:
        for method in ['source', 'refined_product', 'selected_native']:
            rows = [r for r in records if r['family'] == group['family'] and r['method'] == method]
            assert len(rows) == 60
            gaps = [r['best_relative_gap'] for r in rows if r['best_relative_gap'] is not None]
            pools = [r for r in pool_rows if r['family'] == group['family'] and r['method'] == method]
            summaries.append(dict(family=group['family'], method=method,
                physical_candidate_pools=3, independent_new_cost_instances=20,
                pool_cost_pairs=60, shots_per_pool=pools[0]['shots'],
                feasible_counts=[p['feasible_count'] for p in pools],
                unique_feasible_counts=[p['unique_feasible'] for p in pools],
                feasible_space_size=pools[0]['feasible_space_size'],
                optimum_found_pairs=sum(r['optimum_found'] for r in rows),
                near_optimal_found_pairs=sum(r['near_optimal_found'] for r in rows),
                pairs_without_feasible_candidate=sum(r['best_relative_gap'] is None for r in rows),
                median_best_relative_gap_given_feasible=float(np.median(gaps)) if gaps else None,
                mean_uniform_feasible_optimum_hit_given_same_feasible_draws=float(np.mean([r['uniform_feasible_expected_optimum_hit'] for r in rows]))))
    assert len(scenarios) == 60 and len(pool_rows) == 36 and len(records) == 720
    assert all(digest(ROOT/name) == value for name, value in inputs.items())
    summary = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Reuse of hardware-sampled feasible candidate pools under changed objective coefficients; not cost-adaptive quantum optimization, parameter transfer, or independent new QPU confirmation.',
        feasibility_invariant_under_cost_only_changes=True, new_cost_instances=60,
        hardware_candidate_pools=36, additional_qpu_shots=0, dense_statevectors_loaded=0,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        outcomes=summaries)
    save(OUT/'summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
