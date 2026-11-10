"""Check cost rescoring with the independent bit decoder and show every control."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .cost_reuse_analysis import ROOT, OUT, STUDIES, digest, read, save, write_csv
from .width_sweep_models import decode, feasible_costs


def main():
    protocol = read(OUT/'protocol.json')
    for name, checksum in {**protocol['source_sha256'], **read(OUT/'input_sha256.json')}.items():
        assert digest(ROOT/name) == checksum, name
    scenarios = {m['instance_id']: m for m in read(OUT/'cost_scenarios.json')}
    assert len(scenarios) == 60
    pools = {}
    for group in STUDIES:
        folder = ROOT/'outputs'/group['study']
        ids = [group['prefix']+'_r'+str(i) for i in [1, 2, 3]]
        for path in sorted((folder/'raw_counts').glob('confirm*'+group['family']+'*.json')):
            for row in read(path)['rows']:
                if row['case_id'] in ids:
                    pools.setdefault((row['case_id'], row['method']), Counter()).update(row['counts'])
    feasible_sets = {}
    for sid, model in scenarios.items():
        costs = feasible_costs(model)
        values = np.array(sorted(costs), dtype=np.uint64)
        decoded = decode(model, values)
        assert decoded['feasible'].all()
        assert np.allclose(decoded['cost'], [costs[int(v)] for v in values], atol=1e-9, rtol=0)
        feasible_sets[sid] = set(costs)
    records = list(csv.DictReader((OUT/'cost_rescoring.csv').open()))
    for row in records:
        model = scenarios[row['test_instance']]
        pool = pools[row['training_instance'], row['method']]
        valid = {int(s, 2): n for s, n in pool.items() if int(s, 2) in feasible_sets[row['test_instance']]}
        assert sum(pool.values()) == int(row['shots'])
        assert sum(valid.values()) == int(row['feasible_count'])
        assert len(valid) == int(row['unique_feasible'])
        if valid:
            result = decode(model, np.array(list(valid), dtype=np.uint64))
            weights = np.array(list(valid.values()))
            assert result['feasible'].all()
            assert int(weights @ result['optimal']) == int(row['optimum_count'])
            assert int(weights @ result['good']) == int(row['near_optimal_count'])
            assert abs(float(result['cost'].min())-float(row['best_cost'])) < 1e-9
        else:
            assert not row['best_cost'] and int(row['optimum_count']) == int(row['near_optimal_count']) == 0
    summary = []
    for group in STUDIES:
        methods = ['source', 'refined_product',
                   'old_aqc_l2' if group['family'] == 'FLP' else 'fixed_shallow', 'selected_native']
        for method in methods:
            rows = [r for r in records if r['family'] == group['family'] and r['method'] == method]
            assert len(rows) == 60
            gaps = [float(r['best_relative_gap']) for r in rows if r['best_relative_gap']]
            summary.append(dict(family=group['family'], method=method,
                new_cost_instances=20, original_fixed_circuits=3, pool_cost_pairs=60,
                shots_per_pool=int(rows[0]['shots']),
                optimum_present_pairs=sum(r['optimum_found'] == 'True' for r in rows),
                within_five_percent_pairs=sum(r['near_optimal_found'] == 'True' for r in rows),
                pairs_without_feasible_sample=sum(not r['best_relative_gap'] for r in rows),
                median_best_gap_percent_given_feasible=100*float(np.median(gaps)) if gaps else None,
                same_feasible_draws_uniform_expected_optimum_hit_percent=100*float(np.mean([
                    float(r['uniform_feasible_expected_optimum_hit']) for r in rows]))))
    write_csv(OUT/'all_controls_summary.csv', summary)
    save(OUT/'all_controls_summary.json', dict(
        interpretation='Descriptive candidate-pool reuse. Twenty new objectives per family are each evaluated on three existing circuit sample pools. The 60 pairs per method share hardware samples and are not 60 independent hardware runs.',
        uniform_feasible_reference='A classical distribution over valid decisions at the same feasible-draw count; this is not an equal-total-shot or runtime comparison.',
        outcomes=summary))
    save(OUT/'validation.json', dict(all_input_hashes_verified=True,
        original_analysis_source_hash_verified=True, new_cost_scenarios=60,
        rescored_rows_verified_with_bit_decoder=len(records), all_controls_reported=True,
        feasibility_invariant=True, additional_hardware_runs=0,
        verifier_sha256=digest(Path(__file__))))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
