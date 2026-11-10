"""Descriptive runtime, block-variation and finite-pool sample-budget summaries.

No fitting, simulation, hardware submission or new significance tests are run.
Budget calculations sample without replacement from each observed count pool.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def no_hit(total, hits, budget):
    if not hits or not budget:
        return 1.0
    if budget > total - hits:
        return 0.0
    return math.exp(math.fsum(math.log1p(-budget / (total - j))
                              for j in range(hits)))


def best_gap(total, distribution, budget):
    feasible = sum(count for _, count in distribution)
    hit = 1.0 - no_hit(total, feasible, budget)
    cumulative = 0
    previous = 0.0
    expectation = 0.0
    for gap, count in sorted(distribution):
        cumulative += count
        cdf = 1.0 - no_hit(total, cumulative, budget)
        mass = max(0.0, cdf - previous)
        expectation += gap * mass
        previous = cdf
    assert abs(previous - hit) < 1e-10
    return hit, expectation / hit if hit else None


def variability(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        assert len(values) == 2, (key, len(values))
        values.sort(key=lambda r: int(r['block']))
        probabilities = [int(r['feasible_count']) / int(r['shots']) for r in values]
        result.append(dict(zip(group_keys, key),
                           feasible_counts=[int(r['feasible_count']) for r in values],
                           shots=[int(r['shots']) for r in values],
                           absolute_difference_pp=100 * abs(probabilities[0] - probabilities[1])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root', type=Path, default=ROOT.parent)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'data/revision_diagnostics.json')
    args = parser.parse_args()
    inputs = {}

    def checked(path):
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    runtime = read_csv(checked(ROOT / 'data/fit_runtime_records.csv'))
    fits = [row for row in runtime if row['fitting_seconds']]
    assert len(fits) == 324
    runtime_groups = defaultdict(list)
    for row in fits:
        runtime_groups[(row['family'], int(row['n_qubits']))].append(row)
    runtime_summary = []
    for (family, width), rows in sorted(runtime_groups.items()):
        seconds = [float(r['fitting_seconds']) for r in rows]
        runtime_summary.append(dict(family=family, n_qubits=width, fits=len(rows),
                                    median_fit_seconds=statistics.median(seconds),
                                    maximum_fit_seconds=max(seconds),
                                    sum_fit_hours=math.fsum(seconds) / 3600))

    costs = read_csv(checked(ROOT / 'data/candidate_quality_weighted_cost_distributions.csv'))
    summaries = read_csv(checked(ROOT / 'data/candidate_quality_per_case_summary.csv'))
    distribution = defaultdict(list)
    for row in costs:
        distribution[(row['study'], row['case_id'], row['method'])].append(
            (float(row['gap_pct']), int(row['count'])))
    budget_rows = []
    for row in summaries:
        key = (row['study'], row['case_id'], row['method'])
        group = ('FLP42' if row['study'] == 'flp42' else
                 row['case_id'].rsplit('_r', 1)[0].upper())
        if (row['study'], group) not in {('main', 'FLP20'), ('longer', 'TSP25'),
                                        ('longer', 'VRP20'), ('flp42', 'FLP42')}:
            continue
        total = int(row['shots'])
        assert sum(n for _, n in distribution[key]) == int(row['feasible_count'])
        for fraction in [.25, .5, 1.0]:
            budget = int(total * fraction)
            hit, expected = best_gap(total, distribution[key], budget)
            budget_rows.append(dict(study=row['study'], group=group, case_id=row['case_id'],
                                    method=row['method'], full_shots=total,
                                    budget_fraction=fraction, budget_shots=budget,
                                    empirical_feasible_hit_probability=hit,
                                    conditional_expected_best_gap_pct=expected))
    budget_groups = defaultdict(list)
    for row in budget_rows:
        budget_groups[row['group'], row['method'], row['budget_fraction']].append(row)
    budget_summary = []
    for (group, method, fraction), rows in sorted(budget_groups.items()):
        defined = [r['conditional_expected_best_gap_pct'] for r in rows
                   if r['conditional_expected_best_gap_pct'] is not None]
        budget_summary.append(dict(group=group, method=method, fraction=fraction,
                                   instances=len(rows), instances_with_feasible=len(defined),
                                   budget_shots_per_instance=rows[0]['budget_shots'],
                                   minimum_feasible_hit_probability=min(
                                       r['empirical_feasible_hit_probability'] for r in rows),
                                   median_conditional_expected_best_gap_pct=(
                                       statistics.median(defined) if defined else None)))

    block_rows = read_csv(checked(ROOT / 'data/candidate_quality_by_block.csv'))
    variations = variability(block_rows, ['study', 'case_id', 'method'])
    replication = []
    for path in sorted((args.workspace_root / 'outputs/reproducibility_2026_10/raw_counts').glob('confirm_*.json')):
        block = json.loads(checked(path).read_text())
        for row in block['rows']:
            metrics = row['metrics']
            assert sum(row['counts'].values()) == metrics['shots']
            replication.append(dict(study='replication_day_' + str(block['day']),
                                    case_id=row['case_id'], method=row['method'],
                                    block=block['block'], shots=metrics['shots'],
                                    feasible_count=metrics['feasible_count']))
    if replication:
        assert len(replication) == 96
        compact = ROOT / 'data/replication_block_counts.csv'
        with compact.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(replication[0]))
            writer.writeheader()
            writer.writerows(replication)
    else:
        replication = read_csv(ROOT / 'data/replication_block_counts.csv')
        assert len(replication) == 96
    checked(ROOT / 'data/replication_block_counts.csv')
    variations += variability(replication, ['study', 'case_id', 'method'])
    study_variations = defaultdict(list)
    for row in variations:
        study_variations[row['study']].append(row['absolute_difference_pp'])
    variation_summary = [dict(study=study, preparations=len(values),
                              median_absolute_difference_pp=statistics.median(values),
                              maximum_absolute_difference_pp=max(values))
                         for study, values in sorted(study_variations.items())]
    result = dict(scope='Descriptive analyses of existing experiments; no new significance tests.',
                  runtime_scope='Optimizer elapsed times summed across fits; not CPU time or parallel campaign duration. Target construction, MPS conversion, validation, compilation and hardware screening are additional costs.',
                  runtime=runtime_summary,
                  total_optimizer_hours=math.fsum(float(r['fitting_seconds']) for r in fits) / 3600,
                  budget_scope='Exact random subsampling without replacement from each finite observed pool. Expected minimum gap is conditional on at least one feasible outcome. Group summaries take the median of instance-specific expectations, not the minimum across different instances. No chronological order or future hardware model is assumed.',
                  budget=budget_rows, budget_summary=budget_summary,
                  variability_scope='Absolute difference in feasible rates between two blocks. Descriptive; not a drift estimator or a corrected test of heterogeneity.',
                  block_variability=variations, block_variability_summary=variation_summary,
                  inputs_sha256=inputs)
    output = args.output
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['runtime', 'total_optimizer_hours',
                                            'block_variability_summary']}, indent=2))
    print('SELECTED NATIVE BUDGETS')
    for row in budget_summary:
        if row['method'] == 'selected_native':
            print(row)


if __name__ == '__main__':
    main()
