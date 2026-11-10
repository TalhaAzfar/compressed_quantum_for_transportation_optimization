"""Describe ideal optimum enrichment within each feasible output distribution."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from .depth200_optimum_screen import OUT


def run():
    path = OUT/'hardware_plan.json'
    plan = json.loads(path.read_text())
    rows = []
    for cid in plan['cases']:
        source = next(r for r in plan['rows'] if r['case_id'] == cid and r['method'] == 'original')
        events = source['events']
        uniform = events['optimal_strings'] / events['feasible_strings']
        assert 0 < uniform <= 1
        for r in plan['rows']:
            if r['case_id'] != cid:
                continue
            conditional = r['events']['p_optimal'] / r['events']['p_feasible']
            rows.append(dict(case_id=cid, method=r['method'],
                             conditional_optimal_probability=conditional,
                             uniform_feasible_optimal_probability=uniform,
                             optimum_enrichment_vs_uniform_feasible=conditional/uniform,
                             feasible_strings=events['feasible_strings'],
                             optimal_strings=events['optimal_strings']))
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=rows,
                  hardware_count_files_present=len(list((OUT/'raw_counts').glob('b*.json'))),
                  scope='Source-only descriptive diagnostic after fitting; no measured outcomes used. '
                        'Uniform-feasible probabilities use these exactly enumerated small instances, '
                        'not a claim of generic efficient feasible sampling. Native FLP probabilities '
                        'are numerically checked MPS quantities; both original sources are exact.')
    (OUT/'conditional_source_quality.json').write_text(json.dumps(record, indent=2)+'\n')
    with (OUT/'conditional_source_quality.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r['method'] == 'original':
            print(r)


if __name__ == '__main__':
    run()
