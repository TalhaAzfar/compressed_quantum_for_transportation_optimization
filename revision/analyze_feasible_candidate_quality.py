"""Describe original-cost quality of independently confirmed feasible samples.

Only saved counts and small classical model definitions are read. No quantum
jobs, state-vector construction, circuit selection, or inferential tests occur.
"""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np
from .width_sweep_models import decode, feasible_costs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/feasible_candidate_quality_2026_10_14'
STUDIES = {
    'main': ('native_depth_refinement_2026_10', 'hardware_width_sweep_2026'),
    'longer': ('long_anneal_hardware_2026_10', 'long_anneal_hardware_2026_10'),
    'flp42': ('larger_width_feasibility_2026_10', None),
}
METHODS = {'source': 'source', 'selected_native': 'selected_native',
           'refined_product': 'product', 'product': 'product',
           'old_aqc_l2': 'shallow', 'fixed_shallow': 'shallow', 'shallow': 'shallow'}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def csv_file(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def weighted_quantile(values, counts, p):
    """NumPy's linear sample quantile, without expanding repeated observations."""
    pairs = sorted(zip(values, counts))
    if not pairs:
        return None
    x = np.asarray([v for v, n in pairs]); n = np.asarray([n for v, n in pairs])
    cumulative = np.cumsum(n); position = (int(cumulative[-1])-1)*p
    lower, upper = math.floor(position), math.ceil(position)
    a, b = x[np.searchsorted(cumulative, lower, side='right')], x[np.searchsorted(cumulative, upper, side='right')]
    return float(a + (position-lower)*(b-a))


def summarize_distribution(distribution, shots):
    n = sum(distribution.values())
    if not n:
        return dict(feasible_count=0, p_feasible=0., mean_gap_pct=None,
                    q25_gap_pct=None, median_gap_pct=None, q75_gap_pct=None,
                    best_gap_pct=None)
    values, counts = list(distribution), list(distribution.values())
    return dict(feasible_count=n, p_feasible=n/shots,
                mean_gap_pct=sum(v*w for v, w in distribution.items())/n,
                q25_gap_pct=weighted_quantile(values, counts, .25),
                median_gap_pct=weighted_quantile(values, counts, .5),
                q75_gap_pct=weighted_quantile(values, counts, .75),
                best_gap_pct=min(distribution))


def scalar_check(model, bits):
    """Independent decision decoding for recorded feasible outcomes."""
    value = int(bits, 2)
    if model['problem'] == 'flp':
        c, f = model['customers'], model['facilities']
        opened = [bool(value & (1 << j)) for j in range(f)]
        cost = sum(model['opening'][j] for j in range(f) if opened[j])
        for i in range(c):
            selected = [j for j in range(f) if value & (1 << (f+i*f+j))]
            if len(selected) != 1 or not opened[selected[0]]:
                return False, None
            cost += model['service'][i][selected[0]]
        return True, float(cost)
    if model['problem'] == 'tsp':
        m = model['nodes']-1; route = []
        for t in range(m):
            selected = [city for city in range(1, m+1) if value & (1 << (t*m+city-1))]
            if len(selected) != 1:
                return False, None
            route.append(selected[0])
        if len(set(route)) != m:
            return False, None
        route = [0]+route+[0]
        return True, float(sum(model['distance'][a][b] for a,b in zip(route, route[1:])))
    selected = [tuple(arc) for q,arc in enumerate(model['variables']) if value & (1 << q)]
    nodes = model['nodes']; successors = defaultdict(list); predecessors = defaultdict(list)
    for a,b in selected:
        successors[a].append(b); predecessors[b].append(a)
    for node in range(nodes):
        degree = 2 if node == 0 else 1
        if len(successors[node]) != degree or len(predecessors[node]) != degree:
            return False, None
    visited = set()
    for first in successors[0]:
        node = first
        while node != 0:
            if node in visited:
                return False, None
            visited.add(node); node = successors[node][0]
    if visited != set(range(1,nodes)):
        return False, None
    return True, float(sum(model['distance'][a][b] for a,b in selected))


def main():
    OUT.mkdir(exist_ok=True)
    inputs = {}; distributions = defaultdict(Counter); cost_distributions = defaultdict(Counter)
    metadata = {}; blocks = []; decoded_strings = 0; independent_feasible_strings = 0
    for study,(folder,models_folder) in STUDIES.items():
        base = ROOT/'outputs'/folder
        plan_path = base/'confirm_plan.json'; plan = read(plan_path); inputs[str(plan_path.relative_to(ROOT))] = sha(plan_path)
        analysis_path = base/'analysis.json'; analysis = read(analysis_path); inputs[str(analysis_path.relative_to(ROOT))] = sha(analysis_path)
        frozen = analysis.get('confirmation_raw_sha256', analysis.get('raw_sha256', {}))
        rows = plan['rows']; by_method = {(r.get('case_id','flp42'), r['method']): r for r in rows}
        model_cache = {}; feasible_cache = {}; seen = set()
        for path in sorted((base/'raw_counts').glob('confirm*.json')):
            relative = str(path.relative_to(ROOT)); checksum = sha(path)
            if frozen:
                assert relative in frozen and checksum == frozen[relative], relative
            inputs[relative] = checksum; raw = read(path)
            assert raw['phase'] == 'confirm'
            for r in raw['rows']:
                cid = r.get('case_id','flp42'); method = r['method']; spec = by_method[cid,method]
                assert r['candidate_id'] == spec['candidate_id']
                identity = (cid,method,raw['block']); assert identity not in seen; seen.add(identity)
                if cid not in model_cache:
                    model_path = (ROOT/'outputs'/models_folder/'cases'/cid/'model.json') if models_folder else base/'case.json'
                    model = read(model_path); model = model['model'] if not models_folder else model
                    inputs[str(model_path.relative_to(ROOT))] = sha(model_path)
                    model_cache[cid] = model; feasible_cache[cid] = feasible_costs(model)
                    assert abs(min(feasible_cache[cid].values())-model['optimum']) <= 1e-9
                model = model_cache[cid]; reference = feasible_cache[cid]
                key = (study,cid,METHODS[method]); counts = r['counts']; expected = r['metrics']
                assert all(len(b)==model['n_qubits'] and set(b)<=set('01') and isinstance(n,int) and n>0 for b,n in counts.items())
                shots = sum(counts.values()); assert shots == plan['shots_per_circuit'] == expected['shots']
                hits = dict(feasible=0,good=0,optimal=0); costs = Counter(); gaps = Counter()
                items = list(counts.items())
                for start in range(0,len(items),4096):
                    batch = items[start:start+4096]; values = np.array([int(b,2) for b,n in batch],dtype=np.uint64)
                    decoded = decode(model,values); decoded_strings += len(batch)
                    for i,(bits,weight) in enumerate(batch):
                        wanted = int(bits,2) in reference
                        assert bool(decoded['feasible'][i]) == wanted
                        if not wanted:
                            continue
                        valid,cost = scalar_check(model,bits); assert valid
                        assert abs(cost-reference[int(bits,2)]) <= 1e-9
                        assert abs(float(decoded['cost'][i])-cost) <= 1e-9
                        independent_feasible_strings += 1
                        gap = (cost-model['optimum'])/abs(model['optimum'])*100
                        assert gap >= -1e-8; gap = max(0.,gap)
                        costs[cost] += weight; gaps[gap] += weight; hits['feasible'] += weight
                        hits['good'] += weight*(cost <= 1.05*model['optimum']+1e-9)
                        hits['optimal'] += weight*(abs(cost-model['optimum']) <= 1e-9)
                for event,n in hits.items():
                    assert n == expected[event+'_count'], (study,cid,method,event,n,expected)
                assert (min(costs) if costs else None) == expected['best_feasible_cost'] or (
                    costs and abs(min(costs)-expected['best_feasible_cost']) <= 1e-9)
                blocks.append(dict(study=study,case_id=cid,method=METHODS[method],block=raw['block'],shots=shots,**{e+'_count':n for e,n in hits.items()}))
                distributions[key].update(gaps); cost_distributions[key].update(costs)
                if key not in metadata:
                    metadata[key] = dict(study=study,case_id=cid,family=model['problem'].upper(),n_qubits=model['n_qubits'],method=METHODS[method],optimum=model['optimum'],shots=0,good_count=0,optimal_count=0)
                metadata[key]['shots'] += shots; metadata[key]['good_count'] += hits['good']; metadata[key]['optimal_count'] += hits['optimal']
        assert seen == {(cid,method,block) for cid,method in by_method for block in [0,1]}
    assert len(metadata) == 136 and len(blocks) == 272
    cases = []
    for key in sorted(metadata):
        m = metadata[key]; costs = cost_distributions[key]; row = dict(m,**summarize_distribution(distributions[key],m['shots']))
        row.update(best_feasible_cost=min(costs) if costs else None,
                   mean_feasible_cost=sum(c*n for c,n in costs.items())/sum(costs.values()) if costs else None,
                   median_feasible_cost=weighted_quantile(list(costs),list(costs.values()),.5))
        cases.append(row)
    groups = [('FLP20','main','FLP',20),('TSP25','longer','TSP',25),('VRP20','longer','VRP',20),('FLP42','flp42','FLP',42)]
    grouped = []; group_distributions = {}
    for group,study,family,width in groups:
        for method in ['source','product','shallow','selected_native']:
            chosen = [k for k,m in metadata.items() if m['study']==study and m['family']==family and m['n_qubits']==width and m['method']==method]
            merged = Counter(); shots = 0
            for key in chosen:
                merged.update(distributions[key]); shots += metadata[key]['shots']
            row = dict(group=group,study=study,family=family,n_qubits=width,method=method,instances=len(chosen),shots=shots,
                       instances_with_feasible=sum(bool(distributions[k]) for k in chosen),
                       **summarize_distribution(merged,shots))
            grouped.append(row); group_distributions[group+'|'+method] = [dict(gap_pct=g,count=n) for g,n in sorted(merged.items())]
    distribution_rows = [dict(study=k[0],case_id=k[1],method=k[2],cost=c,count=n,gap_pct=max(0.,(c-metadata[k]['optimum'])/abs(metadata[k]['optimum'])*100))
                         for k in sorted(metadata) for c,n in sorted(cost_distributions[k].items())]
    csv_file(OUT/'per_case_summary.csv',cases); csv_file(OUT/'group_summary.csv',grouped)
    csv_file(OUT/'weighted_cost_distributions.csv',distribution_rows); csv_file(OUT/'by_block.csv',blocks)
    dump(OUT/'analysis.json',dict(scope='All independent main/longer/FLP42 confirmation samples; original objectives; descriptive conditional quality only.',
                                inputs_sha256=inputs,cases=cases,groups=grouped,group_distributions=group_distributions))
    dump(OUT/'validation.json',dict(cases=34,method_cases=136,raw_rows=272,decoded_unique_strings=decoded_strings,
                                  independently_checked_feasible_strings=independent_feasible_strings,
                                  every_saved_event_count_reproduced=True,scalar_vector_and_enumeration_agree=True,
                                  selection_counts_excluded=True,no_new_inferential_tests=True,no_new_hardware_jobs=True,
                                  input_sha256=inputs,source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)),
                                  'revision/width_sweep_models.py':sha(ROOT/'revision/width_sweep_models.py'),
                                  'revision/tensor_prefix_models.py':sha(ROOT/'revision/tensor_prefix_models.py')}))
    print('Verified all 136 method/case outcomes in 272 confirmation blocks.')
    for row in grouped:
        print(row['group'],row['method'],'feasible=',row['feasible_count'],'median gap=',row['median_gap_pct'],'IQR=',row['q25_gap_pct'],row['q75_gap_pct'],'best=',row['best_gap_pct'])


if __name__ == '__main__':
    main()
