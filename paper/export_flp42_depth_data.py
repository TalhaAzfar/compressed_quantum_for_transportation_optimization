"""Independently decode the completed FLP42 depth sweep for manuscript figures.

Only the standard library is needed. Plotting uses the resulting compact JSON;
this audit also requires the frozen experiment archive in the workspace.
No fitting, simulation, hardware submission, or new significance test is run.
"""
from pathlib import Path
import argparse
import hashlib
import json

from export_flp30_plot_data import rate_interval, difference_interval

HERE = Path(__file__).resolve().parent


def decode(model, counts):
    f,c = model['facilities'],model['customers']
    mask = (1 << f)-1
    result = dict(shots=0,feasible_count=0,optimal_count=0)
    for bits,count in counts.items():
        assert len(bits) == model['n_qubits'] and set(bits) <= set('01')
        assert isinstance(count,int) and count > 0
        result['shots'] += count
        value = int(bits,2)
        opened = value & mask
        assignment = [(value >> (f+i*f)) & mask for i in range(c)]
        if any(a.bit_count() != 1 or a & ~opened for a in assignment):
            continue
        result['feasible_count'] += count
        cost = sum(model['opening'][j] for j in range(f) if opened >> j & 1)
        cost += sum(model['service'][i][a.bit_length()-1] for i,a in enumerate(assignment))
        if abs(cost-model['optimum']) <= 1e-8:
            result['optimal_count'] += count
    return result


def export(workspace, destination):
    inputs = {}
    def read(path):
        inputs[path.relative_to(workspace).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    folder = workspace/'outputs/flp42_depth_extension_2026_10'
    model = read(folder/'case.json')['model']
    assert model['n_qubits'] == 42 and model['facilities'] == model['customers'] == 6
    jobs = read(folder/'jobs.json')
    assert len(jobs) == 4
    assert all(j['backend'] == 'ibm_rensselaer' and
               j['processor'] == dict(family='Nighthawk',revision='1') for j in jobs)
    analysis = read(folder/'analysis.json')
    selection = read(folder/'screen_selection.json')
    audit = read(folder/'completion_verification.json')
    assert audit['status'] == 'passed' and audit['screen_and_confirmation_counts_redecoded']
    assert audit['analysis_sha256'] == inputs[(folder/'analysis.json').relative_to(workspace).as_posix()]
    for name in ['preparation_protocol.json','hardware_protocol.json',
                 'confirmation_protocol.json','scheduling_amendment.json']:
        protocol = read(folder/name)
        for path,sha in protocol.get('input_sha256',{}).items():
            assert hashlib.sha256((workspace/path).read_bytes()).hexdigest() == sha
    plans = {phase:read(folder/f'{phase}_plan.json') for phase in ['screen','confirm']}
    native = [r for r in plans['screen']['rows'] if r['method'].startswith('aqc_l')]
    assert [r['resources']['cz_depth'] for r in native] == [18,36,54,90,126,162,198]
    assert all(r['resources']['swap_count'] == 0 for r in native)
    selected = plans['confirm']['rows'][0]
    assert selected['method'] == 'selected_native'
    assert selected['candidate_id'] == selection['selected']['candidate_id'] == 'l2_r0'
    indexed,checked,rates = {},[],[]
    for phase,plan in plans.items():
        assert plan['blocks'] == 2
        for block in [0,1]:
            path = folder/'raw_counts'/f'{phase}_b{block}.json'
            raw = read(path)
            job = next(j for j in jobs if (j['phase'],j['block']) == (phase,block))
            assert raw['job_id'] == job['job_id'] and raw['phase'] == phase and raw['block'] == block
            assert [r['method'] for r in raw['rows']] == [r['method'] for r in job['rows']]
            for row in raw['rows']:
                metric = decode(model,row['counts'])
                assert all(metric[k] == row['metrics'][k] for k in metric)
                assert metric['shots'] == plan['shots_per_circuit']
                key = phase,row['method'],block
                assert key not in indexed
                indexed[key] = metric
                checked.append(dict(phase=phase,method=row['method'],block=block,
                                    job_id=job['job_id'],**metric))
        expected = selection['pooled'] if phase == 'screen' else analysis['pooled']
        for row in plan['rows']:
            blocks = [indexed[phase,row['method'],b] for b in [0,1]]
            original = next(r for r in expected if r['method'] == row['method'])
            assert all(sum(b[k] for b in blocks) == original[k] for k in blocks[0])
            p,lo,hi = rate_interval(blocks,'feasible')
            rates.append(dict(study='flp42_depth_sweep',case_id='flp42_r1_depth_sweep',
                 family='FLP',n_qubits=42,phase=phase,method=row['method'],
                 candidate_id=row['candidate_id'],cz_depth=row['resources']['cz_depth'],
                 cz_count=row['resources']['cz_count'],swap_count=row['resources']['swap_count'],
                 fidelity=row['fidelity'],ideal_p_feasible=row['ideal_events']['p_feasible'],
                 shots=original['shots'],feasible_count=original['feasible_count'],
                 optimal_count=original['optimal_count'],
                 block_feasible_counts=[b['feasible_count'] for b in blocks],
                 p_feasible_percent=100*p,ci_low_percent=100*lo,ci_high_percent=100*hi,
                 selected_original_shallow=(row['candidate_id']=='l2_r0')))
    assert len(checked) == 28 and len(rates) == 14
    effects = []
    for row in analysis['comparisons']:
        a = [indexed['confirm','selected_native',b] for b in [0,1]]
        b = [indexed['confirm',row['reference'],i] for i in [0,1]]
        p,lo,hi = difference_interval(a,b,row['event'])
        assert all(abs(x-row[key]) < 1e-12 for x,key in [(p,'difference'),(lo,'low'),(hi,'high')])
        effects.append(dict(study='flp42_depth_sweep',case_id='flp42_r1_depth_sweep',
             family='FLP',n_qubits=42,event=row['event'],
             reference={'product':'refined_product','shallow':'fixed_shallow'}.get(row['reference'],row['reference']),
             compiled_method='selected_native',candidate_id=selected['candidate_id'],cz_depth=18,
             improvement_pp=100*p,ci_low_pp=100*lo,ci_high_pp=100*hi,
             conservative_low_pp=100*row['conservative_low'],
             conservative_high_pp=100*row['conservative_high'],
             p_value=row['p_value'],holm_p_value=row['holm_p_value'],holm_family_size=6,
             identical_circuit_self_control=row['identical_circuit_self_control'],
             compiled_count=sum(r[row['event']+'_count'] for r in a),
             reference_count=sum(r[row['event']+'_count'] for r in b),
             shots_per_method=sum(r['shots'] for r in a),selection_shots_excluded=True))
    fits = []
    for path in sorted((folder/'fits').glob('*.json')):
        fit = read(path)
        if fit.get('index',0) < 2:
            continue
        assert fit['verified']
        fits.append({k:fit[k] for k in ['layers','index','fidelity','initial_fidelity',
                    'evaluations','ansatz_bond_convergence','objective_difference',
                    'status','ideal_events']})
    assert len(fits) == 14
    result = dict(scope='One six-facility/six-customer FLP42 instance, the same complete '
                  '12-step annealing target and physical patch as the initial FLP42 study. '
                  'Additional depths are preparations, not additional problem instances.',
        backend='ibm_rensselaer',processor=dict(family='Nighthawk',revision='1'),
        intervals='Pointwise 95% Wilson-based block means for rates; block MOVER for differences. '
                  'Screening intervals are descriptive and do not correct for circuit selection.',
        inference='Only the independent confirmation enters the separate exploratory six-test '
                  'Holm family. The selected original shallow circuit is repeated as a self-control. '
                  'No pooling with the initial FLP42 acquisition.',
        raw_count_sets_independently_decoded=len(checked),all_saved_counts_reproduced=True,
        all_six_confirmation_effects_and_intervals_reproduced=True,
        selected_candidate_id=selected['candidate_id'],rates=rates,effects=effects,
        fit_summaries=fits,by_block=checked,input_sha256=inputs)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print('FLP42 depth sweep verified:',len(checked),'count sets,',len(rates),'rates,',
          len(effects),'confirmation contrasts,',len(fits),'fits.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,default=HERE.parent)
    parser.add_argument('--output',type=Path,default=HERE/'data/flp42_depth_sweep.json')
    args = parser.parse_args()
    export(args.workspace_root.resolve(),args.output.resolve())
