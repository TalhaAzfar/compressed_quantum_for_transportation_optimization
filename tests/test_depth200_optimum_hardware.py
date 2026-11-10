import pytest
from revision.depth200_optimum_hardware import select_candidates


def row(name,layers,depth,fidelity,verified=True,start=0):
    return dict(name=name,layers=layers,resources={'cz_depth':depth},fidelity=fidelity,verified=verified,start=start)


def test_selection_respects_depth_bands_and_independent_validation():
    rows=[row('product',0,0,.6),row('shallow',2,16,.7),row('middle',6,48,.8),
          row('deeper',12,96,.9),row('deepest',24,192,.88),row('bad',24,192,.999,False),
          row('over_cap',26,208,.9999)]
    selected=select_candidates(rows)
    assert {k:v['name'] for k,v in selected.items()}==dict(product='product',compressed_l2='shallow',compressed_l6='middle',compressed_l12='deeper',compressed_l24='deepest')


def test_missing_verified_deep_candidate_prevents_submission_plan():
    with pytest.raises(ValueError):select_candidates([row('product',0,0,.6),row('shallow',2,16,.7)])


def test_analysis_independently_checks_counts_and_zero_event_intervals(tmp_path,monkeypatch):
    import json
    from revision import depth200_optimum_hardware as hw
    from revision.depth200_optimum_screen import case_data,feasible_values
    from revision.tensor_prefix_models import decode,count_metrics
    monkeypatch.setattr(hw,'OUT',tmp_path)
    methods=['original','product','compressed_l2','compressed_l6','compressed_l12','compressed_l24'];cases=['tsp25','flp30'];arms=['raw','suppressed']
    rows=[];optimal={}
    for cid in cases:
        model=case_data(cid)['model'];values=feasible_values(model);labels=decode(model,values)
        optimal[cid]=format(next(v for v,ok in zip(values,labels['optimal']) if ok),f"0{model['n_qubits']}b")
        for method in methods:
            rows.append(dict(case_id=cid,method=method,resources={'cz_depth':16,'cz_count':100},fidelity=.7,
                             events={'p_feasible':.1,'p_optimal':.01}))
    (tmp_path/'hardware_plan.json').write_text(json.dumps(dict(cases=cases,arms=arms,blocks=2,rows=rows,planned_total_shots=960,primary_methods={c:'compressed_l12' for c in cases})))
    (tmp_path/'raw_counts').mkdir()
    for block in range(2):
        for arm in arms:
            data=[]
            for cid in cases:
                model=case_data(cid)['model'];zero='0'*model['n_qubits']
                for method in methods:
                    counts={zero:20} if method!='compressed_l12' else {zero:15,optimal[cid]:5}
                    data.append(dict(case_id=cid,method=method,counts=counts,metrics=count_metrics(model,counts)))
            (tmp_path/'raw_counts'/f'b{block}_{arm}.json').write_text(json.dumps(dict(block=block,arm=arm,rows=data,job_metrics={})))
    hw.analyze()
    result=json.loads((tmp_path/'hardware_analysis.json').read_text())
    assert result['total_shots']==960 and result['independent_count_sets']==48
    assert len(result['primary_tests'])==8
    assert len(result['secondary_tests'])==64
    for r in result['rows']:
        assert r['shots']==40
        assert r['optimal_count']==(10 if r['method']=='compressed_l12' else 0)
        if r['method']!='compressed_l12':
            assert r['optimal_ci_low']==0 and 0<r['optimal_ci_high']<.1
    for r in result['primary_tests']:
        assert r['p_value']<=r['p_holm']<=1
