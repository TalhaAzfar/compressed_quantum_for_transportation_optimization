"""Selection, acquisition safety and confirmation inference regression tests."""
import json
import pytest
from revision import native_depth_hardware as hardware
from revision import native_depth_analysis as analysis


def candidate(name, layers, fidelity, feasible, depth, method=None):
    return dict(case_id='flp6_r1',candidate_id=name,method=method or name,
        family='FLP',n_qubits=6,layers=layers,exact_fidelity=fidelity,
        ideal_events={'p_feasible':feasible,'p_optimal':feasible/2},
        resources={'cz_depth':depth,'cz_count':depth*2},
        logical_qpy=name+'.qpy',logical_sha256=name,
        isa_qpy=name+'_isa.qpy',isa_sha256=name)


def test_screen_ranking_excludes_controls_and_has_stable_ties():
    rows=[candidate('source',None,1,1,500),candidate('product',0,.5,1,0),
          candidate('deep',8,.99,.4,64),candidate('shallow_b',2,.8,.4,16),
          candidate('shallow_a',2,.8,.4,16)]
    pooled=[dict(case_id=r['case_id'],method=r['method'],shots=4096,
        feasible_count=4000 if r['layers'] in (None,0) else 100,p_feasible=.1) for r in rows]
    ranked=analysis.rank_native(rows,pooled)
    assert [r['candidate_id'] for r in ranked]==['shallow_a','shallow_b','deep']
    pooled[-1]['feasible_count']=99
    assert analysis.rank_native(rows,pooled)[0]['candidate_id']=='shallow_b'
    assert analysis.rank_native(list(reversed(rows)),pooled)[0]['candidate_id']=='shallow_b'


def test_candidate_pool_preserves_better_old_fit_and_depth_caps(monkeypatch):
    old=[candidate('parent_source',None,1,.4,500),candidate('parent_product',0,.5,.1,0),
         candidate('parent_aqc_l2',2,.90,.3,16),candidate('parent_aqc_l6',6,.94,.35,48)]
    new=[]
    for name,layers,fidelity,feasible,depth in [('product',0,.6,.12,0),('base',2,.85,.25,16),
                                              ('d32',4,.89,.31,32),('d64',8,.92,.34,64)]:
        r=candidate(name,layers,fidelity,feasible,depth)
        r.update(id=name,adequate_evaluation_budget=True,verified=True);new.append(r)
    monkeypatch.setattr(hardware,'parent_rows',lambda cid:({'model':{'problem':'flp','n_qubits':6}},old))
    monkeypatch.setattr(hardware,'read',lambda path:{'rows':new,'all_fits_verified':True})
    chosen=hardware.pick_candidates('flp6_r1')
    ids=[r['candidate_id'] for r in chosen]
    assert ids==['parent_source','parent_aqc_l2','product','d32','parent_aqc_l6']
    assert 'base_exact_fidelity' in chosen[1]['selection_reasons']
    assert 'd32_exact_fidelity' in chosen[1]['selection_reasons']
    assert 'd64_exact_fidelity' in chosen[-1]['selection_reasons']
    assert len(ids)==len(set(ids))
    new[-1]['adequate_evaluation_budget']=False
    with pytest.raises(AssertionError,match='inadequately'): hardware.pick_candidates('flp6_r1')


def test_submission_guard_rejects_ambiguous_retries(tmp_path,monkeypatch):
    monkeypatch.setattr(hardware,'OUT',tmp_path)
    assert hardware.submission_state('screen',0,'FLP')==([],False)
    (tmp_path/'intent_screen_FLP_0.json').write_text('{}')
    with pytest.raises(RuntimeError,match='Ambiguous'):hardware.submission_state('screen',0,'FLP')
    job=dict(phase='screen',block=0,family='FLP',job_id='known')
    (tmp_path/'jobs.json').write_text(json.dumps([job]))
    assert hardware.submission_state('screen',0,'FLP')==([job],True)
    assert hardware.submission_state('confirm',0,'FLP')==([job],False)


def test_confirmation_contrast_uses_independent_counts_and_self_control():
    selected=candidate('winner',4,.9,.3,32,'selected_native')
    ref=candidate('parent_aqc_l2',2,.8,.2,16,'old_aqc_l2')
    indexed={}
    for b in [0,1]:
        indexed['flp6_r1','selected_native',b]={'feasible_count':40,'optimal_count':0,'shots':100}
        indexed['flp6_r1','old_aqc_l2',b]={'feasible_count':10,'optimal_count':0,'shots':100}
    row=analysis.contrast([selected,ref],indexed,'old_aqc_l2','feasible')
    assert row['improvement_pp']==pytest.approx(30)
    assert row['p_value']<.001 and row['ci_low_pp']>0
    assert row['conservative_ci_low_pp']<=row['ci_low_pp']
    zero=analysis.contrast([selected,ref],indexed,'old_aqc_l2','optimal')
    assert zero['improvement_pp']==0 and zero['ci_low_pp']<0<zero['ci_high_pp']
    selected['isa_sha256']=ref['isa_sha256']
    self_control=analysis.contrast([selected,ref],indexed,'old_aqc_l2','feasible')
    assert self_control['identical_circuit_self_control'] and self_control['p_value']==1
    assert self_control['improvement_pp']==pytest.approx(30)  # Raw repeat variability retained.


def test_confirmation_family_and_acquisition_budget():
    config=hardware.settings()
    assert len(config['cases'])*len(analysis.EVENTS)*len(analysis.REFERENCES)==162
    screen=len(config['cases'])*9*config['hardware_screen_blocks']*config['hardware_screen_shots_per_circuit']
    confirm=len(config['cases'])*4*config['hardware_confirmation_blocks']*config['hardware_confirmation_shots_per_circuit']
    assert screen+confirm==config['hardware_max_total_shots']==1880064
    assert config['aggregate_rss_limit_bytes']==35000000000


def test_rate_interval_retains_zero_count_uncertainty():
    result=analysis.rate_interval([{'feasible_count':0,'shots':4096}]*2,'feasible')
    assert result[0]==result[1]==0 and result[2]>0
    result=analysis.rate_interval([{'feasible_count':4096,'shots':4096}]*2,'feasible')
    assert result[0]==result[2]==1 and result[1]<1


def test_full_screen_selection_and_independent_confirmation(tmp_path,monkeypatch):
    """Exercise all 27 cases and 162 tests with deliberately reversed outcomes."""
    from revision.width_sweep_models import feasible_costs,metrics
    from revision.contribution_hardware_core import save,read,digest
    config=hardware.settings();real_parent=analysis.PARENT
    out=tmp_path/'study';parent=tmp_path/'parent';out.mkdir()
    monkeypatch.setattr(analysis,'OUT',out);monkeypatch.setattr(analysis,'ROOT',tmp_path)
    monkeypatch.setattr(analysis,'PARENT',parent)
    monkeypatch.setattr(analysis,'settings',lambda:config)
    monkeypatch.setattr(analysis,'plots',lambda rows:None)
    monkeypatch.setattr(analysis,'classical_report',lambda:None)
    models={};rows=[]
    for spec in config['cases']:
        cid=spec['id'];model=read(real_parent/'cases'/cid/'model.json');models[cid]=model
        save(parent/'cases'/cid/'model.json',model)
        for name,layers,depth in [('source',None,500),('old_aqc_l2',2,16),
                                  ('refined_product',0,0),('candidate_3',4,32)]:
            row=candidate(name,layers,.9,.3,depth)
            row.update(case_id=cid,family=spec['problem'].upper(),n_qubits=model['n_qubits'])
            rows.append(row)
    screen=dict(phase='screen',rows=rows,blocks=2,shots_per_circuit=2048,total_shots=len(rows)*4096)
    save(out/'screen_plan.json',screen)
    save(out/'hardware_protocol.json',dict(source_sha256={},screen_plan_sha256=digest(out/'screen_plan.json')))
    jobs=[]
    def fake_acquisition(phase,plan):
        for block in [0,1]:
            for family in ['FLP','TSP','VRP']:
                subset=[r for r in plan['rows'] if r['family']==family]
                job_id=f'{phase}_{family}_{block}';raw=[]
                for row in subset:
                    model=models[row['case_id']];costs=feasible_costs(model)
                    optimum=min(costs,key=costs.get);n=model['n_qubits'];shots=plan['shots_per_circuit']
                    if phase=='screen':k=30 if row['method']=='candidate_3' else 10
                    else:k=2 if row['method']=='selected_native' else 15
                    counts={format(optimum,f'0{n}b'):k,'0'*n:shots-k}
                    raw.append(dict(case_id=row['case_id'],method=row['method'],candidate_id=row['candidate_id'],
                                    counts=counts,metrics=metrics(model,counts)))
                jobs.append(dict(phase=phase,block=block,family=family,job_id=job_id,rows=subset,
                    total_shots=len(subset)*plan['shots_per_circuit'],backend='ibm_rensselaer',
                    processor={'family':'Nighthawk','revision':'1'},plan_sha256=digest(out/(phase+'_plan.json'))))
                save(out/'raw_counts'/f'{phase}_{family}_b{block}.json',
                     dict(job_id=job_id,phase=phase,family=family,block=block,rows=raw))
        save(out/'jobs.json',jobs)
    fake_acquisition('screen',screen)
    analysis.select()
    selection=read(out/'screen_selection.json')
    assert len(selection['selected'])==27
    assert all(r['candidate_id']=='candidate_3' for r in selection['selected'])
    confirmation=read(out/'confirm_plan.json')
    assert confirmation['total_shots']==884736 and len(confirmation['rows'])==108
    frozen=digest(out/'screen_selection.json')
    fake_acquisition('confirm',confirmation)
    analysis.analyze()
    assert digest(out/'screen_selection.json')==frozen
    result=read(out/'analysis.json');effects=result['comparisons']
    assert len(effects)==162 and result['hardware_jobs']==12
    assert all(r['improvement_pp']<0 and r['selected_count']==4 for r in effects)
    # A saved decoder result must not be trusted when it disagrees with counts.
    path=out/'raw_counts'/'confirm_FLP_b0.json';raw=read(path)
    raw['rows'][0]['metrics']['feasible_count']+=1;save(path,raw)
    with pytest.raises(AssertionError):analysis.aggregate('confirm')


def test_publication_figures_keep_all_cases_and_sparse_intervals(tmp_path,monkeypatch):
    monkeypatch.setattr(analysis,'OUT',tmp_path)
    rows=[];flat=[];best=[]
    for spec in hardware.settings()['cases']:
        cid=spec['id'];family=spec['problem'].upper()
        n=int(''.join(c for c in cid.split('_')[0] if c.isdigit()))
        for event in analysis.EVENTS:
            for reference in analysis.REFERENCES:
                rows.append(dict(case_id=cid,family=family,n_qubits=n,event=event,reference=reference,
                    improvement_pp=0.,ci_low_pp=-.01,ci_high_pp=.01,
                    conservative_ci_low_pp=-.02,conservative_ci_high_pp=.02,
                    identical_circuit_self_control=reference=='old_aqc_l2'))
        for cap in [0,16,32,64]:
            values=dict(case_id=cid,family=family,n_qubits=n,cz_depth=cap,exact_fidelity=.5+cap/200,
                        ideal_p_feasible=.001,ideal_p_optimal=.0001)
            best.append(dict(values,cz_depth_cap=cap,criterion='fidelity'))
        flat.append(dict(values,candidate_id='parent_source'))
    analysis.plots(rows);analysis.classical_plots(flat,best)
    assert len(list((tmp_path/'figures').glob('*.pdf')))==15
    assert len(list((tmp_path/'figures').glob('*.png')))==15
    assert len(list((tmp_path/'figures').glob('*.svg')))==15
