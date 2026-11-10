"""Selection, decoding and inferential checks before costly QPU execution."""
import pytest
from revision import long_anneal_hardware as hw
from revision.contribution_hardware_core import read,save,digest


def test_ambiguous_submission_and_incomplete_confirmation_are_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(hw,'OUT',tmp_path)
    save(tmp_path/'intent_screen_d0_TSP_b0.json',{})
    with pytest.raises(RuntimeError,match='Ambiguous'):hw.submission_state('screen',0,0,'TSP')
    with pytest.raises(AssertionError,match='incomplete'):hw.analyze()


def test_identical_controls_and_zero_optimum_counts_keep_uncertainty():
    a=dict(case_id='c',method='selected_native',family='TSP',n_qubits=25,isa_sha256='same',resources=dict(cz_depth=12))
    b=dict(a,method='fixed_shallow');index={}
    for block in [0,1]:
        index['c','selected_native',block]=dict(feasible_count=100,optimal_count=0,shots=32768)
        index['c','fixed_shallow',block]=dict(feasible_count=1,optimal_count=0,shots=32768)
    self_control=hw.contrast([a,b],[index],'fixed_shallow','feasible')
    assert self_control['p_value']==1 and self_control['improvement_pp']>0
    zero=hw.contrast([a,b],[index],'fixed_shallow','optimal')
    assert zero['ci_low_pp']<0<zero['ci_high_pp'] and zero['p_value']==1


def test_independent_confirmation_can_reverse_all_six_screening_winners(tmp_path,monkeypatch):
    from revision.width_sweep_models import feasible_costs,metrics
    c=hw.settings();actual_root=hw.ROOT;out=tmp_path/'study';out.mkdir()
    monkeypatch.setattr(hw,'OUT',out);monkeypatch.setattr(hw,'ROOT',tmp_path)
    monkeypatch.setattr(hw,'check_frozen',lambda:None);monkeypatch.setattr(hw,'report',lambda:None)
    models={};rows=[]
    for spec in c['cases']:
        cid=spec['id'];model=read(actual_root/'outputs/hardware_width_sweep_2026/cases'/cid/'model.json');models[cid]=model
        save(out/'cases'/cid/'model.json',model)
        for method,layers,depth in [('source',None,8000),('refined_product',0,0),('fixed_shallow',2,16),('candidate_3',8,64)]:
            name=cid+'_'+method
            rows.append(dict(case_id=cid,candidate_id=name,method=method,family=spec['problem'].upper(),
                n_qubits=model['n_qubits'],layers=layers,exact_fidelity=.9,
                ideal_events=dict(p_feasible=.05,p_optimal=.0007),resources=dict(cz_depth=depth,cz_count=depth*2),
                logical_qpy=name+'.qpy',logical_sha256=name,isa_qpy=name+'_isa.qpy',isa_sha256=name))
    screen=dict(rows=rows,blocks=2,shots_per_circuit=2048,total_shots=len(rows)*4096)
    save(out/'screen_plan.json',screen)
    save(out/'hardware_protocol.json',dict(source_sha256={},input_sha256={},screen_plan_sha256=digest(out/'screen_plan.json')))
    jobs=[]
    def acquire(phase,plan):
        for block in [0,1]:
            for family in hw.FAMILIES:
                subset=[r for r in plan['rows'] if r['family']==family];job_id=f'{phase}_{family}_{block}';raw=[]
                for row in subset:
                    model=models[row['case_id']];costs=feasible_costs(model);optimum=min(costs,key=costs.get)
                    k=(70 if row['method']=='candidate_3' else 10) if phase=='screen' else (2 if row['method']=='selected_native' else 30)
                    counts={format(optimum,f"0{model['n_qubits']}b"):k,'0'*model['n_qubits']:plan['shots_per_circuit']-k}
                    raw.append(dict(case_id=row['case_id'],method=row['method'],candidate_id=row['candidate_id'],counts=counts,metrics=metrics(model,counts)))
                jobs.append(dict(phase=phase,day=0,block=block,family=family,job_id=job_id,rows=subset,
                    backend='ibm_rensselaer',processor=dict(family='Nighthawk',revision='1'),
                    total_shots=len(subset)*plan['shots_per_circuit'],plan_sha256=digest(out/('screen_plan.json' if phase=='screen' else 'confirm_plan.json'))))
                save(out/'raw_counts'/f'{phase}_d0_{family}_b{block}.json',dict(job_id=job_id,phase=phase,day=0,block=block,family=family,rows=raw))
        save(out/'jobs.json',jobs)
    acquire('screen',screen);hw.select()
    confirm=read(out/'confirm_plan.json');frozen=digest(out/'screen_selection.json')
    assert confirm['total_shots']==1572864 and len(confirm['rows'])==24
    assert all(r['screen_method']=='candidate_3' for r in confirm['rows'] if r['method']=='selected_native')
    acquire('confirm',confirm);save(out/'day0_complete.json',{});hw.analyze()
    assert digest(out/'screen_selection.json')==frozen
    result=read(out/'analysis.json');assert len(result['comparisons'])==36
    assert all(r['selected_count']==4 and r['selected_shots']==65536 and r['loss'] for r in result['comparisons'])
    assert sum(r['corrected_losses'] for r in result['summary'])==36
    # Stored summary metrics cannot override independently decoded raw counts.
    path=out/'raw_counts/confirm_d0_TSP_b0.json';raw=read(path);raw['rows'][0]['metrics']['optimal_count']+=1;save(path,raw)
    with pytest.raises(AssertionError):hw.aggregate('confirm',0)
