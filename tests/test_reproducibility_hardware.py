import copy
import pytest


def fake_case():
    rows=[]
    for label,layers,cap in [('product',0,0),('base',2,16),('d32',4,32),('d64',8,64)]:
        for start in range(3):
            cid=label+str(start)
            rows.append(dict(id=cid,candidate_id=cid,case_id='case',family='FLP',n_qubits=20,layers=layers,
                exact_fidelity=.5+.05*layers+.005*start,ideal_events=dict(p_feasible=.05+.001*start-.001*layers),
                fit=dict(verified=True,adequate_evaluation_budget=True),cz_depth_cap=cap,
                resources=dict(cz_depth=cap,cz_count=10*cap),logical_qpy=cid,logical_sha256=cid,
                isa_qpy=cid,isa_sha256=cid))
    source=dict(rows[0],id='source',candidate_id='source',layers=None)
    return dict(rows=rows,source=source)


def test_candidates_keep_shallow_and_product_controls_and_adequacy_gate():
    from revision.reproducibility_hardware import candidates
    case=fake_case();chosen=candidates(case)
    assert {r['method'] for r in chosen}>={'source','refined_product','fixed_shallow'}
    assert len(chosen)<=9
    assert next(r for r in chosen if r['method']=='fixed_shallow')['layers']==2
    assert any(r['layers']==8 for r in chosen)
    case['rows'][-1]['fit']['adequate_evaluation_budget']=False
    with pytest.raises(AssertionError):candidates(case)


def test_second_day_requires_elapsed_time_and_new_calibration_date():
    from revision.reproducibility_hardware import calibration_gate
    first=dict(last_completed_utc='2026-10-06T23:00:00+00:00',calibration_timestamps=['2026-10-06T18:00:00-04:00'])
    assert not calibration_gate(1,'2026-10-07T01:00:00-04:00','2026-10-07T06:00:00+00:00',first)
    assert not calibration_gate(1,'2026-10-06T18:00:00-04:00','2026-10-08T01:00:00+00:00',first)
    assert calibration_gate(1,'2026-10-07T20:00:00-04:00','2026-10-08T01:00:00+00:00',first)
    assert not calibration_gate(1,'2026-10-08T00:00:00+00:00','2026-10-08T01:00:00+00:00',None)


def test_ambiguous_submission_cannot_be_automatically_repeated(tmp_path,monkeypatch):
    from revision import reproducibility_hardware as hw
    from revision.contribution_hardware_core import save
    monkeypatch.setattr(hw,'OUT',tmp_path)
    save(tmp_path/'intent_screen_d0_FLP_b0.json',{})
    with pytest.raises(RuntimeError,match='Ambiguous'):hw.submission_state('screen',0,0,'FLP')
    save(tmp_path/'jobs.json',[dict(phase='screen',day=0,block=0,family='FLP',job_id='saved')])
    assert hw.submission_state('screen',0,0,'FLP')[1]


def test_identical_circuits_are_self_controls_despite_measured_difference():
    from revision.reproducibility_hardware import contrast
    a=dict(case_id='c',method='selected_native',family='FLP',n_qubits=20,isa_sha256='same',resources=dict(cz_depth=16))
    b=dict(a,method='fixed_shallow');index={}
    for block in range(2):
        index['c','selected_native',block]=dict(feasible_count=100,shots=1000)
        index['c','fixed_shallow',block]=dict(feasible_count=1,shots=1000)
    result=contrast([a,b],[index],'fixed_shallow','feasible')
    assert result['identical_circuit_self_control'] and result['p_value']==1
    assert result['improvement_pp']>0


def test_complete_analysis_cannot_be_claimed_from_one_day(tmp_path,monkeypatch):
    from revision import reproducibility_hardware as hw
    from revision.contribution_hardware_core import save
    monkeypatch.setattr(hw,'OUT',tmp_path)
    save(tmp_path/'day0_complete.json',{})
    with pytest.raises(AssertionError):hw.analyze()


def test_holm_jointly_retains_unfavorable_directions():
    from revision.reproducibility_hardware import corrected
    rows=[dict(p_value=.001,improvement_pp=-2),dict(p_value=.02,improvement_pp=1),dict(p_value=.8,improvement_pp=3)]
    r=corrected(rows)
    assert r[0]['loss'] and not r[0]['gain']
    assert r[1]['gain'] and not r[2]['gain']
