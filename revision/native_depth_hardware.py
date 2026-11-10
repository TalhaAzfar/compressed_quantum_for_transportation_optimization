"""Screen refined native circuits, then confirm selected circuits independently."""
import argparse
import json
from pathlib import Path
from .native_depth_refinement import ROOT,OUT,PARENT,settings
from .contribution_hardware_core import read,save,digest,now,check_isa


def candidate_row(case, row):
    model=case['model']
    return dict(case_id=row['case_id'],family=model['problem'].upper(),
        n_qubits=model['n_qubits'],candidate_id=row['id'],layers=row['layers'],
        exact_fidelity=row['exact_fidelity'],ideal_events=row['ideal_events'],
        logical_qpy=row['logical_qpy'],logical_sha256=row['logical_sha256'],
        isa_qpy=row['isa_qpy'],isa_sha256=row['isa_sha256'],resources=row['resources'])


def parent_rows(cid):
    case=read(PARENT/'cases'/cid/'complete.json')
    plan=read(PARENT/'hardware_plan.json')
    rows=[]
    for row in plan['rows']:
        if row['case_id']!=cid: continue
        fit=row['fit']
        rows.append(dict(case_id=cid,family=row['family'],n_qubits=row['n_qubits'],
            candidate_id='parent_'+row['method'],layers=fit['layers'] if fit else None,
            exact_fidelity=fit['exact_fidelity'] if fit else 1.,ideal_events=row['ideal_events'],
            logical_qpy=row['logical_qpy'],logical_sha256=row['logical_sha256'],
            isa_qpy=row['isa_qpy'],isa_sha256=row['isa_sha256'],resources=row['resources']))
    return case,rows


def pick_candidates(cid):
    completed=read(OUT/'cases'/cid/'complete.json')
    assert completed['all_fits_verified']
    assert all(r['adequate_evaluation_budget'] for r in completed['rows']), 'Continue inadequately optimized fits before hardware'
    case,old=parent_rows(cid)
    new=[candidate_row(case,r) for r in completed['rows']]
    all_fits=[r for r in old+new if r['layers'] is not None]
    product=max([r for r in all_fits if r['layers']==0],key=lambda r:(r['exact_fidelity'],r['candidate_id']))
    chosen=[dict(next(r for r in old if r['candidate_id']=='parent_source'),method='source',selection_reasons=['source']),
            dict(next(r for r in old if r['candidate_id']=='parent_aqc_l2'),method='old_aqc_l2',selection_reasons=['frozen L2']),
            dict(product,method='refined_product',selection_reasons=['highest-fidelity product'])]
    by_id={r['candidate_id']:r for r in chosen}
    base_cap=chosen[1]['resources']['cz_depth']
    for label,cap in [('base',base_cap),('d32',32),('d64',64)]:
        pool=[r for r in all_fits if r['layers']>0 and r['resources']['cz_depth']<=cap]
        for metric in ['exact_fidelity','feasible']:
            key=(lambda r:(r['exact_fidelity'],-r['resources']['cz_depth'],r['candidate_id'])) if metric=='exact_fidelity' else (
                lambda r:(r['ideal_events']['p_feasible'],r['exact_fidelity'],-r['resources']['cz_depth'],r['candidate_id']))
            best=max(pool,key=key); reason=label+'_'+metric
            if best['candidate_id'] in by_id: by_id[best['candidate_id']]['selection_reasons'].append(reason)
            else:
                item=dict(best,method='candidate_'+str(len(chosen)),selection_reasons=[reason])
                chosen.append(item);by_id[item['candidate_id']]=item
    assert len(chosen)<=9
    return chosen


def prepare_screen():
    import pickle
    from .nighthawk_study import qpy_read
    assert (OUT/'preparation_complete.json').exists()
    assert not (OUT/'hardware_protocol.json').exists()
    c=settings(); target=pickle.loads((PARENT/'preflight/target.pkl').read_bytes())
    preparation=read(OUT/'preparation_protocol.json')
    amendment_path=OUT/'amendments'/'contraction_planning_v2'/'protocol.json'
    amendment=read(amendment_path)
    cpu_protocol_path=OUT/'cpu_placement_protocol.json'
    cpu_protocol=read(cpu_protocol_path)
    assert all(digest(ROOT/p)==h for p,h in cpu_protocol['source_sha256'].items())
    assert all(digest(ROOT/p)==h for p,h in amendment['source_sha256'].items())
    assert all(digest(ROOT/p)==h for p,h in amendment['input_sha256'].items())
    assert read(OUT/'preparation_complete.json')['amendment_sha256']==digest(amendment_path)
    for field in ['source_sha256','input_sha256']:
        assert all(digest(ROOT/p)==h for p,h in preparation[field].items())
    rows=[]; inputs=dict(amendment['input_sha256'])
    for spec in c['cases']:
        cid=spec['id']; selected=pick_candidates(cid)
        for row in selected:
            for field in ['logical','isa']:
                path=ROOT/row[field+'_qpy'];assert digest(path)==row[field+'_sha256']
                inputs[str(path.relative_to(ROOT))]=digest(path)
            resources=row['resources']; check_isa(qpy_read(ROOT/row['isa_qpy']),target,resources['physical'],resources['final_logical_mapping'])
        rows.extend(selected)
        for path in (OUT/'cases'/cid).glob('*.json'): inputs[str(path.relative_to(ROOT))]=digest(path)
    shots=len(rows)*c['hardware_screen_blocks']*c['hardware_screen_shots_per_circuit']
    assert shots+27*4*2*4096<=c['hardware_max_total_shots']
    plan=dict(created_utc=now(),phase='screen',rows=rows,total_shots=shots,blocks=2,shots_per_circuit=2048,
              selection_uses_ideal_data_only=True)
    save(OUT/'screen_plan.json',plan)
    names=['revision/native_depth_hardware.py','revision/native_depth_analysis.py','revision/native_depth_pipeline.py',
           'revision/hardware_width_figure.py','revision/audit_tensor_prefix_statistics.py',
           'tests/test_native_depth_hardware.py']
    tests=(OUT/'hardware_tests.log').read_text();assert 'passed' in tests and 'FAILED' not in tests and 'ERROR' not in tests
    hashes={**preparation['source_sha256'],**amendment['source_sha256'],**cpu_protocol['source_sha256'],**{p:digest(ROOT/p) for p in names}}
    for p in names: (OUT/'source_snapshot'/Path(p).name).write_bytes((ROOT/p).read_bytes())
    inputs[str((OUT/'preparation_protocol.json').relative_to(ROOT))]=digest(OUT/'preparation_protocol.json')
    inputs[str(amendment_path.relative_to(ROOT))]=digest(amendment_path)
    inputs[str(cpu_protocol_path.relative_to(ROOT))]=digest(cpu_protocol_path)
    save(OUT/'hardware_protocol.json',dict(frozen_utc=now(),config=c,source_sha256=hashes,input_sha256=inputs,
         effective_contraction_limits=amendment['planning_limits'],numerical_amendment_sha256=digest(amendment_path),
         screen_plan_sha256=digest(OUT/'screen_plan.json'),prior_hardware_outcomes_known=True,
         new_hardware_outcomes_seen=False,
         confirmation_inference='162 separately confirmed selected-native/reference event tests; screening counts never enter confirmation intervals or p-values. When selected-native and old-L2 are the same saved circuit, retain independently measured arms as a self-control and set the method-comparison p-value to 1.'))
    print('Screen plan:',len(rows),'circuits;',shots,'shots',flush=True)


def submission_state(phase,block,family):
    jobs=read(OUT/'jobs.json') if (OUT/'jobs.json').exists() else []
    exists=any(j['phase']==phase and j['block']==block and j['family']==family for j in jobs)
    if not exists and (OUT/f'intent_{phase}_{family}_{block}.json').exists():
        raise RuntimeError('Ambiguous submission: reconcile receipt and remote tags before retrying')
    return jobs,exists


def submit(phase,block,family):
    import numpy as np
    from .nighthawk_study import connect,qpy_read
    from qiskit_ibm_runtime import SamplerV2
    assert phase in ['screen','confirm'] and block in [0,1] and family in ['FLP','TSP','VRP']
    jobs,exists=submission_state(phase,block,family)
    if exists: print('Already submitted',phase,block,family,flush=True);return
    c=settings(); protocol=read(OUT/'hardware_protocol.json')
    assert all(digest(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    assert all(digest(ROOT/p)==h for p,h in protocol['input_sha256'].items())
    if phase=='screen': assert digest(OUT/'screen_plan.json')==protocol['screen_plan_sha256']
    else:
        confirmation=read(OUT/'confirmation_protocol.json')
        assert digest(OUT/'confirm_plan.json')==confirmation['plan_sha256']
        assert digest(OUT/'screen_selection.json')==confirmation['selection_sha256']
    plan=read(OUT/(phase+'_plan.json'));rows=[r for r in plan['rows'] if r['family']==family]
    order=np.random.default_rng(c['ordering_seed']+(0 if phase=='screen' else 100)+block*3+['FLP','TSP','VRP'].index(family)).permutation(len(rows))
    rows=[rows[i] for i in order];shots=len(rows)*plan['shots_per_circuit']
    assert len(jobs)<c['hardware_max_jobs'] and sum(j['total_shots'] for j in jobs)+shots<=c['hardware_max_total_shots']
    _,backend=connect(c)
    assert backend.status().operational and backend.configuration().processor_type['family']=='Nighthawk'
    pubs=[]
    for row in rows:
        assert digest(ROOT/row['isa_qpy'])==row['isa_sha256']; qc=qpy_read(ROOT/row['isa_qpy']);r=row['resources']
        check_isa(qc,backend.target,r['physical'],r['final_logical_mapping'])
        pubs.append((qc,None,plan['shots_per_circuit']))
    save(OUT/f'calibration_{phase}_{family}_{block}.json',json.loads(json.dumps(backend.properties(refresh=True).to_dict(),default=str)))
    sampler=SamplerV2(mode=backend);sampler.options.max_execution_time=c['hardware_max_execution_seconds_per_job']
    sampler.options.dynamical_decoupling.enable=False
    sampler.options.twirling.enable_gates=False;sampler.options.twirling.enable_measure=False
    sampler.options.environment.job_tags=['native_depth_refinement',phase,f'{family}_b{block}',digest(OUT/(phase+'_plan.json'))[:12]]
    record=dict(phase=phase,block=block,family=family,rows=rows,total_shots=shots,shots_per_circuit=plan['shots_per_circuit'],
        backend=backend.name,backend_version=backend.backend_version,processor=backend.configuration().processor_type,
        plan_sha256=digest(OUT/(phase+'_plan.json')),options=json.loads(json.dumps(sampler.options.__dict__,default=str)))
    save(OUT/f'intent_{phase}_{family}_{block}.json',dict(prepared_utc=now(),**record))
    job=sampler.run(pubs)
    save(OUT/f'receipt_{phase}_{family}_{block}.json',dict(received_utc=now(),job_id=job.job_id()))
    jobs.append(dict(submitted_utc=now(),job_id=job.job_id(),**record));save(OUT/'jobs.json',jobs)
    print('Submitted',phase,family,block,job.job_id(),shots,'shots',flush=True)


def collect(phase):
    from .nighthawk_study import connect
    from .width_sweep_models import metrics
    jobs=[j for j in read(OUT/'jobs.json') if j['phase']==phase]
    pending=[j for j in jobs if not (OUT/'raw_counts'/f"{phase}_{j['family']}_b{j['block']}.json").exists()]
    if not pending: return len(jobs)==6
    service,_=connect(settings())
    for record in pending:
        job=service.job(record['job_id']);status=str(job.status())
        print(phase,record['family'],record['block'],status,flush=True)
        if status in ['ERROR','CANCELLED','JobStatus.ERROR','JobStatus.CANCELLED']:
            save(OUT/f"failure_{phase}_{record['family']}_{record['block']}.json",dict(job_id=record['job_id'],status=status,error=job.error_message()))
            raise RuntimeError('Hardware job failed; retained failure requires review before any new acquisition')
        if status not in ['DONE','JobStatus.DONE']: continue
        result=job.result();assert len(result)==len(record['rows']);rows=[]
        for original,pub in zip(record['rows'],result):
            counts=pub.data.meas.get_counts();assert sum(counts.values())==record['shots_per_circuit']
            model=read(PARENT/'cases'/original['case_id']/'model.json')
            rows.append(dict(case_id=original['case_id'],method=original['method'],candidate_id=original['candidate_id'],
                counts=counts,metrics=metrics(model,counts),metadata=json.loads(json.dumps(pub.metadata,default=str))))
        save(OUT/'raw_counts'/f"{phase}_{record['family']}_b{record['block']}.json",
            dict(collected_utc=now(),phase=phase,family=record['family'],block=record['block'],job_id=record['job_id'],rows=rows,
                 job_metrics=json.loads(json.dumps(job.metrics(),default=str))))
    return len(jobs)==6 and all((OUT/'raw_counts'/f"{phase}_{j['family']}_b{j['block']}.json").exists() for j in jobs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('args',nargs='*');a=p.parse_args()
    if a.action=='prepare': prepare_screen()
    elif a.action=='submit': submit(a.args[0],int(a.args[1]),a.args[2])
    elif a.action=='submit_all':
        for b in [0,1]:
            for f in (['FLP','TSP','VRP'] if b==0 else ['VRP','TSP','FLP']): submit(a.args[0],b,f)
    elif a.action=='collect': print('All complete:',collect(a.args[0]),flush=True)
    else: raise ValueError(a.action)
