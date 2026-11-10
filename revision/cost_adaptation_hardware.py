"""Fixed, small Nighthawk comparison of reused, warm-fit and cold-fit circuits."""
import argparse
import json
import pickle

from .cost_adaptation_study import ROOT, OUT, CONFIG, settings, check_frozen
from .contribution_hardware_core import read, save, digest, now, connection_config, check_isa

EVENTS=('feasible','good','optimal')
REFERENCES=('cold','reuse')


def verify(record):
    check_frozen()
    for name,value in {**record.get('source_sha256',{}),**record.get('input_sha256',{})}.items():
        assert digest(ROOT/name)==value,name


def prepare():
    from .nighthawk_study import qpy_read
    check_frozen();c=settings()
    assert (OUT/'preparation_complete.json').exists()
    assert not (OUT/'hardware_protocol.json').exists()
    target=pickle.loads((OUT/'preflight/target.pkl').read_bytes())
    rows=[];inputs={}
    for spec in c['cases']:
        path=OUT/'cases'/spec['id']/'complete.json';case=read(path)
        inputs[str(path.relative_to(ROOT))]=digest(path)
        assert len(case['fits'])==6 and all(f['verified'] and f['evaluations']<=40 for f in case['fits'])
        assert all(f['adequate_evaluation_budget'] for f in case['fits'])
        assert {r['method'] for r in case['rows']}==set(c['methods'])
        for row in case['rows']:
            for kind in ['logical','isa']:
                path=ROOT/row[kind+'_qpy'];assert digest(path)==row[kind+'_sha256']
                inputs[str(path.relative_to(ROOT))]=digest(path)
            r=row['resources'];checked=check_isa(qpy_read(ROOT/row['isa_qpy']),target,r['physical'],r['final_logical_mapping'])
            assert checked['cz_depth']==r['cz_depth'] and r['swap_count']==0
        rows.extend(case['rows'])
    plan=dict(created_utc=now(),rows=rows,blocks=c['hardware_blocks'],shots_per_circuit=c['hardware_shots_per_circuit'])
    plan['total_shots']=len(rows)*plan['blocks']*plan['shots_per_circuit']
    assert len(rows)==18 and plan['total_shots']==c['hardware_max_total_shots']
    save(OUT/'hardware_plan.json',plan)
    names=['revision/cost_adaptation_hardware.py','revision/audit_tensor_prefix_statistics.py',
           'revision/hardware_width_figure.py','revision/width_sweep_models.py']
    hashes={n:digest(ROOT/n) for n in names}
    for name in names:
        (OUT/'source_snapshot'/name.rsplit('/',1)[1]).write_bytes((ROOT/name).read_bytes())
    save(OUT/'hardware_protocol.json',dict(frozen_utc=now(),plan_sha256=digest(OUT/'hardware_plan.json'),
        source_sha256=hashes,input_sha256=inputs,new_hardware_outcomes_seen=False,
        primary_inference=c['primary_inference'],hardware_max_total_shots=c['hardware_max_total_shots']))
    print('Frozen',len(rows),'circuits;',plan['total_shots'],'shots;',36,'contrasts.',flush=True)


def submit(block):
    import numpy as np
    from .nighthawk_study import connect,qpy_read
    from qiskit_ibm_runtime import SamplerV2
    c=settings();assert block in range(c['hardware_blocks'])
    record=read(OUT/'hardware_protocol.json');verify(record)
    assert digest(OUT/'hardware_plan.json')==record['plan_sha256']
    jobs=read(OUT/'jobs.json') if (OUT/'jobs.json').exists() else []
    if any(j['block']==block for j in jobs):return
    assert not (OUT/f'intent_b{block}.json').exists(),'Reconcile uncertain prior submission before retrying.'
    plan=read(OUT/'hardware_plan.json')
    _,backend=connect(dict(connection_config(),backend=c['backend']))
    processor=backend.configuration().processor_type
    assert processor=={'family':c['processor_family_required'],'revision':c['processor_revision_required']}
    assert backend.status().operational
    properties=backend.properties(refresh=True).to_dict();save(OUT/f'calibration_b{block}.json',properties)
    order=np.random.default_rng(c['ordering_seed']+block).permutation(len(plan['rows']))
    rows=[plan['rows'][i] for i in order]
    total=len(rows)*plan['shots_per_circuit']
    assert len(jobs)<c['hardware_max_jobs'] and sum(j['total_shots'] for j in jobs)+total<=c['hardware_max_total_shots']
    pubs=[]
    for row in rows:
        qc=qpy_read(ROOT/row['isa_qpy']);r=row['resources']
        check_isa(qc,backend.target,r['physical'],r['final_logical_mapping'])
        pubs.append((qc,None,plan['shots_per_circuit']))
    sampler=SamplerV2(mode=backend)
    sampler.options.max_execution_time=c['hardware_max_execution_seconds_per_job']
    sampler.options.dynamical_decoupling.enable=False
    sampler.options.twirling.enable_gates=False;sampler.options.twirling.enable_measure=False
    sampler.options.environment.job_tags=['aqc_cost_adaptation_2026_10',f'block_{block}',record['plan_sha256'][:12]]
    metadata=dict(block=block,rows=rows,total_shots=total,shots_per_circuit=plan['shots_per_circuit'],
        backend=backend.name,processor=processor,calibration_timestamp=str(properties['last_update_date']),
        plan_sha256=record['plan_sha256'],options=json.loads(json.dumps(sampler.options.__dict__,default=str)))
    save(OUT/f'intent_b{block}.json',dict(prepared_utc=now(),**metadata))
    job=sampler.run(pubs)
    save(OUT/f'receipt_b{block}.json',dict(received_utc=now(),job_id=job.job_id()))
    jobs.append(dict(submitted_utc=now(),job_id=job.job_id(),**metadata));save(OUT/'jobs.json',jobs)
    print('Submitted block',block,job.job_id(),total,'shots',flush=True)


def collect():
    from .nighthawk_study import connect
    from .width_sweep_models import metrics
    verify(read(OUT/'hardware_protocol.json'))
    jobs=read(OUT/'jobs.json')
    missing=[j for j in jobs if not (OUT/'raw_counts'/f'block_{j["block"]}.json').exists()]
    if missing:
        service,_=connect(dict(connection_config(),backend=settings()['backend']))
        for record in missing:
            job=service.job(record['job_id']);status=str(job.status());print(record['block'],status,flush=True)
            if status in ['ERROR','CANCELLED','JobStatus.ERROR','JobStatus.CANCELLED']:
                save(OUT/f'failure_b{record["block"]}.json',dict(status=status,error=job.error_message()))
                raise RuntimeError('Job failed; no automatic resubmission.')
            if status not in ['DONE','JobStatus.DONE']:continue
            result=job.result();assert len(result)==len(record['rows']);rows=[]
            for original,pub in zip(record['rows'],result):
                counts=pub.data.meas.get_counts();assert sum(counts.values())==record['shots_per_circuit']
                model=read(OUT/'cases'/original['case_id']/'model.json')
                rows.append(dict(case_id=original['case_id'],method=original['method'],candidate_id=original['candidate_id'],
                    counts=counts,metrics=metrics(model,counts),metadata=json.loads(json.dumps(pub.metadata,default=str))))
            save(OUT/'raw_counts'/f'block_{record["block"]}.json',dict(collected_utc=now(),
                block=record['block'],job_id=record['job_id'],rows=rows,
                job_metrics=json.loads(json.dumps(job.metrics(),default=str))))
    complete=len(jobs)==2 and all((OUT/'raw_counts'/f'block_{b}.json').exists() for b in [0,1])
    if complete:save(OUT/'hardware_complete.json',dict(completed_utc=now(),jobs=2,total_shots=sum(j['total_shots'] for j in jobs)))
    print('Hardware complete:',complete,flush=True)


def analyze():
    from .width_sweep_models import metrics
    from .audit_tensor_prefix_statistics import conditional_two_sided,holm_adjust
    from .hardware_width_figure import block_mover_interval
    from .native_depth_analysis import write_csv
    verify(read(OUT/'hardware_protocol.json'))
    assert (OUT/'hardware_complete.json').exists()
    plan=read(OUT/'hardware_plan.json');jobs=read(OUT/'jobs.json');index={};hashes={}
    expected={(r['case_id'],r['method']):r for r in plan['rows']}
    for job in jobs:
        path=OUT/'raw_counts'/f'block_{job["block"]}.json';raw=read(path)
        hashes[str(path.relative_to(ROOT))]=digest(path)
        assert raw['job_id']==job['job_id']
        for row in raw['rows']:
            cid,method=row['case_id'],row['method'];key=cid,method,raw['block']
            assert key not in index and row['candidate_id']==expected[cid,method]['candidate_id']
            checked=metrics(read(OUT/'cases'/cid/'model.json'),row['counts'])
            assert checked==row['metrics'] and checked['shots']==plan['shots_per_circuit']
            index[key]=checked
    assert len(index)==36
    pooled=[];effects=[]
    for row in plan['rows']:
        cid,method=row['case_id'],row['method'];parts=[index[cid,method,b] for b in [0,1]]
        result={k:row[k] for k in ['case_id','family','n_qubits','method','exact_fidelity']}
        result.update(shots=sum(p['shots'] for p in parts),cz_depth=row['resources']['cz_depth'])
        for event in EVENTS:
            count=sum(p[event+'_count'] for p in parts)
            result.update({event+'_count':count,'p_'+event:count/result['shots'],
                'ideal_p_'+event:row['ideal_events']['p_'+event]})
        pooled.append(result)
    for spec in settings()['cases']:
        cid=spec['id'];warm=expected[cid,'warm']
        for reference in REFERENCES:
            ref=expected[cid,reference]
            same=warm['parameters']==ref['parameters']
            for event in EVENTS:
                tables=[(index[cid,'warm',b][event+'_count'],index[cid,'warm',b]['shots'],
                         index[cid,reference,b][event+'_count'],index[cid,reference,b]['shots']) for b in [0,1]]
                p,lo,hi=block_mover_interval(tables)
                effects.append(dict(case_id=cid,family=warm['family'],reference=reference,event=event,
                    warm_count=sum(t[0] for t in tables),reference_count=sum(t[2] for t in tables),
                    shots_per_method=sum(t[1] for t in tables),improvement_pp=100*p,ci_low_pp=100*lo,ci_high_pp=100*hi,
                    identical_parameters=same,p_value=1. if same else conditional_two_sided(tables)))
    assert len(effects)==36
    adjusted=holm_adjust([r['p_value'] for r in effects])
    effects=[dict(r,p_holm=float(p),gain=p<.05 and r['improvement_pp']>0,loss=p<.05 and r['improvement_pp']<0) for r,p in zip(effects,adjusted)]
    summary=[dict(reference=ref,event=event,gains=sum(r['gain'] for r in effects if r['reference']==ref and r['event']==event),
        losses=sum(r['loss'] for r in effects if r['reference']==ref and r['event']==event)) for ref in REFERENCES for event in EVENTS]
    write_csv(OUT/'hardware_outcomes.csv',pooled);write_csv(OUT/'hardware_comparisons.csv',effects)
    save(OUT/'analysis.json',dict(analyzed_utc=now(),summary=summary,comparisons=effects,raw_sha256=hashes,
        quantum_seconds=sum(read(OUT/'raw_counts'/f'block_{b}.json')['job_metrics']['usage']['quantum_seconds'] for b in [0,1])))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','submit','collect','analyze']);parser.add_argument('block',type=int,nargs='?')
    args=parser.parse_args()
    if args.action=='submit':submit(args.block)
    else:globals()[args.action]()
