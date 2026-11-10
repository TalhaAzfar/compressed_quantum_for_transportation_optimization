"""Independent hardware confirmation of six longer-duration annealing targets."""
import argparse
import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from .long_anneal_study import ROOT, OUT, settings, check_frozen
from .contribution_hardware_core import read, save, digest, now, check_isa
from .native_depth_analysis import rank_native

REFERENCES = ('source','refined_product','fixed_shallow')
EVENTS = ('feasible','optimal')
FAMILIES = ('TSP','VRP')


def candidates(case):
    rows = case['rows']
    assert len(rows) == 12 and all(r['fit']['verified'] and r['fit']['adequate_evaluation_budget'] for r in rows)
    product = max((r for r in rows if r['layers']==0),key=lambda r:(r['exact_fidelity'],r['id']))
    shallow = max((r for r in rows if r['layers']==2),key=lambda r:(r['exact_fidelity'],r['id']))
    chosen = [(case['source'],'source'),(product,'refined_product'),(shallow,'fixed_shallow')]
    ids = {r['id'] for r,_ in chosen}
    for cap in [shallow['cz_depth_cap'],32,64]:
        pool = [r for r in rows if r['layers'] and r['resources']['cz_depth']<=cap]
        for criterion in ['fidelity','feasible']:
            best = max(pool,key=lambda r:(r['exact_fidelity'] if criterion=='fidelity' else r['ideal_events']['p_feasible'],
                                         r['exact_fidelity'],-r['resources']['cz_depth'],r['id']))
            if best['id'] not in ids:
                chosen.append((best,'candidate_'+str(len(chosen))));ids.add(best['id'])
    assert len(chosen)<=9
    fields=['candidate_id','case_id','family','n_qubits','layers','exact_fidelity','ideal_events',
            'logical_qpy','logical_sha256','isa_qpy','isa_sha256','resources']
    return [dict({k:r[k] for k in fields},method=method) for r,method in chosen]


def verify_files(record):
    for field in ['source_sha256','input_sha256']:
        assert all(digest(ROOT/p)==h for p,h in record.get(field,{}).items()), 'Frozen acquisition input changed'


def prepare():
    import pickle
    from .nighthawk_study import qpy_read
    check_frozen();c=settings()
    assert (OUT/'preparation_complete.json').exists()
    assert not (OUT/'hardware_protocol.json').exists()
    tests=(OUT/'hardware_tests.log').read_text()
    assert 'passed' in tests and 'FAILED' not in tests and 'ERROR' not in tests
    target=pickle.loads((OUT/'preflight/target.pkl').read_bytes());rows=[];inputs={}
    for spec in c['cases']:
        folder=OUT/'cases'/spec['id'];case=read(folder/'complete.json')
        selected=candidates(case)
        # Remove duplicate logical circuits among optional candidates. Required
        # controls remain separately measured even when selected as treatment.
        seen=[];dedup=[]
        for row in selected:
            qc=qpy_read(ROOT/row['logical_qpy'])
            if row['method'].startswith('candidate_') and any(qc==old for old in seen): continue
            seen.append(qc);dedup.append(row)
        for row in dedup:
            for field in ['logical','isa']:
                assert digest(ROOT/row[field+'_qpy'])==row[field+'_sha256']
                inputs[row[field+'_qpy']]=row[field+'_sha256']
            r=row['resources'];check_isa(qpy_read(ROOT/row['isa_qpy']),target,r['physical'],r['final_logical_mapping'])
        rows.extend(dedup)
        for p in folder.glob('*.json'): inputs[str(p.relative_to(ROOT))]=digest(p)
    plan=dict(created_utc=now(),rows=rows,blocks=c['screen_blocks'],shots_per_circuit=c['screen_shots'])
    plan['total_shots']=len(rows)*plan['blocks']*plan['shots_per_circuit']
    assert plan['total_shots']+6*4*c['confirmation_blocks']*c['confirmation_shots']<=c['hardware_max_total_shots']
    save(OUT/'screen_plan.json',plan)
    names=['revision/long_anneal_hardware.py','tests/test_long_anneal_hardware.py','tests/test_reproducibility_hardware.py',
           'revision/native_depth_analysis.py','revision/hardware_width_figure.py','revision/audit_tensor_prefix_statistics.py']
    protocol=read(OUT/'preparation_protocol.json')
    hashes={**protocol['source_sha256'],**{n:digest(ROOT/n) for n in names}}
    for name in names:(OUT/'source_snapshot'/Path(name).name).write_bytes((ROOT/name).read_bytes())
    inputs.update(protocol['input_sha256'])
    inputs[str((OUT/'preparation_protocol.json').relative_to(ROOT))]=digest(OUT/'preparation_protocol.json')
    inputs[str((OUT/'hardware_tests.log').relative_to(ROOT))]=digest(OUT/'hardware_tests.log')
    save(OUT/'hardware_protocol.json',dict(frozen_utc=now(),config=c,source_sha256=hashes,input_sha256=inputs,
        screen_plan_sha256=digest(OUT/'screen_plan.json'),new_hardware_outcomes_seen=False,
        prior_results_seen=True,primary_inference=c['primary_inference'],secondary_inference=c['secondary_inference']))
    print('Frozen screening:',len(rows),'circuits;',plan['total_shots'],'shots',flush=True)


def aggregate(phase,day=0):
    from .width_sweep_models import metrics
    plan=read(OUT/('screen_plan.json' if phase=='screen' else 'confirm_plan.json'))
    expected={(r['case_id'],r['method']):r for r in plan['rows']}
    jobs=[j for j in read(OUT/'jobs.json') if j['phase']==phase and j['day']==day]
    assert len(jobs)==4 and len({(j['family'],j['block']) for j in jobs})==4
    index={};inputs={}
    for job in jobs:
        assert job['backend']==settings()['backend'] and job['processor']['family']=='Nighthawk'
        path=OUT/'raw_counts'/f'{phase}_d{day}_{job["family"]}_b{job["block"]}.json'
        raw=read(path);inputs[str(path.relative_to(ROOT))]=digest(path)
        assert raw['job_id']==job['job_id']
        assert (raw['phase'],raw['day'],raw['block'],raw['family'])==(phase,day,job['block'],job['family'])
        assert job['plan_sha256']==digest(OUT/('screen_plan.json' if phase=='screen' else 'confirm_plan.json'))
        for row in raw['rows']:
            key=(row['case_id'],row['method'],job['block']);assert key not in index
            assert expected[key[:2]]['candidate_id']==row['candidate_id']
            model=read(OUT/'cases'/row['case_id']/'model.json');checked=metrics(model,row['counts'])
            assert checked==row['metrics'] and checked['shots']==plan['shots_per_circuit']
            index[key]=dict(case_id=row['case_id'],method=row['method'],**checked)
    assert set(index)=={(cid,method,b) for cid,method in expected for b in range(2)}
    return plan,index,inputs


def select():
    assert not (OUT/'confirmation_protocol.json').exists()
    assert not any(j['phase']=='confirm' for j in read(OUT/'jobs.json'))
    verify_files(read(OUT/'hardware_protocol.json'))
    plan,index,inputs=aggregate('screen');rows=[];rankings={}
    for spec in settings()['cases']:
        cid=spec['id'];cr=[r for r in plan['rows'] if r['case_id']==cid];pooled=[]
        for row in cr:
            rr=[index[cid,row['method'],b] for b in range(2)]
            shots=sum(r['shots'] for r in rr);count=sum(r['feasible_count'] for r in rr)
            pooled.append(dict(case_id=cid,method=row['method'],shots=shots,feasible_count=count,p_feasible=count/shots))
        ranked=rank_native(cr,pooled);rankings[cid]=ranked
        chosen=next(r for r in cr if r['candidate_id']==ranked[0]['candidate_id'])
        rows.append(dict(chosen,method='selected_native',screen_method=chosen['method']))
        rows += [r for r in cr if r['method'] in REFERENCES]
    assert len(rows)==24
    save(OUT/'screen_selection.json',dict(frozen_utc=now(),rankings=rankings,screen_raw_sha256=inputs,
        scope='One feasibility-based selection. No confirmation outcomes used.'))
    save(OUT/'confirm_plan.json',dict(created_utc=now(),rows=rows,blocks=2,days=1,shots_per_circuit=settings()['confirmation_shots'],
                                    total_shots=24*2*settings()['confirmation_shots']))
    save(OUT/'confirmation_protocol.json',dict(frozen_utc=now(),plan_sha256=digest(OUT/'confirm_plan.json'),
        selection_sha256=digest(OUT/'screen_selection.json'),screen_raw_sha256=inputs,confirmation_outcomes_seen=False,
        primary_inference=settings()['primary_inference'],secondary_inference=settings()['secondary_inference']))
    print('Frozen independent confirmation: 24 circuits, 65536 shots each.',flush=True)


def submission_state(phase,day,block,family):
    jobs=read(OUT/'jobs.json') if (OUT/'jobs.json').exists() else []
    exists=any((j['phase'],j['day'],j['block'],j['family'])==(phase,day,block,family) for j in jobs)
    stem=f'{phase}_d{day}_{family}_b{block}'
    if not exists and (OUT/('intent_'+stem+'.json')).exists():
        raise RuntimeError('Ambiguous submission: reconcile saved receipt and remote tags before retrying')
    return jobs,exists,stem


def submit(phase,day,block,family):
    import numpy as np
    from .nighthawk_study import connect,qpy_read
    from qiskit_ibm_runtime import SamplerV2
    assert phase in ['screen','confirm'] and day==0 and block in [0,1] and family in FAMILIES
    assert phase!='screen' or day==0
    jobs,exists,stem=submission_state(phase,day,block,family)
    if exists:return True
    c=settings();protocol=read(OUT/'hardware_protocol.json');verify_files(protocol)
    plan_path=OUT/('screen_plan.json' if phase=='screen' else 'confirm_plan.json');plan=read(plan_path)
    if phase=='screen':assert digest(plan_path)==protocol['screen_plan_sha256']
    else:
        confirm=read(OUT/'confirmation_protocol.json')
        assert digest(plan_path)==confirm['plan_sha256']
        assert digest(OUT/'screen_selection.json')==confirm['selection_sha256']
        assert all(digest(ROOT/p)==h for p,h in confirm['screen_raw_sha256'].items())
    _,backend=connect(c);assert backend.status().operational
    props=backend.properties(refresh=True).to_dict();stamp=str(props['last_update_date'])
    rows=[r for r in plan['rows'] if r['family']==family]
    order=np.random.default_rng(c['ordering_seed']+(0 if phase=='screen' else 100)+20*day+2*block+FAMILIES.index(family)).permutation(len(rows))
    rows=[rows[i] for i in order];shots=len(rows)*plan['shots_per_circuit']
    assert len(jobs)<c['hardware_max_jobs'] and sum(j['total_shots'] for j in jobs)+shots<=c['hardware_max_total_shots']
    pubs=[]
    for row in rows:
        assert digest(ROOT/row['isa_qpy'])==row['isa_sha256']
        qc=qpy_read(ROOT/row['isa_qpy']);r=row['resources']
        check_isa(qc,backend.target,r['physical'],r['final_logical_mapping'])
        pubs.append((qc,None,plan['shots_per_circuit']))
    save(OUT/('calibration_'+stem+'.json'),props)
    sampler=SamplerV2(mode=backend);sampler.options.max_execution_time=c['hardware_max_execution_seconds_per_job']
    sampler.options.dynamical_decoupling.enable=False
    sampler.options.twirling.enable_gates=False;sampler.options.twirling.enable_measure=False
    sampler.options.environment.job_tags=['aqc_long_anneal_2026_10',stem,digest(plan_path)[:12]]
    record=dict(phase=phase,day=day,block=block,family=family,rows=rows,total_shots=shots,
        shots_per_circuit=plan['shots_per_circuit'],backend=backend.name,backend_version=backend.backend_version,
        processor=backend.configuration().processor_type,calibration_timestamp=stamp,plan_sha256=digest(plan_path),
        options=json.loads(json.dumps(sampler.options.__dict__,default=str)))
    save(OUT/('intent_'+stem+'.json'),dict(prepared_utc=now(),**record))
    job=sampler.run(pubs)
    save(OUT/('receipt_'+stem+'.json'),dict(received_utc=now(),job_id=job.job_id()))
    jobs.append(dict(submitted_utc=now(),job_id=job.job_id(),**record));save(OUT/'jobs.json',jobs)
    print('Submitted',stem,job.job_id(),shots,'shots',flush=True)
    return True


def collect(phase,day):
    from .nighthawk_study import connect
    from .width_sweep_models import metrics
    jobs=[j for j in read(OUT/'jobs.json') if j['phase']==phase and j['day']==day]
    def path(j):return OUT/'raw_counts'/f'{phase}_d{day}_{j["family"]}_b{j["block"]}.json'
    pending=[j for j in jobs if not path(j).exists()]
    if pending:
        service,_=connect(settings())
        for record in pending:
            job=service.job(record['job_id']);status=str(job.status())
            print(phase,day,record['family'],record['block'],status,flush=True)
            if status in ['ERROR','CANCELLED','JobStatus.ERROR','JobStatus.CANCELLED']:
                save(OUT/('failure_'+record['job_id']+'.json'),dict(status=status,error=job.error_message()))
                raise RuntimeError('Hardware failure retained; no automatic resubmission')
            if status not in ['DONE','JobStatus.DONE']:continue
            result=job.result();assert len(result)==len(record['rows']);rows=[]
            for original,pub in zip(record['rows'],result):
                counts=pub.data.meas.get_counts();assert sum(counts.values())==record['shots_per_circuit']
                model=read(OUT/'cases'/original['case_id']/'model.json')
                rows.append(dict(case_id=original['case_id'],method=original['method'],candidate_id=original['candidate_id'],
                    counts=counts,metrics=metrics(model,counts),metadata=json.loads(json.dumps(pub.metadata,default=str))))
            save(path(record),dict(collected_utc=now(),phase=phase,day=day,block=record['block'],family=record['family'],
                job_id=record['job_id'],rows=rows,job_metrics=json.loads(json.dumps(job.metrics(),default=str))))
    complete=len(jobs)==4 and all(path(j).exists() for j in jobs)
    if complete and phase=='confirm' and not (OUT/f'day{day}_complete.json').exists():
        aggregate(phase,day)
        save(OUT/f'day{day}_complete.json',dict(completed_utc=now(),
            last_completed_utc=max(read(path(j))['collected_utc'] for j in jobs),
            calibration_timestamps=sorted({j['calibration_timestamp'] for j in jobs}),
            interpretation='Independent two-block confirmation at the prespecified shot budget.'))
    return complete


def contrast(rows,indices,reference,event):
    from .audit_tensor_prefix_statistics import conditional_two_sided
    from .hardware_width_figure import block_mover_interval,block_difference_interval
    a=next(r for r in rows if r['method']=='selected_native');b=next(r for r in rows if r['method']==reference)
    cid=a['case_id'];tables=[]
    for index in indices:
        for block in range(2):
            x=index[cid,'selected_native',block];y=index[cid,reference,block]
            tables.append((x[event+'_count'],x['shots'],y[event+'_count'],y['shots']))
    p,lo,hi=block_mover_interval(tables);_,clo,chi=block_difference_interval(tables)
    ka,na,kb,nb=map(sum,zip(*tables));same=a['isa_sha256']==b['isa_sha256']
    return dict(case_id=cid,family=a['family'],n_qubits=a['n_qubits'],event=event,reference=reference,
        selected_count=ka,selected_shots=na,reference_count=kb,reference_shots=nb,
        improvement_pp=100*p,ci_low_pp=100*lo,ci_high_pp=100*hi,
        conservative_ci_low_pp=100*clo,conservative_ci_high_pp=100*chi,
        p_value=1. if same else conditional_two_sided(tables),identical_circuit_self_control=same,
        selected_cz_depth=a['resources']['cz_depth'],reference_cz_depth=b['resources']['cz_depth'])


def corrected(rows):
    from .audit_tensor_prefix_statistics import holm_adjust
    adjusted=holm_adjust([r['p_value'] for r in rows])
    return [dict(r,p_holm=float(p),gain=p<.05 and r['improvement_pp']>0,
                 loss=p<.05 and r['improvement_pp']<0) for r,p in zip(rows,adjusted)]


def analyze():
    import csv
    assert (OUT/'day0_complete.json').exists(), 'Confirmation is incomplete'
    check_frozen();verify_files(read(OUT/'hardware_protocol.json'))
    plan,index,inputs=aggregate('confirm',0);effects=[]
    assert len(plan['rows'])==24 and plan['total_shots']==1572864
    for spec in settings()['cases']:
        rows=[r for r in plan['rows'] if r['case_id']==spec['id']]
        for event in EVENTS:
            for ref in REFERENCES:effects.append(contrast(rows,[index],ref,event))
    effects=corrected(effects);assert len(effects)==36
    summary=[]
    for ref in REFERENCES:
        for event in EVENTS:
            rr=[r for r in effects if r['reference']==ref and r['event']==event]
            summary.append(dict(reference=ref,event=event,cases=len(rr),
                corrected_gains=sum(r['gain'] for r in rr),corrected_losses=sum(r['loss'] for r in rr)))
    pooled=[]
    for row in plan['rows']:
        blocks=[index[row['case_id'],row['method'],b] for b in range(2)]
        shots=sum(r['shots'] for r in blocks)
        record={k:row[k] for k in ['case_id','method','candidate_id','family','n_qubits','exact_fidelity']}
        record.update(shots=shots,cz_depth=row['resources']['cz_depth'],cz_count=row['resources']['cz_count'])
        for event in EVENTS:
            count=sum(r[event+'_count'] for r in blocks)
            record[event+'_count']=count;record['p_'+event]=count/shots
            record['ideal_p_'+event]=row['ideal_events']['p_'+event]
        pooled.append(record)
    save(OUT/'analysis.json',dict(completed_utc=now(),comparisons=effects,summary=summary,
        confirmation_raw_sha256=inputs,counts_independently_decoded=True,
        source_duration=8,source_steps=24,confirmation_shots_per_method=65536,
        primary_inference=settings()['primary_inference'],
        interpretation='Six previously diagnosed instances with independent hardware confirmation after screening. All 36 contrasts corrected together; no two-day replication or extrapolation to other instances.'))
    for name,rows in [('confirmation_comparisons.csv',effects),('confirm_pooled.csv',pooled)]:
        with (OUT/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Analyzed all 36 confirmation contrasts:',summary,flush=True)
    report()


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    data=read(OUT/'analysis.json');colors={'TSP':'#D55E00','VRP':'#009E73'};markers={'TSP':'^','VRP':'s'}
    for ref,label in [('source','24-step annealing source'),('refined_product','fitted product state'),('fixed_shallow','shallow AQC')]:
        fig,axes=plt.subplots(1,2,figsize=(10,4.2))
        for ax,event in zip(axes,EVENTS):
            for r in data['comparisons']:
                if (r['reference'],r['event'])!=(ref,event):continue
                offset=(int(r['case_id'][-1])-2)*.16;p=r['improvement_pp'];family=r['family']
                ax.errorbar(r['n_qubits']+offset,p,yerr=[[p-r['ci_low_pp']],[r['ci_high_pp']-p]],
                    fmt=markers[family],color=colors[family],capsize=3,
                    mfc='white' if r['identical_circuit_self_control'] else colors[family])
            ax.axhline(0,color='gray',ls='--',lw=.8);ax.grid(axis='y',alpha=.2)
            ax.set(xlabel='Number of qubits',ylabel=('Feasibility' if event=='feasible' else 'Optimum')+' improvement (percentage points)',xticks=[20,25],xlim=(19,26))
            ax.spines[['top','right']].set_visible(False)
        fig.legend(handles=[Line2D([],[],color=colors[f],marker=markers[f],ls='',label=f) for f in FAMILIES],loc='upper right',ncol=2,frameon=False)
        fig.suptitle('Native AQC versus '+label,x=.09,ha='left')
        fig.text(.09,.02,'Nighthawk; 65,536 confirmation shots per method; pointwise 95% intervals; 36 tests jointly corrected.',fontsize=8)
        fig.tight_layout(rect=[0,.06,1,.91])
        for ext in ['pdf','png']:fig.savefig(OUT/'figures'/('confirmation_vs_'+ref+'.'+ext),bbox_inches='tight',dpi=200)
        plt.close(fig)
    save(OUT/'report_complete.json',dict(completed_utc=now(),analysis_sha256=digest(OUT/'analysis.json'),
        total_shots=sum(j['total_shots'] for j in read(OUT/'jobs.json')),summary=data['summary']))


def run():
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'analysis.json').exists():return
    while not (OUT/'preparation_complete.json').exists():
        assert not (OUT/'preparation_stop.json').exists(), 'Classical preparation stopped'
        if (OUT/'preparation_progress.json').exists():
            progress=read(OUT/'preparation_progress.json')
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(progress['updated_utc'])).total_seconds()
            assert age<180,'Preparation heartbeat stale'
        save(OUT/'pipeline_progress.json',dict(updated_utc=now(),phase='waiting_for_validated_preparation'))
        time.sleep(30)
    if not (OUT/'hardware_protocol.json').exists():prepare()
    for phase,day in [('screen',0),('confirm',0)]:
        if phase=='confirm' and not (OUT/'confirmation_protocol.json').exists():select()
        deadline=time.time()+settings()['maximum_calendar_wait_days']*86400
        for block in [0,1]:
            for family in (FAMILIES if block==0 else tuple(reversed(FAMILIES))):
                while not submit(phase,day,block,family):
                    assert time.time()<deadline,'Hardware submission wait exceeded'
                    save(OUT/'pipeline_progress.json',dict(updated_utc=now(),phase='waiting_for_submission',day=day))
                    time.sleep(60)
        deadline=time.time()+86400
        while not collect(phase,day):
            assert time.time()<deadline,'Collection wait exceeded one day; inspect before continuing'
            save(OUT/'pipeline_progress.json',dict(updated_utc=now(),phase='collecting',acquisition=phase,day=day))
            time.sleep(30)
    analyze()
    save(OUT/'pipeline_complete.json',dict(completed_utc=now(),independent_confirmation_complete=True,confirmation_shots_per_method=65536))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','select','analyze','run']);a=p.parse_args()
    try:globals()[a.action]()
    except BaseException as error:
        if a.action=='run':save(OUT/'pipeline_stop.json',dict(stopped_utc=now(),type=type(error).__name__,reason=str(error)))
        raise
