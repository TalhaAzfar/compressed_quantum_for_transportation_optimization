"""Bounded, contemporaneous depth-ladder sampling on IBM Rensselaer."""
import argparse
import json
import pickle
from pathlib import Path
from .depth200_optimum_screen import ROOT,OUT,save,case_data
from .flp42_fidelity_parallel import digest,now


def config():
    base=json.loads((ROOT/'configs/aqc_tensor_qaoa_v2.json').read_text())
    return {k:base[k] for k in ['account','instance_name','backend','processor_family_required']}


def select_candidates(rows):
    rows=[r for r in rows if r.get('verified')]
    groups={'product':[r for r in rows if r['layers']==0]}
    groups.update({f'compressed_l{layer}':[r for r in rows if r['layers']==layer and r['resources']['cz_depth']<=200]
                   for layer in [2,6,12,24]})
    if not all(groups.values()):raise ValueError('Each comparison requires a verified product and every native depth')
    return {key:max(group,key=lambda r:(r['fidelity'],-r['resources']['cz_depth'],-r['start']))
            for key,group in groups.items()}


def prepare():
    import importlib.metadata
    from .nighthawk_study import qpy_read,qpy_write
    from .tensor_prefix_sabre import compile_isa_sabre
    from .readout_indicator_mitigation import bit_mapping
    from .validate_tensor_prefix import check_circuit
    from .depth200_native_cleanup import cleanup,cleaned_resources
    assert (OUT/'fitting_complete.json').exists(), 'Complete all declared attempts before final selection'
    testlog=(OUT/'tests.log').read_text()
    assert 'passed' in testlog and 'FAILED' not in testlog and 'ERROR' not in testlog
    target=pickle.loads((OUT/'preflight/target_snapshot.pkl').read_bytes());rows=[];primary_methods={}
    for cid in ['tsp25','flp30']:
        case=case_data(cid);model=case['model'];patch=case['patch']
        selected=select_candidates([json.loads(p.read_text()) for p in (OUT/'fits').glob(f'{cid}_*.json')])
        primary_methods[cid]=max(['compressed_l12','compressed_l24'],key=lambda name:(selected[name]['fidelity'],-selected[name]['resources']['cz_depth']))
        original=case['methods']['standard_p2'];source=ROOT/original['logical_qpy'];qc=qpy_read(source)
        isa,resources=compile_isa_sabre(qc,target,patch['physical'],patch['edges'],seed=81003)
        path=OUT/'circuits'/f'{cid}_original_isa.qpy';qpy_write(path,isa)
        reference=json.loads((OUT/'references'/f'{cid}_selection.json').read_text());assert reference['status']=='verified'
        ideal=(json.loads((OUT/'screen'/f'{cid}_standard_p2_exact.json').read_text())['events']
               if cid=='tsp25' else next(r['events'] for r in json.loads((OUT/'exact_flp_source_events.json').read_text())['rows']
                                         if r['case_id']==cid and r['method']=='standard_p2'))
        originalrow=dict(method='original',case_id=cid,logical_qpy=str(source.relative_to(ROOT)),logical_sha256=digest(source),
             isa_qpy=str(path.relative_to(ROOT)),isa_sha256=digest(path),fidelity=1.,events=ideal,
             resources={k:v for k,v in resources.items() if k!='routing_trace'},shots=16384 if cid=='tsp25' else 6144)
        rows.append(originalrow)
        for method,r in selected.items():
            bit_mapping(qpy_read(ROOT/r['isa_qpy']),patch['physical'])
            rows.append(dict(method=method,case_id=cid,selected_fit=r['name'],
                 **{k:r[k] for k in ['logical_qpy','logical_sha256','isa_qpy','isa_sha256','fidelity','events','resources']},
                 shots=(12288 if method=='product' else 16384) if cid=='tsp25' else 6144))
        bit_mapping(isa,patch['physical'])
    cleanup_records=[]
    for row in rows:
        before=qpy_read(ROOT/row['isa_qpy']);after,certificate=cleanup(before)
        original_path=row['isa_qpy'];original_hash=row['isa_sha256']
        path=OUT/'circuits'/f"{row['case_id']}_{row['method']}_clean_isa.qpy"
        qpy_write(path,after)
        row['resources']=cleaned_resources(after,row['resources'],target,certificate)
        row.update(before_cleanup_isa_qpy=original_path,before_cleanup_isa_sha256=original_hash,
                   isa_qpy=str(path.relative_to(ROOT)),isa_sha256=digest(path))
        cleanup_records.append(dict(case_id=row['case_id'],method=row['method'],before_sha256=original_hash,
                                    after_sha256=row['isa_sha256'],certificate=certificate))
    save(OUT/'native_cleanup_validation.json',dict(completed_utc=now(),records=cleanup_records,
         source_sha256=digest(ROOT/'revision/depth200_native_cleanup.py'),scope='Input-state-independent one-qubit block and exact commuting-CZ cancellation certificates; identical procedure for every source and compiled control.'))
    plan=dict(created_utc=now(),cases=['tsp25','flp30'],rows=rows,blocks=2,arms=['raw','suppressed'],primary_methods=primary_methods,
              max_total_shots=524288,planned_total_shots=sum(r['shots'] for r in rows)*4,
              max_jobs=4,max_seconds_per_job=180,
              selection='Highest independently evaluated source-state fidelity within EACH native depth; all four depths are executed. Primary method is fixed before acquisition as highest fidelity among depths96/192. Ties prefer fewer CZ layers, then lower start index. No final hardware outcome used for selection.',
              primary='Preselected deeper method versus original: raw optimum probability and raw feasible probability, for each case and noise-management arm; 8 two-sided Fisher tests in one Holm family.',
              secondary='All four depths versus original and product: optimum and feasible probabilities, two cases and two arms, 64 two-sided Fisher tests in a distinct Holm family. Within-5%-optimum success, best feasible gap and block consistency are descriptive. The post-acquisition best depth is exploratory, not a preselected confirmed winner.',
              depth_cap_scope='All compressed output circuits have complete CZ depth <=200; the uncompressed source is a reference that may exceed this cap.',
              sampling='Two independently randomized contemporaneous blocks per arm. TSP source and each native depth: 32768 shots per method per arm; product:24576. FLP:12288 per method per arm. No repair or feasibility postselection.',
              interpretation='Exploratory follow-up on previously studied instances. Exact optima label sampled outcomes only; neither initialization nor fitting uses optimal bitstrings. Absence of hits is not proof of zero probability.',
              noise_management='Raw disables DD and twirling. Suppressed enables XX dynamical decoupling and gate twirling, with measure twirling disabled. Both report actual sampled bitstrings, not mitigated probability estimates.',
              compiler_cleanup='Before acquisition, all methods receive independently certified one-qubit block optimization and exact CZ cancellation. Actual resource counts and before/after QPYs are retained; logical fitted states are unchanged.',
              config=config())
    assert plan['planned_total_shots']==plan['max_total_shots']
    allowed={tuple(sorted(edge)) for edge in target['cz']}
    for row in rows:check_circuit(row,case_data(row['case_id']),target,allowed)
    p=OUT/'hardware_plan.json'
    if p.exists():
        old=json.loads(p.read_text());assert all(old[k]==v for k,v in plan.items() if k!='created_utc')
    else:save(p,plan)
    sources=['revision/depth200_optimum_hardware.py','revision/depth200_optimum_screen.py',
             'revision/tensor_prefix_models.py','revision/tensor_prefix_sabre.py','revision/readout_indicator_mitigation.py',
             'revision/validate_tensor_prefix.py','revision/audit_tensor_prefix_statistics.py','revision/depth200_native_cleanup.py']
    save(OUT/'hardware_validation.json',dict(created_utc=now(),passed=True,plan_sha256=digest(p),
         source_sha256={s:digest(ROOT/s) for s in sources},tests_sha256=digest(OUT/'tests.log'),
         packages={name:importlib.metadata.version(name) for name in ['qiskit','qiskit-aer','qiskit-addon-aqc-tensor','qiskit-ibm-runtime','quimb','numpy','scipy']},
         circuits=len(rows),max_compressed_depth=max(r['resources']['cz_depth'] for r in rows if r['method']!='original')))
    print('Prepared',plan['planned_total_shots'],'shots;',[(r['case_id'],r['method'],r['resources']['cz_depth']) for r in rows],flush=True)


def submit(block,arm):
    import numpy as np
    from qiskit_ibm_runtime import SamplerV2
    from .nighthawk_study import connect,qpy_read
    from .full_circuit_hardware import existing_submission
    jobs,exists=existing_submission(OUT,block,arm)
    if exists:print('Already submitted',block,arm,flush=True);return
    path=OUT/'hardware_plan.json';plan=json.loads(path.read_text());validation=json.loads((OUT/'hardware_validation.json').read_text())
    assert validation['passed'] and validation['plan_sha256']==digest(path)
    assert all(digest(ROOT/p)==h for p,h in validation['source_sha256'].items())
    assert block in range(plan['blocks']) and arm in plan['arms']
    order=np.random.default_rng(110051+2*block+(arm=='suppressed')).permutation(len(plan['rows']))
    rows=[plan['rows'][i] for i in order];shots=sum(r['shots'] for r in rows)
    assert len(jobs)<plan['max_jobs'] and sum(j['total_shots'] for j in jobs)+shots<=plan['max_total_shots']
    _,backend=connect(plan['config']);assert backend.status().operational
    pubs=[]
    for row in rows:
        assert digest(ROOT/row['isa_qpy'])==row['isa_sha256']
        qc=qpy_read(ROOT/row['isa_qpy'])
        for inst in qc.data:
            if inst.operation.name=='barrier':continue
            qargs=tuple(qc.find_bit(q).index for q in inst.qubits)
            assert backend.target.instruction_supported(operation_name=inst.operation.name,qargs=qargs)
        pubs.append((qc,None,row['shots']))
    save(OUT/f'device_calibration_{block}_{arm}.json',json.loads(json.dumps(backend.properties(refresh=True).to_dict(),default=str)))
    sampler=SamplerV2(mode=backend);sampler.options.max_execution_time=plan['max_seconds_per_job']
    sampler.options.dynamical_decoupling.enable=arm=='suppressed'
    if arm=='suppressed':sampler.options.dynamical_decoupling.sequence_type='XX'
    sampler.options.twirling.enable_gates=arm=='suppressed';sampler.options.twirling.enable_measure=False
    if arm=='suppressed':
        sampler.options.twirling.num_randomizations=64;sampler.options.twirling.shots_per_randomization='auto'
    sampler.options.environment.job_tags=['depth200_optimum',f'b{block}_{arm}',digest(path)[:12]]
    record=dict(block=block,arm=arm,rows=rows,total_shots=shots,backend=backend.name,
                backend_version=backend.backend_version,processor=backend.configuration().processor_type,
                plan_sha256=digest(path),options=json.loads(json.dumps(sampler.options.__dict__,default=str)))
    save(OUT/f'intent_{block}_{arm}.json',dict(prepared_utc=now(),**record))
    job=sampler.run(pubs)
    save(OUT/f'receipt_{block}_{arm}.json',dict(job_id=job.job_id(),received_utc=now()))
    jobs.append(dict(job_id=job.job_id(),submitted_utc=now(),**record));save(OUT/'jobs.json',jobs)
    print('Submitted',block,arm,job.job_id(),shots,'shots',flush=True)


def collect():
    from .nighthawk_study import connect
    from .tensor_prefix_models import count_metrics
    jobs=json.loads((OUT/'jobs.json').read_text())
    pending=[j for j in jobs if not (OUT/'raw_counts'/f"b{j['block']}_{j['arm']}.json").exists()]
    if not pending:return
    service,_=connect(config())
    for item in pending:
        job=service.job(item['job_id']);status=str(job.status());print(item['job_id'],status,flush=True)
        if status in ['ERROR','CANCELLED','JobStatus.ERROR','JobStatus.CANCELLED']:
            save(OUT/f"failure_{item['block']}_{item['arm']}.json",dict(job_id=item['job_id'],status=status,error=job.error_message(),checked_utc=now()))
            continue
        if status not in ['DONE','JobStatus.DONE']:continue
        result=job.result();assert len(result)==len(item['rows']);rows=[]
        for entry,pub in zip(item['rows'],result):
            counts=pub.data.meas.get_counts();assert sum(counts.values())==entry['shots']
            model=case_data(entry['case_id'])['model']
            rows.append(dict(case_id=entry['case_id'],method=entry['method'],counts=counts,
                             metrics=count_metrics(model,counts),metadata=json.loads(json.dumps(pub.metadata,default=str))))
        save(OUT/'raw_counts'/f"b{item['block']}_{item['arm']}.json",dict(collected_utc=now(),job_id=item['job_id'],block=item['block'],
             arm=item['arm'],rows=rows,job_metrics=json.loads(json.dumps(job.metrics(),default=str))))
        print('Collected',item['job_id'],flush=True)


def analyze():
    import collections,csv
    import numpy as np
    from scipy.stats import beta,fisher_exact
    from .tensor_prefix_models import count_metrics,decode
    from .validate_tensor_prefix import independent_decode
    from .audit_tensor_prefix_statistics import conditional_two_sided
    plan=json.loads((OUT/'hardware_plan.json').read_text())
    raw=[json.loads(p.read_text()) for p in (OUT/'raw_counts').glob('b*.json')]
    assert len(raw)==4,'Wait for all fixed-budget blocks before final analysis'
    assert {(r['block'],r['arm']) for r in raw}=={(b,a) for b in range(plan['blocks']) for a in plan['arms']}
    pooled={};blocks=[]
    def intervals(r):
        n=r['shots']
        for e in ['feasible','good','optimal']:
            k=r[e+'_count'];r[e+'_ci_low']=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
            r[e+'_ci_high']=1. if k==n else float(beta.ppf(.975,k+1,n-k))
        return r
    for acquisition in raw:
        for row in acquisition['rows']:
            key=(row['case_id'],acquisition['arm'],row['method'])
            independent={event:0 for event in ['feasible','good','optimal']}
            model=case_data(key[0])['model']
            for bitstring,count in row['counts'].items():
                decoded=independent_decode(model,bitstring)
                for event in independent:independent[event]+=count*decoded[event]
            assert all(value==row['metrics'][event+'_count'] for event,value in independent.items())
            pooled.setdefault(key,collections.Counter()).update(row['counts'])
            blocks.append(dict(case_id=key[0],arm=key[1],method=key[2],block=acquisition['block'],**intervals(row['metrics'])))
    rows=[]
    for (cid,arm,method),counts in sorted(pooled.items()):
        model=case_data(cid)['model'];r=count_metrics(model,counts);intervals(r)
        decoded=decode(model,[int(k.replace(' ',''),2) for k in counts]);weights=np.array(list(counts.values()))
        feasible_weight=weights[decoded['feasible']]
        r['mean_feasible_cost']=(float(feasible_weight@decoded['cost'][decoded['feasible']]/feasible_weight.sum())
                                 if feasible_weight.sum() else None)
        selected=next(x for x in plan['rows'] if x['case_id']==cid and x['method']==method)
        r.update(case_id=cid,arm=arm,method=method,optimum=model['optimum'],
                 best_feasible_gap=None if r['best_feasible_cost'] is None else max(0.,r['best_feasible_cost']/model['optimum']-1),
                 cz_depth=selected['resources']['cz_depth'],cz_count=selected['resources']['cz_count'],
                 fidelity=selected['fidelity'],ideal_p_feasible=selected['events']['p_feasible'],ideal_p_optimal=selected['events']['p_optimal'])
        rows.append(r)
    tests=[]
    for cid in plan['cases']:
        for arm in plan['arms']:
            method=plan['primary_methods'][cid]
            a=next(r for r in rows if (r['case_id'],r['arm'],r['method'])==(cid,arm,method))
            b=next(r for r in rows if (r['case_id'],r['arm'],r['method'])==(cid,arm,'original'))
            for e in ['optimal','feasible']:
                ka,kb=a[e+'_count'],b[e+'_count'];_,p=fisher_exact([[ka,a['shots']-ka],[kb,b['shots']-kb]],alternative='two-sided')
                tables=[]
                for block in range(plan['blocks']):
                    left=next(r for r in blocks if (r['case_id'],r['arm'],r['method'],r['block'])==(cid,arm,method,block))
                    right=next(r for r in blocks if (r['case_id'],r['arm'],r['method'],r['block'])==(cid,arm,'original',block))
                    tables.append((left[e+'_count'],left['shots'],right[e+'_count'],right['shots']))
                tests.append(dict(case_id=cid,arm=arm,method=method,event=e,test_count=ka,control_count=kb,shots_each=a['shots'],
                                  difference=a['p_'+e]-b['p_'+e],p_value=float(p),stratified_p=conditional_two_sided(tables)))
    order=np.argsort([r['p_value'] for r in tests]);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(tests)-rank)*tests[index]['p_value']));tests[index]['p_holm']=running
    order=np.argsort([r['stratified_p'] for r in tests]);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(tests)-rank)*tests[index]['stratified_p']));tests[index]['stratified_p_holm']=running
    secondary=[]
    for cid in plan['cases']:
        for arm in plan['arms']:
            for method in ['compressed_l2','compressed_l6','compressed_l12','compressed_l24']:
                a=next(r for r in rows if (r['case_id'],r['arm'],r['method'])==(cid,arm,method))
                for control in ['original','product']:
                    b=next(r for r in rows if (r['case_id'],r['arm'],r['method'])==(cid,arm,control))
                    for e in ['optimal','feasible']:
                        ka,kb=a[e+'_count'],b[e+'_count'];_,p=fisher_exact([[ka,a['shots']-ka],[kb,b['shots']-kb]],alternative='two-sided')
                        secondary.append(dict(case_id=cid,arm=arm,method=method,control=control,event=e,
                             test_count=ka,control_count=kb,test_shots=a['shots'],control_shots=b['shots'],
                             difference=a['p_'+e]-b['p_'+e],p_value=float(p)))
    order=np.argsort([r['p_value'] for r in secondary]);running=0.
    for rank,index in enumerate(order):
        running=max(running,min(1.,(len(secondary)-rank)*secondary[index]['p_value']));secondary[index]['p_holm']=running
    for name,table in [('hardware_pooled',rows),('hardware_by_block',blocks),('primary_contrasts',tests),('secondary_contrasts',secondary)]:
        with (OUT/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    assert sum(r['shots'] for r in rows)==plan['planned_total_shots']
    qpu_seconds=sum(float(r['job_metrics'].get('usage',{}).get('quantum_seconds',0)) for r in raw)
    save(OUT/'hardware_analysis.json',dict(completed_utc=now(),total_shots=sum(r['shots'] for r in rows),rows=rows,primary_tests=tests,secondary_tests=secondary,
         independent_count_sets=len(blocks),reported_qpu_seconds=qpu_seconds,
         confidence_intervals='Pointwise exact Clopper-Pearson 95%; Fisher tests use a Holm family of 8. No population generalization from two known instances.'))
    print(json.dumps([dict(case=r['case_id'],arm=r['arm'],method=r['method'],optimal=r['optimal_count'],feasible=r['feasible_count'],shots=r['shots']) for r in rows],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action');p.add_argument('args',nargs='*');a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='submit':submit(int(a.args[0]),a.args[1])
    elif a.action=='collect':collect()
    elif a.action=='analyze':analyze()
    else:raise ValueError(a.action)
