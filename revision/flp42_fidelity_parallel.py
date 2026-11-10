"""Parallel, larger-budget fidelity follow-up on the saved 42-qubit instance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/flp42_fidelity_parallel.json'
EXTRA=""

def settings():
    c=json.loads(CONFIG.read_text());return c,ROOT/c['output_dir'],ROOT/c['parent_dir']
def now():return datetime.now(timezone.utc).isoformat()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def runner():return [sys.executable,'-m','revision.local_runner','--extra-site',EXTRA,'revision.flp42_fidelity_parallel']
def state(qc,bond,cutoff):
    from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
    from .full_circuit_explicit import explicit_settings
    return tensornetwork_from_circuit(qc,explicit_settings(bond,cutoff))

def freeze():
    import importlib.metadata
    c,out,parent=settings()
    for name in ['targets','fits','circuits','logs','checkpoints','source_snapshot']:(out/name).mkdir(parents=True,exist_ok=True)
    sources=['configs/flp42_fidelity_parallel.json','revision/flp42_fidelity_parallel.py','revision/cpu_memory_guard.py',
        'revision/tensor_prefix_core.py','revision/full_circuit_explicit.py','revision/steps_native_sweep.py',
        'revision/tensor_prefix_sabre.py','tests/test_flp42_fidelity_parallel.py']
    inputs=[parent/'cases/flp42.json',parent/'targets/flp42_half.pkl',parent/'targets/flp42_half.json',
        parent/'circuits/flp42_half_target.qpy',parent/'circuits/flp42_full_target.qpy',ROOT/c['target_snapshot']]
    inputs += list((parent/'fits').glob('flp42_half_l*.json'))
    record=dict(config=c,source_sha256={p:digest(ROOT/p) for p in sources},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs})
    path=out/'protocol.json'
    if path.exists():assert all(json.loads(path.read_text())[k]==v for k,v in record.items()),'Frozen protocol changed'
    else:
        save(path,dict(frozen_utc=now(),prior_fidelity_results_known=True,packages={p:importlib.metadata.version(p) for p in ['qiskit','quimb','qiskit-addon-aqc-tensor','numpy','scipy']},**record))
        for p in sources:shutil.copy2(ROOT/p,out/'source_snapshot'/Path(p).name)

def choose(records,layers=None):
    accepted=[r for r in records if r.get('verified') and (layers is None or r['layers']==layers)]
    return max(accepted,key=lambda r:r['fidelity']) if accepted else None

def checkpoint_fit(target,ansatz,x0,c,checkpoint=None):
    import numpy as np
    from scipy.optimize import minimize
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from .full_circuit_explicit import explicit_settings
    from .tensor_prefix_core import fidelity
    normalized=target.copy();normalized._psi/=np.sqrt(float(abs(normalized.psi.H@normalized.psi)))
    objective=MaximizeStateFidelity(normalized,ansatz,explicit_settings(c['fit_bond'],c['fit_cutoff']))
    start=time.monotonic();cpu_start=time.process_time();history=[];best=dict(loss=float('inf'),parameters=np.asarray(x0))
    class BudgetStop(Exception):pass
    def fun(theta):
        if history and (len(history)>=c['fit_max_evaluations'] or time.monotonic()-start>=c['fit_seconds']):raise BudgetStop()
        loss,gradient=objective.loss_function(theta);loss=float(loss);gradient=np.asarray(gradient,float)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):raise ValueError('Nonfinite loss or gradient')
        history.append(dict(evaluation=len(history)+1,seconds=time.monotonic()-start,loss=loss))
        if loss<best['loss']:best.update(loss=loss,parameters=np.array(theta))
        if checkpoint:
            save(checkpoint,dict(parameters=best['parameters'].tolist(),objective_fidelity=1-best['loss'],
                evaluations=len(history),history=history,seconds=time.monotonic()-start,updated_utc=now()))
        return loss,gradient
    try:
        result=minimize(fun,x0,jac=True,method='L-BFGS-B',options=dict(maxiter=c['fit_maxiter'],maxls=12,ftol=1e-11,gtol=1e-7))
        status=str(result.message);iterations=int(result.nit)
    except BudgetStop:status='budget_best_completed_evaluation';iterations=None
    except MemoryError:
        if not history:raise
        status='allocation_failure_best_completed_evaluation';iterations=None
    assert history
    fit_seconds=time.monotonic()-start
    bound=ansatz.assign_parameters(best['parameters'])
    low=state(bound,c['fit_bond'],c['check_cutoff']);high=state(bound,c['check_bond'],c['check_cutoff'])
    f=fidelity(normalized,high);convergence=fidelity(low,high);difference=abs(1-best['loss']-f)
    record=dict(parameters=best['parameters'].tolist(),fidelity=f,initial_fidelity=1-history[0]['loss'],
        objective_fidelity=1-best['loss'],ansatz_bond_convergence=convergence,objective_difference=difference,
        verified=bool(convergence>=c['convergence_min'] and difference<c['objective_tolerance']),
        status=status,iterations=iterations,evaluations=len(history),history=history,
        fitting_seconds=fit_seconds,seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start)
    return bound,record

def target_worker(endpoint,bond):
    import signal
    from .nighthawk_study import qpy_read
    from .steps_native_sweep import alarm_handler,BudgetExpired
    from .recover_full_circuit_deadline import is_deadline_exception
    c,out,parent=settings();started=time.monotonic();name=f'{endpoint}_b{bond}'
    qc=qpy_read(parent/'circuits'/f'flp42_{endpoint}_target.qpy')
    signal.signal(signal.SIGALRM,alarm_handler);signal.alarm(c['target_seconds_per_bond'])
    result=None
    try:result=state(qc,bond,c['fit_cutoff']);status='completed'
    except (BudgetExpired,MemoryError):status='target_budget_exceeded'
    except SystemError as exc:
        if is_deadline_exception(exc):status='target_budget_exceeded'
        else:raise
    finally:signal.alarm(0)
    record=dict(endpoint=endpoint,bond=bond,status=status,seconds=time.monotonic()-started,completed_utc=now(),affinity=sorted(os.sched_getaffinity(0)))
    if result is not None:
        p=out/'targets'/(name+'.pkl');p.write_bytes(pickle.dumps(result))
        record.update(pickle_sha256=digest(p),norm=float(abs(result.psi.H@result.psi)),achieved_bond=result.psi.max_bond())
    save(out/'targets'/(name+'.json'),record);print(name,status,flush=True)

def half_seed(layers,index):
    import numpy as np
    from .steps_native_sweep import extend_parameters
    c,out,parent=settings();case=json.loads((parent/'cases/flp42.json').read_text());edges=case['patch']['edges']
    if layers==6:source=json.loads((parent/'fits'/f'flp42_half_l{2 if index==3 else 6}.json').read_text())
    else:
        rows=[json.loads((parent/'fits/flp42_half_l6.json').read_text())]
        rows += [json.loads(p.read_text()) for p in (out/'fits').glob('half_l6_r*.json')]
        source=choose(rows,6)
    assert source and source['verified']
    x=extend_parameters(source['parameters'],42,len(edges),source['layers'],layers)
    x+=np.random.default_rng(c['seed']+layers*10+index).normal(0,c['perturbations'][index],len(x))
    return x,dict(source_layers=source['layers'],source_fidelity=source['fidelity'],source_parameter_sha256=hashlib.sha256(json.dumps(source['parameters']).encode()).hexdigest(),perturbation=c['perturbations'][index])

def fit_worker(endpoint,layers,index):
    import numpy as np
    from .tensor_prefix_core import native_ansatz
    from .steps_native_sweep import extend_parameters
    from .tensor_prefix_sabre import compile_isa_sabre
    from .nighthawk_study import qpy_write
    c,out,parent=settings();case=json.loads((parent/'cases/flp42.json').read_text());name=f'{endpoint}_l{layers}_r{index}'
    ansatz=native_ansatz(42,case['patch']['edges'],layers)
    if endpoint=='half':
        target=pickle.loads((parent/'targets/flp42_half.pkl').read_bytes());x,init=half_seed(layers,index)
    else:
        selection=json.loads((out/'targets/full_selection.json').read_text());assert selection['status']=='converged'
        target=pickle.loads((out/'targets'/f"full_b{selection['selected_bond']}.pkl").read_bytes())
        if layers==0:
            x=np.random.default_rng(c['seed']).normal(0,.02,ansatz.num_parameters);init=dict(kind='product_normal_0.02')
        elif index<2:
            candidates=[json.loads((parent/'fits/flp42_half_l6.json').read_text())]+[json.loads(p.read_text()) for p in (out/'fits').glob('half_l6_r*.json')]
            source=choose(candidates,6);x=np.asarray(source['parameters'])
            if layers>6:x=extend_parameters(x,42,len(case['patch']['edges']),6,layers)
            x=x+np.random.default_rng(c['seed']+100+index).normal(0,c['perturbations'][index],len(x))
            init=dict(kind='best_half_l6',source_fidelity=source['fidelity'],perturbation=c['perturbations'][index])
        else:
            source=json.loads((out/'fits/full_l0_r0.json').read_text());assert source['verified']
            x=extend_parameters(source['parameters'],42,len(case['patch']['edges']),0,layers)
            perturb=0. if index==2 else .02;x+=np.random.default_rng(c['seed']+100+index).normal(0,perturb,len(x))
            init=dict(kind='full_product_extension',source_fidelity=source['fidelity'],perturbation=perturb)
    c=dict(c,fit_seconds=120 if layers==0 else c[f'fit_seconds_{layers}'])
    bound,record=checkpoint_fit(target,ansatz,x,c,out/'checkpoints'/(name+'.json'))
    backend=pickle.loads((ROOT/c['target_snapshot']).read_bytes())
    isa,res=compile_isa_sabre(bound,backend,case['patch']['physical'],case['patch']['edges'],seed=81003)
    assert res['swap_count']==0 and res['cz_count']==2*len(case['patch']['edges'])*layers
    logical=out/'circuits'/(name+'.qpy');hardware=out/'circuits'/(name+'_isa.qpy')
    qpy_write(logical,bound);qpy_write(hardware,isa)
    record.update(endpoint=endpoint,layers=layers,start=index,initialization=init,parameter_count=ansatz.num_parameters,
        resources={k:v for k,v in res.items() if k!='routing_trace'},logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=digest(logical),
        isa_qpy=str(hardware.relative_to(ROOT)),isa_sha256=digest(hardware),affinity=sorted(os.sched_getaffinity(0)),completed_utc=now())
    save(out/'fits'/(name+'.json'),record);print(name,record['fidelity'],record['verified'],flush=True)

def select_targets():
    from .tensor_prefix_core import fidelity
    c,out,parent=settings();completed={};comparisons=[]
    for bond in c['full_target_bonds']:
        p=out/'targets'/f'full_b{bond}.json'
        if p.exists() and json.loads(p.read_text())['status']=='completed':completed[bond]=pickle.loads((out/'targets'/f'full_b{bond}.pkl').read_bytes())
    chosen=None
    for low,high in zip(c['full_target_bonds'],c['full_target_bonds'][1:]):
        if low in completed and high in completed:
            f=fidelity(completed[low],completed[high]);comparisons.append(dict(low_bond=low,high_bond=high,fidelity=f))
            if chosen is None and f>=c['convergence_min']:chosen=high
    save(out/'targets/full_selection.json',dict(status='converged' if chosen else 'unresolved',selected_bond=chosen,comparisons=comparisons,completed_utc=now()))
    p=out/'targets/half_b256.pkl'
    if p.exists():
        old=pickle.loads((parent/'targets/flp42_half.pkl').read_bytes());new=pickle.loads(p.read_bytes())
        save(out/'targets/half_sensitivity.json',dict(old_bond=128,new_bond=256,fidelity=fidelity(old,new),note='Primary fits still use the archived target; this is an independent tighter-cutoff sensitivity check.'))
    print('Full target selection',chosen,comparisons,flush=True)

def job_worker(job,cpu):
    assert cpu in os.sched_getaffinity(0);os.sched_setaffinity(0,{cpu})
    started=time.monotonic();cpu_started=time.process_time()
    try:
        if job['kind']=='target':target_worker(job['endpoint'],job['bond'])
        else:fit_worker(job['endpoint'],job['layers'],job['start'])
    finally:
        import resource
        _,out,_=settings();save(out/'logs'/(job['name']+'_usage.json'),dict(seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu_started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,affinity=[cpu]))

def task(endpoint,layers=None,index=0,bond=None):
    if bond is not None:return dict(kind='target',endpoint=endpoint,bond=bond,name=f'{endpoint}_b{bond}',result=f'targets/{endpoint}_b{bond}.json')
    name=f'{endpoint}_l{layers}_r{index}';return dict(kind='fit',endpoint=endpoint,layers=layers,start=index,name=name,result=f'fits/{name}.json')

def pool(jobs):
    c,out,_=settings();cpus=sorted(os.sched_getaffinity(0));assert len(cpus)==c['physical_cores']
    pending=[j for j in jobs if not (out/j['result']).exists()];active=[]
    try:
        while pending or active:
            free=[cpu for cpu in cpus if cpu not in [a['cpu'] for a in active]]
            while pending and free:
                job=pending.pop(0);cpu=free.pop(0);log=open(out/'logs'/(job['name']+'.log'),'a')
                child=subprocess.Popen(runner()+['worker',json.dumps(job),'--cpu',str(cpu)],stdout=log,stderr=subprocess.STDOUT)
                budget=c['target_seconds_per_bond']+90 if job['kind']=='target' else (120 if job['layers']==0 else c[f"fit_seconds_{job['layers']}"])+300
                active.append(dict(job=job,cpu=cpu,child=child,log=log,started=time.monotonic(),deadline=budget))
                print('Started',job['name'],'CPU',cpu,flush=True)
            for item in active[:]:
                child=item['child'];expired=time.monotonic()-item['started']>item['deadline']
                if expired and child.poll() is None:child.kill()
                if child.poll() is None:continue
                code=child.wait();item['log'].close();job=item['job'];path=out/job['result']
                process=dict(returncode=code,seconds=time.monotonic()-item['started'],cpu=item['cpu'],status='wall_time_limit' if expired else ('completed' if code==0 else 'worker_failed'))
                save(out/'logs'/(job['name']+'_process.json'),process)
                if not path.exists():save(path,dict(**job,status='worker_failed',verified=False,process=process))
                row=json.loads(path.read_text());print('Finished',job['name'],row.get('fidelity'),row.get('status'),flush=True);active.remove(item)
            time.sleep(.2)
    finally:
        for item in active:
            if item['child'].poll() is None:item['child'].kill();item['child'].wait()
            item['log'].close()

def experiment():
    c,out,_=settings()
    # Long target constructions overlap independent half-target restarts.
    jobs=[task('full',bond=b) for b in reversed(c['full_target_bonds'])]
    jobs += [task('half',6,i) for i in range(c['starts'])]
    jobs += [task('half',bond=c['half_reference_check_bond'])]
    pool(jobs);select_targets()
    selected=json.loads((out/'targets/full_selection.json').read_text())
    if selected['status']=='converged':pool([task('full',0,0)])
    jobs=[task('half',10,i) for i in range(c['starts'])]
    if selected['status']=='converged' and json.loads((out/'fits/full_l0_r0.json').read_text()).get('verified'):
        jobs += [task('full',6,i) for i in range(c['starts'])]
    pool(jobs)
    fits=[json.loads(p.read_text()) for p in sorted((out/'fits').glob('*.json'))]
    rows=[]
    for endpoint,layers in [('half',6),('half',10),('full',6)]:
        records=[r for r in fits if r['endpoint']==endpoint and r['layers']==layers];best=choose(records,layers)
        rows.append(dict(endpoint=endpoint,layers=layers,attempts=len(records),verified=sum(bool(r.get('verified')) for r in records),
            best_fidelity=best['fidelity'] if best else None,best_start=best['start'] if best else None,
            cz_depth=best['resources']['cz_depth'] if best else None,cz_count=best['resources']['cz_count'] if best else None,
            threshold_met=bool(best and best['fidelity']>=c['fit_threshold'])))
    save(out/'summary.json',dict(completed_utc=now(),rows=rows,full_target=selected,new_hardware_shots=0))
    print('FIDELITY EXPERIMENT COMPLETE',flush=True)

def run():
    import re
    from .cpu_memory_guard import guarded_run
    freeze();c,out,_=settings()
    log=(out/'tests.log').read_text();assert 'FAILED' not in log and 'ERROR' not in log
    assert int(re.findall(r'(\d+) passed',log)[-1])>=110
    if (out/'summary.json').exists():print('Completed experiment already exists');return
    result=guarded_run(runner()+['experiment'],out/'experiment.log',cores=c['physical_cores'],threads=1,
        address_gib=c['process_address_gib'],rss_gib=c['aggregate_rss_gib'],reserve_gib=c['system_reserve_gib'],seconds=10800)
    save(out/'experiment_process.json',result);print(result,flush=True)
    if result['status']!='completed':raise RuntimeError(result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['run','experiment','worker','select_targets']);p.add_argument('job',nargs='?');p.add_argument('--cpu',type=int);a=p.parse_args()
    if a.action=='worker':job_worker(json.loads(a.job),a.cpu)
    else:globals()[a.action]()
