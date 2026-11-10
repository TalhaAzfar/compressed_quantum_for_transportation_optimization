"""Complete QAOA output compression with a declared native depth ladder."""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from .depth200_optimum_screen import ROOT, OUT, EXTRA, save, case_data, mps_metrics


def exact_state(qc):
    import numpy as np
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_statevector
    assert qc.num_qubits <= 25
    saved=qc.copy();saved.save_statevector()
    result=AerSimulator(method='statevector',max_parallel_threads=1,
                       max_parallel_experiments=1,max_memory_mb=4096).run(saved,shots=1).result()
    assert result.success
    return np.asarray(result.data(0)['statevector'])


def reference_worker(cid, bond):
    from .nighthawk_study import qpy_read
    from .placement_aer_reference import reference
    from .flp42_fidelity_parallel import digest, now
    from .full_circuit_compression import mps_dense_qiskit
    import numpy as np
    started=time.monotonic();case=case_data(cid);model=case['model']
    source=ROOT/case['methods']['standard_p2']['logical_qpy'];qc=qpy_read(source)
    state,energy=reference(qc,model,bond,1e-13)
    row=dict(case_id=cid,source='standard_p2',bond=bond,energy=energy,
             events=mps_metrics(model,state),source_sha256=digest(source),verified=False)
    if model['n_qubits']<=25:
        vector=exact_state(qc)
        exact_path=OUT/'references'/f'{cid}_standard_p2_exact.npy'
        if not exact_path.exists():
            temp=exact_path.with_name(exact_path.stem+f'_b{bond}_tmp.npy')
            np.save(temp,vector);temp.replace(exact_path)
        dense=mps_dense_qiskit(state,model['n_qubits'])
        overlap=float(abs(np.vdot(vector,dense))**2/np.vdot(dense,dense).real)
        old=json.loads((OUT/'screen'/f'{cid}_standard_p2_exact.json').read_text())
        relative={key:abs(row['events'][key]-old['events'][key])/old['events'][key]
                  for key in ['p_feasible','p_good','p_optimal']}
        row.update(exact_fidelity=overlap,event_relative_errors=relative,
                   verified=bool(overlap>=.999 and max(relative.values())<=.05))
    path=OUT/'references'/f'{cid}_standard_p2_b{bond}.pkl'
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(pickle.dumps(state))
    row.update(state_pickle=str(path.relative_to(ROOT)),state_sha256=digest(path),seconds=time.monotonic()-started,completed_utc=now())
    save(path.with_suffix('.json'),row)
    print(cid,bond,{k:v for k,v in row.items() if k not in ['state_sha256','source_sha256']},flush=True)


def euler_extend(parameters,n,edge_count,old_layers,new_layers):
    import numpy as np
    assert new_layers>=old_layers>=0
    x=np.asarray(parameters,float)
    assert len(x)==3*n*(old_layers+1)+edge_count*old_layers
    return np.r_[x,np.zeros((new_layers-old_layers)*(3*n+edge_count))]


def projected_qaoa_initialization(model,edges,angles,layers):
    """Embed QAOA with only native interactions; full target is unaltered."""
    import numpy as np
    from .native_aqc import edge_matchings
    angles=np.asarray(angles).reshape(-1,2);p=len(angles);assert layers>=p
    n=model['n_qubits'];q=np.asarray(model['quadratic']);linear=np.asarray(model['linear'])
    h=-linear/2-(q.sum(0)+q.sum(1))/4;j=(q+q.T)/4;values=[]
    for k,(gamma,beta) in enumerate(angles):
        for site in range(n):
            values.extend([0.,0.,2*gamma*h[site]] if k==0 else
                          [np.pi/2,2*angles[k-1,1],-np.pi/2+2*gamma*h[site]])
        for matching in edge_matchings(edges):
            for u,v in matching:values.append(2*gamma*j[u,v])
    for site in range(n):values.extend([np.pi/2,2*angles[-1,1],-np.pi/2])
    return euler_extend(values,n,len(edges),p,layers)


def select_reference(cid):
    from .tensor_prefix_core import fidelity
    rows=[json.loads(p.read_text()) for p in (OUT/'references').glob(f'{cid}_standard_p2_b*.json')]
    rows.sort(key=lambda r:r['bond']);comparisons=[]
    if cid=='tsp25':
        accepted=[r for r in rows if r['verified']]
        chosen=min(accepted,key=lambda r:r['bond']) if accepted else None
    else:
        chosen=None
        for a,b in zip(rows,rows[1:]):
            x=pickle.loads((ROOT/a['state_pickle']).read_bytes());y=pickle.loads((ROOT/b['state_pickle']).read_bytes())
            overlap=fidelity(x,y);relative={key:abs(a['events'][key]-b['events'][key])/b['events'][key]
                                          for key in ['p_feasible','p_good','p_optimal']}
            ok=overlap>=.999 and abs(a['energy']-b['energy'])<=1e-4 and max(relative.values())<=.05
            comparisons.append(dict(low=a['bond'],high=b['bond'],fidelity=overlap,event_relative_errors=relative,
                                    energy_difference=abs(a['energy']-b['energy']),verified=ok))
            if ok and chosen is None:chosen=b
    result=dict(case_id=cid,status='verified' if chosen else 'unresolved',chosen=chosen,comparisons=comparisons)
    save(OUT/'references'/f'{cid}_selection.json',result);print(cid,result['status'],flush=True)
    return result


def fit_worker(cid,layers,start):
    import numpy as np
    import quimb.tensor as qtn
    from .placement_study_core import native_euler_ansatz
    from .flp42_fidelity_parallel import checkpoint_fit,digest,now
    from .tensor_prefix_sabre import compile_isa_sabre
    from .nighthawk_study import qpy_write
    from .depth200_optimum_screen import feasible_values,metrics_from_amplitudes
    case=case_data(cid);model=case['model'];patch=case['patch'];n=model['n_qubits']
    selection=json.loads((OUT/'references'/f'{cid}_selection.json').read_text());assert selection['status']=='verified'
    saved=pickle.loads((ROOT/selection['chosen']['state_pickle']).read_bytes())
    target=qtn.CircuitMPS(N=n,psi0=saved.psi.copy())
    ansatz=native_euler_ansatz(n,patch['edges'],layers)
    name=f'{cid}_l{layers}_r{start}';rng=np.random.default_rng(110021+layers*10+start)
    prior=[json.loads(p.read_text()) for p in (OUT/'fits').glob(f'{cid}_l*_r*.json')]
    prior=[r for r in prior if r.get('verified') and r['layers']<layers]
    if layers==0:
        x=rng.normal(0,.02,ansatz.num_parameters);initialization=dict(kind='product_random')
    elif layers==2 and start==1:
        angles=case['methods']['standard_p2']['training']['parameters']
        x=projected_qaoa_initialization(model,patch['edges'],angles,layers)
        initialization=dict(kind='native_projection_of_source_QAOA',source_angles=angles)
    else:
        assert prior,'A verified shallower fit is required for identity extension'
        old=max(prior,key=lambda r:r['fidelity'])
        x=euler_extend(old['parameters'],n,len(patch['edges']),old['layers'],layers)
        if start:x+=rng.normal(0,.005,len(x))
        initialization=dict(kind='best_shallower_extension',source=old['name'],perturbation=.005 if start else 0.)
    budgets={0:120,2:240,6:360,12:540,24:720}
    c=dict(fit_bond=128,check_bond=256,fit_cutoff=1e-12,check_cutoff=1e-13,
           fit_max_evaluations=600,fit_maxiter=500,fit_seconds=budgets[layers],
           convergence_min=.999,objective_tolerance=2e-5)
    bound,row=checkpoint_fit(target,ansatz,x,c,OUT/'checkpoints'/(name+'.json'))
    row['optimizer_objective_consistent']=row['verified']
    row['target_reference_fidelity']=row['fidelity']
    # The independent evaluation is part of acceptance, not the fitting loss.
    from .flp42_fidelity_parallel import state
    high=state(bound,256,1e-13);row['events']=mps_metrics(model,high)
    if n<=25:
        from .full_circuit_compression import mps_dense_qiskit
        exact=exact_state(bound);original=np.load(OUT/'references'/f'{cid}_standard_p2_exact.npy',mmap_mode='r')
        row['exact_fidelity']=float(abs(np.vdot(original,exact))**2)
        dense=mps_dense_qiskit(high,n)
        row['ansatz_exact_overlap']=float(abs(np.vdot(exact,dense))**2/np.vdot(dense,dense).real)
        values=feasible_values(model);events=metrics_from_amplitudes(model,values,exact[values])
        relative={k:abs(row['events'][k]-events[k])/max(events[k],1e-300) for k in ['p_feasible','p_good','p_optimal']}
        row.update(mps_events=row['events'],events=events,event_relative_errors=relative)
        # Exact output probabilities and fidelity certify the candidate even if
        # the approximate training loss or MPS check is inaccurate. Such loss
        # discrepancies remain visible, rather than vetoing an exact result.
        row['exact_norm']=float(np.vdot(exact,exact).real)
        row['fidelity']=row['exact_fidelity']
        row['verified']=bool(abs(row['exact_norm']-1)<1e-8)
        row['verification']='Exact 25-qubit source/output fidelity and event probabilities; optimizer loss is a separate diagnostic.'
    else:
        from .tensor_prefix_core import fidelity
        higher=state(bound,512,1e-13);events=mps_metrics(model,higher)
        relative={k:abs(row['events'][k]-events[k])/max(events[k],1e-300) for k in ['p_feasible','p_good','p_optimal']}
        overlap=fidelity(high,higher)
        row.update(events=events,event_relative_errors=relative,validation_state_overlap=overlap,
                   fidelity=fidelity(target,higher),verified=bool(overlap>=.999 and max(relative.values())<=.05),
                   verification='Independent bond-256/512 output-state and rare-event convergence; not an exact certificate.')
    backend=pickle.loads((OUT/'preflight/target_snapshot.pkl').read_bytes())
    isa,resources=compile_isa_sabre(bound,backend,patch['physical'],patch['edges'],seed=81003)
    assert resources['swap_count']==0 and resources['cz_depth']<=200
    logical=OUT/'circuits'/(name+'.qpy');hardware=OUT/'circuits'/(name+'_isa.qpy')
    logical.parent.mkdir(parents=True,exist_ok=True);qpy_write(logical,bound);qpy_write(hardware,isa)
    row.update(name=name,case_id=cid,layers=layers,start=start,initialization=initialization,
               target_reference=selection['chosen'],fit_config=c,resources={k:v for k,v in resources.items() if k!='routing_trace'},
               logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=digest(logical),
               isa_qpy=str(hardware.relative_to(ROOT)),isa_sha256=digest(hardware),completed_utc=now())
    save(OUT/'fits'/(name+'.json'),row)
    print(name,'fidelity',row['fidelity'],'verified',row['verified'],'events',row['events'],flush=True)


def run_pool(jobs, label, deadline):
    cpus=sorted(os.sched_getaffinity(0));assert len(cpus)==4
    pending=list(jobs);active=[]
    while pending or active:
        free=[cpu for cpu in cpus if cpu not in [x['cpu'] for x in active]]
        while pending and free:
            job=pending.pop(0);name='_'.join(map(str,job));cpu=free.pop(0)
            logpath=OUT/'logs'/(name+'.log');logpath.parent.mkdir(parents=True,exist_ok=True);log=logpath.open('a')
            cmd=[sys.executable,'-m','revision.local_runner','--extra-site',EXTRA,
                 'revision.depth200_optimum_fit',*map(str,job),'--cpu',str(cpu)]
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            active.append(dict(child=child,cpu=cpu,log=log,name=name,started=time.monotonic()))
            print('Started',name,'CPU',cpu,flush=True)
        for item in active[:]:
            expired=time.monotonic()-item['started']>deadline
            if expired and item['child'].poll() is None:item['child'].kill()
            if item['child'].poll() is None:continue
            code=item['child'].wait();item['log'].close()
            save(OUT/'logs'/(item['name']+'_process.json'),dict(returncode=code,seconds=time.monotonic()-item['started'],
                 cpu=item['cpu'],status='wall_time_limit' if expired else ('completed' if code==0 else 'worker_failed')))
            print('Finished',item['name'],code,flush=True);active.remove(item)
        time.sleep(.2)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action');p.add_argument('args',nargs='*');p.add_argument('--cpu',type=int)
    a=p.parse_args()
    if a.cpu is not None:os.sched_setaffinity(0,{a.cpu})
    if a.action=='reference':reference_worker(a.args[0],int(a.args[1]))
    elif a.action=='fit':fit_worker(a.args[0],int(a.args[1]),int(a.args[2]))
    elif a.action=='select':select_reference(a.args[0])
    elif a.action=='reference_pool':run_pool([('reference',c,b) for c in ['tsp25','flp30'] for b in [256,512]],'reference',1500)
    elif a.action=='fit_pool':
        layer=int(a.args[0]);run_pool([('fit',c,layer,s) for c in ['tsp25','flp30'] for s in range(1 if layer==0 else 2)],f'fit_l{layer}',1800)
    else:raise ValueError(a.action)
