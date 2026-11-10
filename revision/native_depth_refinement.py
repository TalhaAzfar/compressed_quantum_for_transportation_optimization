"""Refine all width-sweep instances with preserved starts and deeper native fits."""
import argparse
import json
import os
import pickle
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from .contribution_hardware_core import ROOT, read, save, digest, now
from .native_depth_benchmark import PARENT, OUT, load_target

CONFIG = ROOT/'configs/native_depth_refinement.json'
EXTRA = ""


def settings(): return read(CONFIG)


def depth_layers(case):
    base = next(r for r in case['rows'] if r['method'] == 'aqc_l2')
    per_layer = base['resources']['cz_depth']/2
    return [('product',0), ('base',2)]+[(f'd{b}',int(b//per_layer)) for b in settings()['new_native_depth_caps']]


def freeze():
    c = settings()
    for name in ['cases','logs','circuits','source_snapshot','raw_counts']:
        (OUT/name).mkdir(parents=True,exist_ok=True)
    tests = (OUT/'numerical_tests.log').read_text()
    assert '2 passed' in tests and 'FAILED' not in tests
    names = ['configs/native_depth_refinement.json','revision/native_depth_refinement.py',
        'revision/native_depth_objective.py','revision/native_depth_benchmark.py',
        'revision/cpu_memory_guard.py','revision/hardware_width_sweep.py',
        'revision/tensor_prefix_core.py','revision/tensor_prefix_sabre.py',
        'revision/steps_native_sweep.py','revision/full_circuit_compression.py',
        'revision/nighthawk_study.py','revision/width_sweep_models.py',
        'revision/depth200_native_cleanup.py','tests/test_native_depth_refinement.py']
    inputs = [PARENT/'release_checksums.json',PARENT/'preflight/target.pkl']
    inputs += [PARENT/'cases'/s['id']/'complete.json' for s in c['cases']]
    record = dict(config=c,source_sha256={p:digest(ROOT/p) for p in names},
                  input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs})
    path = OUT/'preparation_protocol.json'
    if path.exists():
        old = read(path); assert all(old[k] == v for k,v in record.items()), 'Frozen preparation changed'
    else:
        save(path,dict(frozen_utc=now(),prior_hardware_outcomes_known=True,new_hardware_outcomes_seen=False,**record))
        for p in names: shutil.copy2(ROOT/p, OUT/'source_snapshot'/Path(p).name)


def parent_candidates(case):
    return [dict(id='parent_'+r['method'],case_id=r['case_id'],layers=r['fit']['layers'],
         parameters=r['fit']['parameters'],exact_fidelity=r['fit']['exact_fidelity'],
         ideal_events=r['ideal_events'],verified=True,parent_row=r)
         for r in case['rows'] if r['fit']]


def fit_one(objective, ansatz, x0, config, checkpoint, *, reference, target_dense, model):
    import numpy as np
    from scipy.optimize import minimize
    from .hardware_width_sweep import exact_state,state_events
    started = time.monotonic(); history = []; best = dict(loss=float('inf'),parameters=np.array(x0))
    cap = config['fit_seconds_large'] if model['n_qubits'] >= 25 else config['fit_seconds_small']
    class FitLimit(Exception): pass
    def fun(x):
        if history and (len(history)>=config['fit_max_evaluations'] or time.monotonic()-started>=cap): raise FitLimit()
        loss, gradient = objective.loss_function(x); loss=float(loss); gradient=np.asarray(gradient,float)
        assert np.isfinite(loss) and np.isfinite(gradient).all()
        history.append(dict(evaluation=len(history)+1,seconds=time.monotonic()-started,loss=loss,
                            gradient_inf_norm=float(np.max(np.abs(gradient)))))
        if loss < best['loss']:
            best.update(loss=loss,parameters=np.array(x))
            save(checkpoint,dict(updated_utc=now(),parameters=best['parameters'].tolist(),loss=loss,
                 evaluations=len(history),seconds=time.monotonic()-started,history=history))
        return loss,gradient
    try:
        result = minimize(fun,x0,jac=True,method='L-BFGS-B',options=dict(
            maxiter=config['fit_maxiter'],maxls=20,ftol=config['fit_ftol'],gtol=config['fit_gtol']))
        status=str(result.message); iterations=int(result.nit); converged=bool(result.success)
    except FitLimit:
        status='evaluation_or_time_budget_best_completed_checkpoint'; iterations=None; converged=False
    assert history
    bound=ansatz.assign_parameters(best['parameters']); actual=exact_state(bound)
    norm=float(np.vdot(actual,actual).real); assert abs(norm-1)<1e-8
    exact_f=float(abs(np.vdot(reference,actual))**2)
    target_f=float(abs(np.vdot(target_dense,actual))**2)
    error=abs(target_f-(1-best['loss'])); assert error<1e-8,(error,target_f,best['loss'])
    events=state_events(actual,model)
    return bound,dict(parameters=best['parameters'].tolist(),exact_fidelity=exact_f,
        target_fidelity=target_f,objective_difference=error,ideal_events=events,
        initial_objective_fidelity=1-history[0]['loss'],objective_fidelity=1-best['loss'],
        history=history,evaluations=len(history),iterations=iterations,status=status,converged=converged,
        adequate_evaluation_budget=converged or len(history)>=config['minimum_evaluations_if_not_converged'],
        fitting_seconds=history[-1]['seconds'],total_seconds=time.monotonic()-started,
        seconds_cap=cap,verified=True,exact_output_norm=norm)


def worker(cid):
    import numpy as np
    from .native_depth_objective import make_objective
    from .tensor_prefix_core import native_ansatz
    from .steps_native_sweep import extend_parameters
    from .full_circuit_compression import mps_dense_qiskit
    from .tensor_prefix_sabre import compile_isa_sabre
    from .depth200_native_cleanup import cleanup,cleaned_resources
    from .nighthawk_study import qpy_write
    c=settings(); folder=OUT/'cases'/cid; folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'complete.json').exists()
    case,reference,target=load_target(cid); model=case['model']; n=model['n_qubits']
    target_dense=mps_dense_qiskit(target,n); target_dense/=np.linalg.norm(target_dense)
    source_check=float(abs(np.vdot(reference,target_dense))**2); assert source_check>=c['minimum_target_fidelity']
    save(folder/'inputs.json',dict(case_id=cid,model=model,physical=case['physical'],edges=case['edges'],
        target_fidelity=source_check,parent_case_sha256=digest(PARENT/'cases'/cid/'complete.json'),
        target_mps_sha256=digest(OUT/'targets'/cid/'target.pkl'),reference_sha256=digest(OUT/'targets'/cid/'reference.npy')))
    backend=pickle.loads((PARENT/'preflight/target.pkl').read_bytes())
    candidates=parent_candidates(case); rows=[]
    for label,layers in depth_layers(case):
        ansatz=native_ansatz(n,case['edges'],layers)
        prior=[r for r in candidates if r['layers']<=layers]
        print(cid,label,'creating contraction',flush=True)
        objective,contraction=make_objective(target,ansatz,repeats=c['contraction_repeats'],
            max_tensor_bytes=c['max_tensor_bytes'],max_slice_write_bytes=c['max_slice_write_bytes'],
            max_slices=c['max_slices'],seed=c['parameter_seed']+layers)
        save(folder/f'{label}_contraction.json',contraction)
        print(cid,label,'slices',contraction['slices'],'estimated slice bytes',contraction['estimated_write_bytes_per_slice'],flush=True)
        for start in range(c['product_starts'] if layers==0 else c['native_starts']):
            name=f'{cid}_{label}_r{start}'; path=folder/(name+'.json')
            if path.exists():
                row=read(path); candidates.append(row); rows.append(row); continue
            available=[r for r in candidates if r['layers']<=layers]
            if start==2:
                available=[r for r in available if r['layers']==0]
            old=max(available,key=lambda r:r['exact_fidelity'])
            x=extend_parameters(old['parameters'],n,len(case['edges']),old['layers'],layers)
            perturbation=c['start_perturbations'][start]
            if perturbation:
                x+=np.random.default_rng(c['parameter_seed']+model['seed']+100*layers+start).normal(0,perturbation,len(x))
            initialization=dict(source_candidate=old['id'],source_layers=old['layers'],
                source_exact_fidelity=old['exact_fidelity'],perturbation=perturbation,
                identity_preserving_extension=perturbation==0)
            bound,row=fit_one(objective,ansatz,x,c,folder/(name+'_checkpoint.json'),
                reference=reference,target_dense=target_dense,model=model)
            logical=OUT/'circuits'/(name+'.qpy'); isa_path=OUT/'circuits'/(name+'_isa.qpy')
            qpy_write(logical,bound)
            isa,resources=compile_isa_sabre(bound,backend,case['physical'],case['edges'])
            assert resources['swap_count']==0
            isa,certificate=cleanup(isa); resources=cleaned_resources(isa,resources,backend,certificate)
            cap=0 if layers==0 else (next(r for r in case['rows'] if r['method']=='aqc_l2')['resources']['cz_depth'] if label=='base' else int(label[1:]))
            assert resources['cz_depth']<=cap,(name,resources['cz_depth'],cap)
            qpy_write(isa_path,isa)
            row.update(id=name,case_id=cid,family=model['problem'].upper(),n_qubits=n,label=label,layers=layers,start=start,
                initialization=initialization,resources=resources,cz_depth_cap=cap,
                logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=digest(logical),
                isa_qpy=str(isa_path.relative_to(ROOT)),isa_sha256=digest(isa_path),
                contraction_sha256=digest(folder/f'{label}_contraction.json'),completed_utc=now())
            save(path,row); candidates.append(row); rows.append(row)
            print(name,'F',row['exact_fidelity'],'feasible',row['ideal_events']['p_feasible'],
                  'evaluations',row['evaluations'],'converged',row['converged'],flush=True)
        del objective
    save(folder/'complete.json',dict(completed_utc=now(),case_id=cid,rows=rows,
        all_fits_verified=all(r['verified'] for r in rows),
        adequate_fits=sum(r['adequate_evaluation_budget'] for r in rows),
        protocol_sha256=digest(OUT/'preparation_protocol.json')))


def run():
    import psutil
    from .cpu_memory_guard import physical_cpu_ids
    freeze(); c=settings(); cpus=physical_cpu_ids()
    assert len(cpus)>=c['parallel_workers']*c['physical_cores_per_worker']
    pending=[s['id'] for s in c['cases'] if not (OUT/'cases'/s['id']/'complete.json').exists()]
    # Start one case of each setting first, keeping every requested case.
    pending.sort(key=lambda s:(int(s.rsplit('r',1)[1]),int(''.join(x for x in s.split('_')[0] if x.isdigit()))))
    active={}; finished={}; peak=0; started=time.monotonic(); parent=psutil.Process()
    try:
        while pending or active:
            for slot in [i for i in range(c['parallel_workers']) if i not in active]:
                if not pending: break
                if psutil.virtual_memory().available<c['system_reserve_bytes']+1000000000: break
                cid=pending.pop(0); log=(OUT/'logs'/f'{cid}.log').open('a')
                allocated=cpus[slot*c['physical_cores_per_worker']:(slot+1)*c['physical_cores_per_worker']]
                command=[sys.executable,'-m','revision.cpu_memory_guard','--address-gib',str(c['worker_address_gib']),
                    '--cpus',','.join(map(str,allocated)),'--threads',str(c['numerical_threads_per_worker']),'--',
                    sys.executable,'-m','revision.local_runner','--extra-site',EXTRA,'revision.native_depth_refinement','worker',cid]
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[slot]=dict(child=child,log=log,case_id=cid,started=time.monotonic(),cpus=allocated,peak_rss=0)
                print('Started',cid,'PID',child.pid,'CPUs',allocated,flush=True)
            total=parent.memory_info().rss
            for slot,item in list(active.items()):
                child=item['child']
                if child.poll() is not None:
                    item['log'].close(); ok=child.returncode==0 and (OUT/'cases'/item['case_id']/'complete.json').exists()
                    record=dict(status='complete' if ok else 'failed',returncode=child.returncode,
                        seconds=time.monotonic()-item['started'],cpu_ids=item['cpus'],peak_worker_rss_bytes=item['peak_rss'])
                    finished[item['case_id']]=record; save(OUT/'logs'/(item['case_id']+'_process.json'),record)
                    print(item['case_id'],record['status'],flush=True); del active[slot]; continue
                rss=0
                try:
                    process=psutil.Process(child.pid)
                    for member in [process]+process.children(recursive=True):
                        try: rss+=member.memory_info().rss
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                item['peak_rss']=max(item['peak_rss'],rss); total+=rss
                if rss>c['worker_rss_limit_bytes']: raise MemoryError(f"Worker {item['case_id']} reached its RSS guard")
            peak=max(peak,total)
            if total>c['aggregate_rss_limit_bytes']: raise MemoryError('Aggregate 35 GB memory guard')
            if psutil.virtual_memory().available<c['system_reserve_bytes']: raise MemoryError('System available-memory reserve')
            save(OUT/'preparation_progress.json',dict(updated_utc=now(),finished=finished,
                active=[dict(case_id=i['case_id'],cpu_ids=i['cpus'],seconds=time.monotonic()-i['started'],peak_rss_bytes=i['peak_rss']) for i in active.values()],
                pending=pending,peak_total_rss_bytes=peak,elapsed_seconds=time.monotonic()-started))
            time.sleep(.5)
        failed=[k for k,v in finished.items() if v['status']!='complete']
        assert not failed,failed
        assert all((OUT/'cases'/s['id']/'complete.json').exists() for s in c['cases'])
        save(OUT/'preparation_complete.json',dict(completed_utc=now(),cases=27,peak_total_rss_bytes=peak,
             protocol_sha256=digest(OUT/'preparation_protocol.json')))
    finally:
        for item in active.values():
            try: os.killpg(item['child'].pid,signal.SIGKILL)
            except ProcessLookupError: pass
            item['child'].wait(); item['log'].close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','worker']);p.add_argument('case',nargs='?');a=p.parse_args()
    if a.action=='worker': worker(a.case)
    else: {'freeze':freeze,'run':run}[a.action]()
