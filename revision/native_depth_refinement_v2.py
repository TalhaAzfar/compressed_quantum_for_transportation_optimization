"""Resume the frozen study with improved contraction planning and full checkpoints."""
import argparse
import copy
import os
import pickle
import shutil
import time
from pathlib import Path
from . import native_depth_refinement as original
from .native_depth_refinement import ROOT,OUT,PARENT,settings,parent_candidates,depth_layers
from .contribution_hardware_core import read,save,digest,now

AMEND=OUT/'amendments'/'contraction_planning_v2'
PLANNING_LIMITS=dict(max_tensor_bytes=536870912,max_slice_write_bytes=3000000000)


def fit_one(objective,ansatz,x0,config,checkpoint,*,reference,target_dense,model):
    import numpy as np
    from scipy.optimize import minimize
    from .hardware_width_sweep import exact_state,state_events
    started=time.monotonic();prior=read(checkpoint) if checkpoint.exists() else None
    history=copy.deepcopy(prior['history']) if prior else []
    recorded_before=len(history);seconds_before=float(prior['seconds']) if prior else 0.
    x0=np.array(prior['parameters'] if prior else x0)
    best=dict(loss=float(prior['loss']) if prior else float('inf'),parameters=x0.copy())
    cap=config['fit_seconds_large'] if model['n_qubits']>=25 else config['fit_seconds_small']
    class FitLimit(Exception):pass
    def fun(x):
        elapsed=seconds_before+time.monotonic()-started
        if len(history)>recorded_before and (len(history)>=config['fit_max_evaluations'] or elapsed>=cap):raise FitLimit()
        loss,gradient=objective.loss_function(x);loss=float(loss);gradient=np.asarray(gradient,float)
        assert np.isfinite(loss) and np.isfinite(gradient).all()
        if prior and len(history)==recorded_before:assert abs(loss-prior['loss'])<1e-8
        history.append(dict(evaluation=len(history)+1,seconds=seconds_before+time.monotonic()-started,
            loss=loss,gradient_inf_norm=float(np.max(np.abs(gradient)))))
        if loss<best['loss']:best.update(loss=loss,parameters=np.array(x))
        # Save every completed evaluation, including unsuccessful line searches.
        save(checkpoint,dict(updated_utc=now(),parameters=best['parameters'].tolist(),loss=best['loss'],
            evaluations=len(history),seconds=history[-1]['seconds'],history=history,
            recorded_evaluations_before_resume=recorded_before,planner='v2'))
        return loss,gradient
    if recorded_before>=config['fit_max_evaluations'] or seconds_before>=cap:
        status='interrupted_budget_exhausted_validate_saved_checkpoint';iterations=0;converged=False
    else:
        try:
            result=minimize(fun,x0,jac=True,method='L-BFGS-B',options=dict(
                maxiter=max(1,config['fit_maxiter']-recorded_before),maxls=20,
                ftol=config['fit_ftol'],gtol=config['fit_gtol']))
            status=str(result.message);iterations=int(result.nit);converged=bool(result.success)
        except FitLimit:
            status='evaluation_or_time_budget_best_completed_checkpoint';iterations=None;converged=False
    assert history
    bound=ansatz.assign_parameters(best['parameters']);actual=exact_state(bound)
    norm=float(np.vdot(actual,actual).real);assert abs(norm-1)<1e-8
    exact_f=float(abs(np.vdot(reference,actual))**2);target_f=float(abs(np.vdot(target_dense,actual))**2)
    error=abs(target_f-(1-best['loss']));assert error<1e-8
    return bound,dict(parameters=best['parameters'].tolist(),exact_fidelity=exact_f,target_fidelity=target_f,
        objective_difference=error,ideal_events=state_events(actual,model),
        initial_objective_fidelity=1-history[0]['loss'],objective_fidelity=1-best['loss'],
        history=history,evaluations=len(history),iterations=iterations,status=status,converged=converged,
        adequate_evaluation_budget=converged or len(history)>=config['minimum_evaluations_if_not_converged'],
        fitting_seconds=history[-1]['seconds'],total_seconds=seconds_before+time.monotonic()-started,
        seconds_cap=cap,verified=True,exact_output_norm=norm,planner='v2',
        recorded_evaluations_before_resume=recorded_before,recorded_seconds_before_resume=seconds_before,
        continuation_note='Interrupted v1 work not yet checkpointed is retained in the interruption archive; recorded evaluations and time consume the remaining fit budget. New checkpoints retain every completed evaluation.')


def worker(cid):
    import numpy as np
    from .native_depth_objective_v2 import make_objective
    from .native_depth_benchmark import load_target
    from .tensor_prefix_core import native_ansatz
    from .steps_native_sweep import extend_parameters
    from .full_circuit_compression import mps_dense_qiskit
    from .tensor_prefix_sabre import compile_isa_sabre
    from .depth200_native_cleanup import cleanup,cleaned_resources
    from .nighthawk_study import qpy_write
    c=settings();folder=OUT/'cases'/cid;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'complete.json').exists()
    case,reference,target=load_target(cid);model=case['model'];n=model['n_qubits']
    target_dense=mps_dense_qiskit(target,n);target_dense/=np.linalg.norm(target_dense)
    source_check=float(abs(np.vdot(reference,target_dense))**2);assert source_check>=c['minimum_target_fidelity']
    inputs=dict(case_id=cid,model=model,physical=case['physical'],edges=case['edges'],target_fidelity=source_check,
        parent_case_sha256=digest(PARENT/'cases'/cid/'complete.json'),target_mps_sha256=digest(OUT/'targets'/cid/'target.pkl'),
        reference_sha256=digest(OUT/'targets'/cid/'reference.npy'))
    if (folder/'inputs.json').exists():
        previous=read(folder/'inputs.json')
        assert all(previous[k]==v for k,v in inputs.items() if k!='target_fidelity')
        assert abs(previous['target_fidelity']-source_check)<1e-12
    else:save(folder/'inputs.json',inputs)
    backend=pickle.loads((PARENT/'preflight/target.pkl').read_bytes());candidates=parent_candidates(case);rows=[]
    for label,layers in depth_layers(case):
        objective=None;ansatz=None;contraction_path=None
        for start in range(c['product_starts'] if layers==0 else c['native_starts']):
            name=f'{cid}_{label}_r{start}';path=folder/(name+'.json')
            if path.exists():
                row=read(path);candidates.append(row);rows.append(row);continue
            if objective is None:
                ansatz=native_ansatz(n,case['edges'],layers)
                print(cid,label,'planning v2 contraction',flush=True)
                objective,diag=make_objective(target,ansatz,repeats=c['contraction_repeats'],
                    **PLANNING_LIMITS,
                    max_slices=c['max_slices'],seed=c['parameter_seed']+layers)
                index=len(list(folder.glob(label+'_contraction_v2_*.json')))
                contraction_path=folder/f'{label}_contraction_v2_{index}.json';save(contraction_path,diag)
                print(cid,label,'slices',diag['slices'],'planning seconds',diag['setup_seconds'],flush=True)
            available=[r for r in candidates if r['layers']<=layers]
            if start==2:available=[r for r in available if r['layers']==0]
            old=max(available,key=lambda r:r['exact_fidelity'])
            x=extend_parameters(old['parameters'],n,len(case['edges']),old['layers'],layers)
            perturbation=c['start_perturbations'][start]
            if perturbation:x+=np.random.default_rng(c['parameter_seed']+model['seed']+100*layers+start).normal(0,perturbation,len(x))
            initialization=dict(source_candidate=old['id'],source_layers=old['layers'],source_exact_fidelity=old['exact_fidelity'],
                perturbation=perturbation,identity_preserving_extension=perturbation==0)
            checkpoint=folder/(name+'_checkpoint.json')
            if checkpoint.exists():initialization['resumed_checkpoint_sha256']=digest(checkpoint)
            bound,row=fit_one(objective,ansatz,x,c,checkpoint,reference=reference,target_dense=target_dense,model=model)
            logical=OUT/'circuits'/(name+'.qpy');isa_path=OUT/'circuits'/(name+'_isa.qpy')
            assert not logical.exists() and not isa_path.exists()
            qpy_write(logical,bound);isa,resources=compile_isa_sabre(bound,backend,case['physical'],case['edges'])
            assert resources['swap_count']==0
            isa,certificate=cleanup(isa);resources=cleaned_resources(isa,resources,backend,certificate)
            cap=0 if layers==0 else (next(r for r in case['rows'] if r['method']=='aqc_l2')['resources']['cz_depth'] if label=='base' else int(label[1:]))
            assert resources['cz_depth']<=cap
            qpy_write(isa_path,isa)
            row.update(id=name,case_id=cid,family=model['problem'].upper(),n_qubits=n,label=label,layers=layers,start=start,
                initialization=initialization,resources=resources,cz_depth_cap=cap,
                logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=digest(logical),
                isa_qpy=str(isa_path.relative_to(ROOT)),isa_sha256=digest(isa_path),
                contraction_path=str(contraction_path.relative_to(ROOT)),contraction_sha256=digest(contraction_path),completed_utc=now())
            save(path,row);candidates.append(row);rows.append(row)
            print(name,'F',row['exact_fidelity'],'feasible',row['ideal_events']['p_feasible'],
                  'evaluations',row['evaluations'],'converged',row['converged'],flush=True)
        del objective
    save(folder/'complete.json',dict(completed_utc=now(),case_id=cid,rows=rows,all_fits_verified=all(r['verified'] for r in rows),
        adequate_fits=sum(r['adequate_evaluation_budget'] for r in rows),protocol_sha256=digest(OUT/'preparation_protocol.json'),
        amendment_sha256=digest(AMEND/'protocol.json')))


def freeze_amendment():
    assert not (OUT/'hardware_protocol.json').exists()
    test=(AMEND/'tests.log').read_text();assert '3 passed' in test and 'FAILED' not in test and 'ERROR' not in test
    benchmark=read(AMEND/'tsp25_r1_l8_benchmark.json')
    assert benchmark['plan']['slices']<=32 and max(benchmark['timings'])<60
    for cid in ['flp20_r1','vrp20_r1']:
        check=read(AMEND/f'{cid}_l8_benchmark.json')
        assert max(check['timings'])<15 and check['peak_rss_bytes']<settings()['worker_rss_limit_bytes']
    assert benchmark['peak_rss_bytes']<settings()['worker_rss_limit_bytes']
    names=['revision/native_depth_objective_v2.py','revision/native_depth_refinement_v2.py',
        'revision/native_depth_benchmark_v2.py','tests/test_native_depth_planning_v2.py']
    inputs=[AMEND/'interrupted_progress.json',AMEND/'tests.log',
            OUT/'benchmarks'/'tsp25_r1_l8_bounded.json',AMEND/'tsp25_r1_l8_benchmark.json',
            OUT/'cases'/'flp20_r1'/'d64_contraction.json',AMEND/'flp20_r1_l8_benchmark.json',AMEND/'vrp20_r1_l8_benchmark.json']
    for cid in ['flp20_r1','vrp20_r1','tsp25_r1']:
        inputs.extend(p for p in (AMEND/cid).glob('*') if p.is_file())
    record=dict(source_sha256={p:digest(ROOT/p) for p in names},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        planning_limits=PLANNING_LIMITS,
        original_protocol_sha256=digest(OUT/'preparation_protocol.json'),
        scope='Numerical execution planning and continuation only. Same targets, ansatzes, parameter starts, fitting objective and hardware design. Completed fits retained; partial fits continue within recorded remaining budgets. Per-contraction storage allowance is increased within the unchanged 9 GB worker and 35 GB aggregate RSS guards, after measured benchmarks.',
        reason='A v1 25-qubit depth-64 plan required 1024 slices. Rank candidate trees after slicing and reject plans above 32 slices before evaluation.',
        aborted_work='All three interrupted cases and their original diagnostics/checkpoints are retained. Work after the last v1 checkpoint is not recoverable as an exact evaluation count; it is not silently reported as completed optimization.',
        benchmark_sha256=digest(AMEND/'tsp25_r1_l8_benchmark.json'))
    if (AMEND/'protocol.json').exists():assert all(read(AMEND/'protocol.json')[k]==v for k,v in record.items())
    else:
        save(AMEND/'protocol.json',dict(frozen_utc=now(),new_hardware_outcomes_seen=False,**record))
        for p in names:shutil.copy2(ROOT/p,OUT/'source_snapshot'/Path(p).name)


def run():
    import subprocess
    import psutil  # Define its Popen subclass before installing the launcher adapter.
    freeze_amendment();previous=read(AMEND/'interrupted_progress.json')
    old_popen=subprocess.Popen;old_save=original.save
    def popen(command,*args,**kwargs):
        if isinstance(command,list) and 'revision.native_depth_refinement' in command and 'worker' in command:
            command=['revision.native_depth_refinement_v2' if x=='revision.native_depth_refinement' else x for x in command]
        return old_popen(command,*args,**kwargs)
    def preserving_save(path,data):
        if Path(path).name=='preparation_progress.json':
            data=dict(data,finished={**previous['finished'],**data['finished']},
                peak_total_rss_bytes=max(previous['peak_total_rss_bytes'],data['peak_total_rss_bytes']),
                elapsed_seconds=previous['elapsed_seconds']+data['elapsed_seconds'],
                numerical_amendment='contraction_planning_v2')
        elif Path(path).name=='preparation_complete.json':
            data=dict(data,peak_total_rss_bytes=max(previous['peak_total_rss_bytes'],data['peak_total_rss_bytes']),
                amendment_sha256=digest(AMEND/'protocol.json'))
        old_save(path,data)
    subprocess.Popen=popen;original.save=preserving_save
    try:original.run()
    finally:subprocess.Popen=old_popen;original.save=old_save


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','worker']);p.add_argument('case',nargs='?');a=p.parse_args()
    worker(a.case) if a.action=='worker' else run()
