"""Bounded AQC parameter adaptation at fixed architecture for changed costs."""
import argparse
import importlib.metadata
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .contribution_hardware_core import ROOT, read, save, digest, now, connection_config

CONFIG = ROOT/'configs/cost_adaptation_2026_10.json'
OUT = ROOT/'outputs/cost_adaptation_2026_10'
TRAIN = ROOT/'outputs/native_depth_refinement_2026_10'
MODELS = ROOT/'outputs/hardware_width_sweep_2026'
EXTRA = ""


def settings():
    return read(CONFIG)


def training(cid):
    case = read(MODELS/'cases'/cid/'complete.json')
    rows = read(TRAIN/'cases'/cid/'complete.json')['rows']
    eligible = [r for r in rows if r['layers'] == settings()['layers']]
    selected = min(eligible, key=lambda r: (-r['exact_fidelity'], r['id']))
    assert len(eligible) == 3
    return case, selected


def preflight():
    from .nighthawk_study import connect
    c = settings()
    _, backend = connect(dict(connection_config(), backend=c['backend']))
    status = backend.status()
    processor = backend.configuration().processor_type
    assert status.operational and processor == {'family': c['processor_family_required'], 'revision': c['processor_revision_required']}
    folder = OUT/'preflight'; folder.mkdir(parents=True, exist_ok=True)
    assert not (OUT/'protocol.json').exists()
    target = backend.target
    edges = {tuple(sorted(e)) for e in target['cz']}
    for cid in {s['training_case'] for s in c['cases']}:
        case, _ = training(cid)
        assert all(tuple(sorted((case['physical'][u], case['physical'][v]))) in edges for u, v in case['edges'])
    props = backend.properties(refresh=True).to_dict()
    save(folder/'device.json', dict(observed_utc=now(), backend=backend.name,
        processor=processor, operational=status.operational, pending_jobs=status.pending_jobs,
        calibration_timestamp=str(props['last_update_date'])))
    save(folder/'properties.json', props)
    (folder/'target.pkl').write_bytes(pickle.dumps(target))
    print(read(folder/'device.json'), flush=True)


def freeze():
    c = settings()
    assert len(c['cases']) == 6
    names = ['configs/cost_adaptation_2026_10.json', 'revision/cost_adaptation_study.py',
             'revision/native_depth_refinement_v2.py', 'revision/native_depth_objective_v2.py',
             'revision/native_depth_objective.py', 'revision/hardware_width_sweep.py',
             'revision/width_sweep_models.py', 'revision/tensor_prefix_models.py',
             'revision/tensor_prefix_core.py', 'revision/nighthawk_study.py',
             'revision/tensor_prefix_sabre.py', 'revision/depth200_native_cleanup.py',
             'tests/test_cost_adaptation.py']
    inputs = [OUT/'preflight'/n for n in ['device.json', 'properties.json', 'target.pkl']]
    inputs.append(OUT/'preparation_tests.log')
    assert 'passed' in inputs[-1].read_text()
    for cid in {s['training_case'] for s in c['cases']}:
        inputs.extend([MODELS/'cases'/cid/'complete.json', TRAIN/'cases'/cid/'complete.json'])
    record = dict(config=c, source_sha256={n:digest(ROOT/n) for n in names},
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in inputs},
        packages={n:importlib.metadata.version(n) for n in ['numpy','scipy','quimb','torch','qiskit','qiskit-aer','qiskit-addon-aqc-tensor']})
    path = OUT/'protocol.json'
    if path.exists():
        assert all(read(path)[k] == v for k,v in record.items())
    else:
        for name in names:
            dst=OUT/'source_snapshot'/Path(name).name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name,dst)
        save(path,dict(frozen_utc=now(),new_cost_fit_outcomes_seen=False,**record))


def check_frozen():
    protocol=read(OUT/'protocol.json')
    for name,value in {**protocol['source_sha256'],**protocol['input_sha256']}.items():
        assert digest(ROOT/name)==value,name


def worker(cid):
    import numpy as np
    from .width_sweep_models import make_model, validate_model, feasible_costs
    from .tensor_prefix_models import numpy_model
    from .tensor_prefix_core import native_ansatz
    from .hardware_width_sweep import exact_state, dense_target, state_events
    from .full_circuit_compression import mps_dense_qiskit
    from .native_depth_objective_v2 import make_objective
    from .native_depth_refinement_v2 import fit_one
    from .nighthawk_study import evolution, qpy_write
    from .tensor_prefix_sabre import compile_isa_sabre
    from .depth200_native_cleanup import cleanup, cleaned_resources
    check_frozen(); c=settings(); folder=OUT/'cases'/cid; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return
    spec=next(s for s in c['cases'] if s['id']==cid)
    parent,old=training(spec['training_case'])
    model=make_model({k:v for k,v in spec.items() if k!='training_case'})
    assert set(feasible_costs(model))==set(feasible_costs(parent['model']))
    save(folder/'model.json',model)
    save(folder/'model_validation.json',validate_model(model,exhaustive=False))
    n=model['n_qubits']; ansatz=native_ansatz(n,parent['edges'],c['layers'])
    source=evolution(numpy_model(model),c['source_duration'],c['source_steps'])
    reference=exact_state(source)
    old_reference=exact_state(evolution(numpy_model(parent['model']),c['source_duration'],c['source_steps']))
    source_overlap=float(abs(np.vdot(reference,old_reference))**2); del old_reference
    target=dense_target(reference,n,c['target_bond']); target._psi/=np.sqrt(float(abs(target.psi.H@target.psi)))
    target_dense=mps_dense_qiskit(target,n); target_dense/=np.linalg.norm(target_dense)
    target_fidelity=float(abs(np.vdot(reference,target_dense))**2)
    assert target_fidelity >= c['minimum_target_fidelity']
    save(folder/'target_validation.json',dict(target_exact_fidelity=target_fidelity,
        source_overlap_with_training=source_overlap,ideal_source_events=state_events(reference,model)))
    objective,contraction=make_objective(target,ansatz,repeats=c['contraction_repeats'],
        max_tensor_bytes=c['max_tensor_bytes'],max_slice_write_bytes=c['max_slice_write_bytes'],
        max_slices=c['max_slices'],seed=c['parameter_seed'],
        warm_path_file=TRAIN/'cases'/spec['training_case']/'base_contraction.json')
    save(folder/'contraction.json',contraction)
    old_parameters=np.array(old['parameters'])
    assert len(old_parameters)==ansatz.num_parameters
    reuse=ansatz.assign_parameters(old_parameters)
    reuse_state=exact_state(reuse)
    reuse_row=dict(method='reuse',parameters=old_parameters.tolist(),
        exact_fidelity=float(abs(np.vdot(reference,reuse_state))**2),
        ideal_events=state_events(reuse_state,model),evaluations=0,fitting_seconds=0.)
    del reuse_state
    candidates=[]
    for method in ['warm','cold']:
        for start,perturbation in enumerate(c[method+'_start_perturbations']):
            path=folder/f'{method}_{start}.json'
            if path.exists(): candidates.append(read(path)); continue
            x0=old_parameters.copy() if method=='warm' else np.zeros(ansatz.num_parameters)
            rng=np.random.default_rng(c['parameter_seed']+spec['seed']+start)
            x0+=rng.normal(0,perturbation,len(x0))
            _,fit=fit_one(objective,ansatz,x0,c,folder/f'{method}_{start}_checkpoint.json',
                reference=reference,target_dense=target_dense,model=model)
            assert fit['evaluations'] <= c['fit_max_evaluations']
            assert fit['adequate_evaluation_budget']
            row=dict(method=method,start=start,initialization=method,perturbation=perturbation,**fit)
            save(path,row); candidates.append(row)
            print(cid,method,start,'eval',fit['evaluations'],'F',fit['exact_fidelity'],flush=True)
    chosen=[reuse_row]+[min([r for r in candidates if r['method']==method],
        key=lambda r:(-r['exact_fidelity'],r['start'])) for method in ['warm','cold']]
    backend=pickle.loads((OUT/'preflight/target.pkl').read_bytes())
    rows=[]
    for row in chosen:
        bound=ansatz.assign_parameters(row['parameters']); name=cid+'_'+row['method']
        logical=OUT/'circuits'/f'{name}.qpy'; isa_path=OUT/'circuits'/f'{name}_isa.qpy'
        qpy_write(logical,bound)
        isa,resources=compile_isa_sabre(bound,backend,parent['physical'],parent['edges'])
        assert resources['swap_count']==0
        isa,certificate=cleanup(isa); resources=cleaned_resources(isa,resources,backend,certificate)
        qpy_write(isa_path,isa)
        rows.append(dict(case_id=cid,family=model['problem'].upper(),n_qubits=n,method=row['method'],
            candidate_id=name,exact_fidelity=row['exact_fidelity'],ideal_events=row['ideal_events'],
            selected_start=row.get('start'),parameters=row['parameters'],resources=resources,
            logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=digest(logical),
            isa_qpy=str(isa_path.relative_to(ROOT)),isa_sha256=digest(isa_path)))
    save(folder/'complete.json',dict(completed_utc=now(),case_id=cid,model=model,
        training_case=spec['training_case'],training_candidate=old['id'],training_fitting_seconds=old['fitting_seconds'],
        training_evaluations=old['evaluations'],target_validation=read(folder/'target_validation.json'),
        rows=rows,fits=candidates,protocol_sha256=digest(OUT/'protocol.json')))


def run():
    from .cpu_memory_guard import physical_cpu_ids
    freeze(); c=settings(); cpus=physical_cpu_ids(); size=c['physical_cores_per_worker']
    assert len(cpus)>=c['parallel_workers']*size
    tasks=[s['id'] for s in c['cases'] if not (OUT/'cases'/s['id']/'complete.json').exists()]
    active={}
    while tasks or active:
        for slot in range(c['parallel_workers']):
            if slot in active or not tasks:continue
            cid=tasks.pop(0);cores=cpus[slot*size:(slot+1)*size]
            path=OUT/'logs'/f'{cid}.log';path.parent.mkdir(exist_ok=True)
            log=path.open('a')
            cmd=[sys.executable,'-m','revision.cpu_memory_guard','--address-gib',str(c['worker_address_gib']),
                 '--cpus',','.join(map(str,cores)),'--threads',str(size),'--',sys.executable,'-m',
                 'revision.local_runner','--extra-site',EXTRA,'revision.cost_adaptation_study','worker',cid]
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            active[slot]=(child,log,cid,time.monotonic())
            print('Started',cid,'PID',child.pid,'CPUs',cores,flush=True)
        for slot,(child,log,cid,started) in list(active.items()):
            if child.poll() is None:continue
            log.close()
            save(OUT/'logs'/f'{cid}_process.json',dict(returncode=child.returncode,seconds=time.monotonic()-started))
            if child.returncode:raise RuntimeError(f'{cid} failed; inspect its log')
            assert (OUT/'cases'/cid/'complete.json').exists()
            del active[slot]
        if active:time.sleep(1)
    save(OUT/'preparation_complete.json',dict(completed_utc=now(),cases=6,fits=36,
        protocol_sha256=digest(OUT/'protocol.json')))


def guarded():
    from .cpu_memory_guard import guarded_run
    command=[sys.executable,'-m','revision.local_runner','--extra-site',EXTRA,
             'revision.cost_adaptation_study','run']
    result=guarded_run(command,OUT/'logs'/'preparation.log',cores=12,threads=4,
        address_gib=32,rss_gib=26,reserve_gib=8,seconds=2400)
    save(OUT/'preparation_resources.json',result)
    print(result,flush=True)
    if result['status']!='completed':raise RuntimeError(result['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','run','worker','guarded']);parser.add_argument('case_id',nargs='?')
    args=parser.parse_args()
    if args.action=='worker':worker(args.case_id)
    else:globals()[args.action]()
