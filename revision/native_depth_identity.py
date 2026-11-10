"""Bounded diagnostic of nontrivial identity-block initialization (Grant 2019)."""
import argparse
import pickle
import numpy as np
from .native_depth_refinement import ROOT,OUT,PARENT,settings
from .contribution_hardware_core import read,save,digest,now


def identity_extension(old,n,edge_count,previous_layers,layers,seed):
    from .steps_native_sweep import extend_parameters
    x=extend_parameters(old,n,edge_count,previous_layers,layers)
    added=layers-previous_layers
    assert added>0 and added%2==0
    half=added//2;stride=2*n+edge_count;rng=np.random.default_rng(seed)
    for i in range(half):
        angles=rng.uniform(np.pi/4,3*np.pi/4,edge_count)*rng.choice([-1,1],edge_count)
        left=(previous_layers+i)*stride+2*n
        right=(previous_layers+added-1-i)*stride+2*n
        x[left:left+edge_count]=angles;x[right:right+edge_count]=-angles
    for k in range(1,half):
        angles=rng.uniform(-np.pi,np.pi,n)
        left=(previous_layers+k)*stride
        right=(previous_layers+added-k)*stride
        x[left:left+2*n:2]=angles;x[right:right+2*n:2]=-angles
    return x


def run(cid='tsp16_r1',seed=11791):
    from .native_depth_benchmark import load_target
    from .native_depth_objective_v2 import make_objective
    from .native_depth_refinement_v2 import fit_one,PLANNING_LIMITS
    from .tensor_prefix_core import native_ansatz
    from .hardware_width_sweep import exact_state
    from .full_circuit_compression import mps_dense_qiskit
    from .tensor_prefix_sabre import compile_isa_sabre
    from .depth200_native_cleanup import cleanup,cleaned_resources
    from .nighthawk_study import qpy_write
    folder=OUT/'identity_block_diagnostic'/cid;folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'result.json').exists()
    case,reference,target=load_target(cid);model=case['model'];n=model['n_qubits'];assert n<=16
    completed=read(OUT/'cases'/cid/'complete.json')
    base=max([r for r in completed['rows'] if r['layers']==2],key=lambda r:r['exact_fidelity'])
    layers=8;ansatz=native_ansatz(n,case['edges'],layers)
    x=identity_extension(base['parameters'],n,len(case['edges']),2,layers,seed)
    initial=exact_state(ansatz.assign_parameters(x))
    old=exact_state(native_ansatz(n,case['edges'],2).assign_parameters(base['parameters']))
    identity_fidelity=float(abs(np.vdot(old,initial))**2);assert identity_fidelity>1-1e-10
    save(folder/'protocol.json',dict(frozen_utc=now(),case_id=cid,seed=seed,layers=layers,
        initial_candidate=base['id'],initial_source_fidelity=base['exact_fidelity'],identity_fidelity=identity_fidelity,
        reference='Grant et al., Quantum 3, 214 (2019), https://doi.org/10.22331/q-2019-12-09-214',
        scope='One known-instance initialization diagnostic; no new hardware and no general improvement claim.',
        source_sha256=digest(__file__),settings=settings()))
    objective,diagnostic=make_objective(target,ansatz,**PLANNING_LIMITS,seed=101901)
    save(folder/'contraction.json',diagnostic)
    dense=mps_dense_qiskit(target,n);dense/=np.linalg.norm(dense)
    bound,row=fit_one(objective,ansatz,x,settings(),folder/'checkpoint.json',
        reference=reference,target_dense=dense,model=model)
    backend=pickle.loads((PARENT/'preflight/target.pkl').read_bytes())
    isa,resources=compile_isa_sabre(bound,backend,case['physical'],case['edges'])
    isa,certificate=cleanup(isa);resources=cleaned_resources(isa,resources,backend,certificate)
    assert resources['swap_count']==0 and resources['cz_depth']<=64
    logical=folder/'logical.qpy';physical=folder/'isa.qpy';qpy_write(logical,bound);qpy_write(physical,isa)
    row.update(id=cid+'_identity_block',case_id=cid,layers=layers,initial_candidate=base['id'],
        initial_identity_fidelity=identity_fidelity,resources=resources,
        logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=digest(logical),
        isa_qpy=str(physical.relative_to(ROOT)),isa_sha256=digest(physical),completed_utc=now())
    save(folder/'result.json',row)
    print('Identity-block diagnostic:',row['exact_fidelity'],row['ideal_events'],row['evaluations'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case',nargs='?',default='tsp16_r1');a=p.parse_args();run(a.case)
