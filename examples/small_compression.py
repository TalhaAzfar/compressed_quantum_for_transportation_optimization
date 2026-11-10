"""Small offline AQC-Tensor example. It never connects to a quantum service."""
import argparse
import json
import os
from pathlib import Path
import sys

for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(name,'1')
os.environ.setdefault('QUIMB_NUMBA_CACHE','FALSE')
os.environ.setdefault('JAX_ENABLE_X64','True')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import minimize
from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from revision.width_sweep_models import make_model,decode
from revision.tensor_prefix_core import native_ansatz,target_circuit,mps_state,simulator

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--problem',choices=['flp','tsp','vrp'],default='flp')
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--maxiter',type=int,default=25)
    args=p.parse_args()
    specs={'flp':dict(problem='flp',customers=2,facilities=2,seed=81201),
           'tsp':dict(problem='tsp',nodes=4,seed=81202),
           'vrp':dict(problem='vrp',nodes=4,vehicles=2,seed=81203)}
    model=make_model(specs[args.problem]);n=model['n_qubits']
    # A connected chain is a toy coupling graph, not a current IBM calibration.
    edges=[(i,i+1) for i in range(n-1)]
    source=target_circuit(model,1.,steps=12,duration=2.)
    bond=2**(n//2)
    target=mps_state(source,bond)
    target._psi/=np.sqrt(float(abs(target.psi.H@target.psi)))
    ansatz=native_ansatz(n,edges,args.layers)
    objective=MaximizeStateFidelity(target,ansatz,simulator(bond))
    x=np.random.default_rng(81002).normal(0,.05,ansatz.num_parameters)
    result=minimize(objective.loss_function,x,jac=True,method='L-BFGS-B',options={'maxiter':args.maxiter})
    compiled=ansatz.assign_parameters(result.x)
    a=Statevector(source).data;b=Statevector(compiled).data
    feasible=decode(model,np.arange(2**n,dtype=np.uint64))['feasible']
    def depth(qc):
        isa=transpile(qc,basis_gates=['rz','sx','x','cz'],coupling_map=[list(e) for e in edges]+[list(reversed(e)) for e in edges],optimization_level=1,seed_transpiler=81003)
        return isa.depth(lambda op:op.operation.num_qubits==2)
    report=dict(problem=args.problem,n_qubits=n,source_steps=12,source_duration=2.,
                layers=args.layers,iterations=int(result.nit),fidelity=float(abs(np.vdot(a,b))**2),
                ideal_source_p_feasible=float(np.sum(abs(a[feasible])**2)),
                ideal_compiled_p_feasible=float(np.sum(abs(b[feasible])**2)),
                source_cz_depth=depth(source),compiled_cz_depth=depth(compiled),
                hardware_jobs_submitted=0,scope='Small correctness example. These are ideal probabilities, not hardware results.')
    if abs(report['fidelity']-(1-float(result.fun)))>1e-6:raise ValueError('MPS/exact fidelity mismatch')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
