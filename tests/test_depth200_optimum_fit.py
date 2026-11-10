import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from revision.depth200_optimum_fit import euler_extend, projected_qaoa_initialization
from revision.placement_study_core import native_euler_ansatz
from revision.tensor_prefix_models import make_model


def test_euler_growth_preserves_state_including_final_rotations():
    n=5;edges=[(0,1),(1,2),(2,3),(3,4),(0,3)]
    a=native_euler_ansatz(n,edges,2);x=np.random.default_rng(42).normal(size=a.num_parameters)
    b=native_euler_ansatz(n,edges,6)
    y=euler_extend(x,n,len(edges),2,6)
    assert state_fidelity(Statevector.from_instruction(a.assign_parameters(x)),
                          Statevector.from_instruction(b.assign_parameters(y)))>1-1e-12


def test_native_qaoa_initializer_matches_independent_gates():
    model=make_model(dict(problem='flp',facilities=2,customers=1,seed=201));n=model['n_qubits']
    edges=[(0,1),(1,2),(2,3)];angles=np.array([[.2,-.3],[.4,-.25]])
    q=np.asarray(model['quadratic']);h=-np.asarray(model['linear'])/2-(q.sum(0)+q.sum(1))/4
    j=(q+q.T)/4
    qc=QuantumCircuit(n);qc.h(range(n))
    for gamma,beta in angles:
        for u in range(n):qc.rz(2*gamma*h[u],u)
        for u,v in edges:qc.rzz(2*gamma*j[u,v],u,v)
        qc.rx(2*beta,range(n))
    x=projected_qaoa_initialization(model,edges,angles,6)
    a=native_euler_ansatz(n,edges,6).assign_parameters(x)
    assert state_fidelity(Statevector.from_instruction(a),Statevector.from_instruction(qc))>1-1e-12
