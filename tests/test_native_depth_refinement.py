import numpy as np


def test_identity_extension_preserves_arbitrary_preparation():
    from qiskit.quantum_info import Statevector
    from revision.tensor_prefix_core import native_ansatz
    from revision.steps_native_sweep import extend_parameters
    edges = [(0,1),(1,2),(2,3),(0,3)]
    old = native_ansatz(4, edges, 2); new = native_ansatz(4, edges, 8)
    x = np.random.default_rng(17001).normal(0,.4,old.num_parameters)
    y = extend_parameters(x,4,len(edges),2,8)
    a = Statevector(old.assign_parameters(x)).data
    b = Statevector(new.assign_parameters(y)).data
    assert abs(np.vdot(a,b))**2 > 1-1e-12


def test_sliced_and_unsliced_aqc_gradient_against_exact_finite_difference():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from revision.hardware_width_sweep import dense_target
    from revision.tensor_prefix_core import native_ansatz
    from revision.native_depth_objective import make_objective
    qc = QuantumCircuit(5); qc.h(range(5)); qc.rx(.52,0); qc.ry(-.73,2)
    qc.rzz(.37,0,4); qc.cx(2,3); qc.rz(.44,1); qc.rxx(.23,1,4)
    reference = Statevector(qc).data
    target = dense_target(reference,5,32)
    ansatz = native_ansatz(5,[(0,1),(1,2),(2,3),(3,4),(0,4)],3)
    x = np.random.default_rng(17002).normal(0,.3,ansatz.num_parameters)
    exact = lambda a:1-abs(np.vdot(reference,Statevector(ansatz.assign_parameters(a)).data))**2
    finite = []
    for i in range(len(x)):
        dx = np.zeros_like(x); dx[i] = 1e-6
        finite.append((exact(x+dx)-exact(x-dx))/2e-6)
    for slices in [1,2]:
        obj, check = make_objective(target,ansatz,repeats=2,max_tensor_bytes=10**9,
                                   max_slice_write_bytes=10**9,max_slices=4096,minimum_slices=slices)
        value, gradient = obj.loss_function(x)
        assert abs(float(value)-exact(x)) < 1e-10
        assert np.max(np.abs(np.asarray(gradient)-finite)) < 2e-6
        if slices == 2: assert check['slices'] > 1
