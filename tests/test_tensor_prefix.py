import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from revision.corrected_study import ising_operator
from revision.tensor_prefix_models import make_model, numpy_model, decode, count_metrics
from revision.tensor_prefix_core import native_ansatz, target_circuit, mps_state, fidelity, qaoa, simulator


@pytest.mark.parametrize("spec", [dict(problem="flp", customers=2, facilities=2, seed=81201),
                                  dict(problem="tsp", nodes=4, seed=81202)])
def test_qubo_and_independent_decoding(spec):
    model = make_model(spec)
    n = model["n_qubits"]
    values = np.arange(2**n, dtype=np.uint64)
    events = decode(model, values)
    bits = ((values[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
    Q, linear = np.array(model["quadratic"]), np.array(model["linear"])
    qubo = bits@linear+np.einsum("bi,ij,bj->b", bits, Q, bits)+model["constant"]/model["scale"]
    assert qubo == pytest.approx(events["energy"])
    diagonal = ising_operator(numpy_model(model)).to_matrix().diagonal().real
    assert (diagonal-diagonal[0]) == pytest.approx(qubo-qubo[0])
    assert events["cost"][events["feasible"]].min() == pytest.approx(model["optimum"])
    assert events["optimal"][np.argmin(qubo)]


def test_aqc_tensor_mps_gradient_and_normalization():
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    model = make_model(dict(problem="flp", customers=2, facilities=2, seed=81203))
    target_qc = target_circuit(model, .5, steps=3)
    # Exact-state comparison needs an untruncated MPS, including during fitting.
    from functools import partial
    import quimb.tensor as qtn
    from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    settings = QuimbSimulator(partial(qtn.CircuitMPS, max_bond=32, cutoff=0.),
                              autodiff_backend="jax")
    target = tensornetwork_from_circuit(target_qc, settings)
    target._psi /= np.sqrt(abs(target.psi.H@target.psi))
    ansatz = native_ansatz(6, [(i,i+1) for i in range(5)], 1)
    objective = MaximizeStateFidelity(target, ansatz, settings)
    theta = np.random.default_rng(81).normal(0, .2, ansatz.num_parameters)
    loss, grad = objective.loss_function(theta)
    expected = Statevector.from_instruction(target_qc).data
    actual = Statevector.from_instruction(ansatz.assign_parameters(theta)).data
    assert 1-loss == pytest.approx(abs(np.vdot(expected, actual))**2, abs=1e-7)
    for k in [0, 7, len(theta)-1]:
        shift = np.zeros_like(theta); shift[k] = 1e-5
        fd = (objective.loss_function(theta+shift)[0]-objective.loss_function(theta-shift)[0])/2e-5
        assert grad[k] == pytest.approx(fd, abs=1e-6)


def test_qaoa_prefix_input_and_common_continuation_fidelity():
    model = make_model(dict(problem="flp", customers=2, facilities=2, seed=81204))
    prep = target_circuit(model, .5, steps=2)
    a = Statevector.from_instruction(prep).data
    assert abs(np.vdot(a, Statevector.from_instruction(qaoa(prep, model, [0.,0.])).data))**2 == pytest.approx(1.)
    simple = QuantumCircuit(6); simple.h(range(6))
    b = Statevector.from_instruction(simple).data
    qa = Statevector.from_instruction(qaoa(prep, model, [.3,-.2])).data
    qb = Statevector.from_instruction(qaoa(simple, model, [.3,-.2])).data
    assert abs(np.vdot(qa,qb))**2 == pytest.approx(abs(np.vdot(a,b))**2)


def test_large_models_do_not_allocate_full_bit_space_and_counts_decode():
    for spec in [dict(problem="flp", customers=5, facilities=5, seed=81205),dict(problem="tsp",nodes=6,seed=81206)]:
        model = make_model(spec)
        assert model["n_qubits"] in [25,30]
        assert len(model["linear"]) == model["n_qubits"]
        counts = {"0"*model["n_qubits"]:5,"1"*model["n_qubits"]:3}
        result = count_metrics(model, counts)
        assert result["shots"] == 8 and result["feasible_count"] == 0
        assert result["cvar"] <= result["mean_energy"]


def test_real_target_native_edges_and_qaoa_measurement_mapping():
    import pickle
    from revision.corrected_study import ROOT
    from revision.tensor_prefix_core import path_patch, compile_isa
    target=pickle.loads((ROOT/"outputs/aqc_tensor_qaoa_v1/preflight/target_snapshot.pkl").read_bytes())
    model=make_model(dict(problem="flp",customers=2,facilities=2,seed=81207))
    physical,edges,_=path_patch(target,6)
    ansatz=native_ansatz(6,[(i,i+1) for i in range(5)],1)
    theta=np.random.default_rng(81).normal(0,.1,ansatz.num_parameters)
    qc=qaoa(ansatz.assign_parameters(theta),model,[.2,-.2])
    isa,record=compile_isa(qc,target,physical,edges)
    assert record["transpilation_fidelity"]==pytest.approx(1.,abs=1e-8)
    assert isa.num_clbits==6 and record["cz_count"]>0
