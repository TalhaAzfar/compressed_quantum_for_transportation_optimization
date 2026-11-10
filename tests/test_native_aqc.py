import json
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

from revision import native_aqc
from revision.corrected_study import anneal
from revision.nighthawk_study import instance, evolution


def config():
    return json.loads(Path("configs/nighthawk_validation.json").read_text())


def test_adjoint_gradient_matches_central_difference():
    n, edges = 3, [(0, 1), (1, 2)]
    ops = native_aqc.gate_sequence(n, edges, 2)
    rng = np.random.default_rng(123)
    target = rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    target /= np.linalg.norm(target)
    params = rng.normal(size=len(ops))
    loss, gradient = native_aqc.loss_gradient(params, n, ops, target)
    numerical = []
    for k in range(len(params)):
        d = np.zeros_like(params)
        d[k] = 1e-6
        numerical.append((native_aqc.loss_gradient(params+d, n, ops, target)[0]
                          -native_aqc.loss_gradient(params-d, n, ops, target)[0])/2e-6)
    assert np.allclose(gradient, numerical, atol=2e-9)


def test_native_numpy_matches_qiskit_and_has_only_allowed_entanglers():
    n, edges, layers = 4, [(0, 1), (1, 3), (2, 3)], 2
    ops = native_aqc.gate_sequence(n, edges, layers)
    params = np.random.default_rng(456).normal(size=len(ops))
    qc = native_aqc.circuit(n, edges, layers, params)
    assert np.allclose(native_aqc.ansatz_state(n, ops, params), Statevector.from_instruction(qc).data)
    for instruction in qc.data:
        if instruction.operation.num_qubits == 2:
            pair = tuple(sorted(qc.find_bit(q).index for q in instruction.qubits))
            assert pair in edges


def test_exact_problem_enumeration_matches_qubo():
    ins = instance(2, 3, 712, config())
    n = ins["n_qubits"]
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    direct = bits @ ins["linear"] + np.einsum("bi,ij,bj->b", bits, ins["quadratic"], bits)
    assert np.allclose(direct, ins["energy"])
    assert np.all(ins["optimal"][np.isclose(direct, direct.min())])


def test_evolution_and_continuation_match_independent_simulator():
    ins = instance(2, 2, 710, config())
    prefix = anneal(ins, 2., 16, end_fraction=.5)
    state = anneal(ins, 2., 8, state=prefix, start_fraction=.5)
    qc = evolution(ins, 2., 16, end=.5).compose(evolution(ins, 2., 8, start=.5, prepare=False))
    assert np.isclose(abs(np.vdot(state, Statevector.from_instruction(qc).data))**2, 1., atol=1e-10)


def test_fit_recovers_representable_state():
    n, edges, layers = 3, [(0, 1), (1, 2)], 2
    ops = native_aqc.gate_sequence(n, edges, layers)
    params = np.random.default_rng(7).normal(0, .12, len(ops))
    target = native_aqc.ansatz_state(n, ops, params)
    _, state, record = native_aqc.fit(n, edges, layers, target, [8], 100)
    assert record["fidelity"] > .99999
    assert np.isclose(np.linalg.norm(state), 1.)
