"""Connected-graph state-specific compilation with an exact adjoint gradient.

No problem interactions are removed. Only the variational entangler graph is
restricted. The adjoint reconstructs forward states with inverse gates, using
O(2**n) working state memory rather than storing a state per gate.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import minimize


def edge_matchings(edges):
    """Deterministic greedy edge coloring; disjoint edges can run in parallel."""
    groups = []
    for u, v in sorted({tuple(sorted(e)) for e in edges}):
        for group in groups:
            if not any(u in e or v in e for e in group):
                group.append((u, v))
                break
        else:
            groups.append([(u, v)])
    return groups


def gate_sequence(n, edges, layers):
    ops = []
    for _ in range(layers):
        ops.extend(("ry", q) for q in range(n))
        ops.extend(("rz", q) for q in range(n))
        for group in edge_matchings(edges):
            ops.extend(("rzz", u, v) for u, v in group)
    ops.extend(("ry", q) for q in range(n))
    ops.extend(("rz", q) for q in range(n))
    return ops


def pauli_action(state, op):
    kind, q, *rest = op
    if kind == "ry":
        out = np.empty_like(state)
        src, dst = state.reshape(-1, 2, 2**q), out.reshape(-1, 2, 2**q)
        dst[:, 0, :], dst[:, 1, :] = -1j * src[:, 1, :], 1j * src[:, 0, :]
        return out
    out = state.copy()
    out.reshape(-1, 2, 2**q)[:, 1, :] *= -1
    if rest:
        out.reshape(-1, 2, 2**rest[0])[:, 1, :] *= -1
    return out


def rotate(state, op, angle):
    return np.cos(angle / 2) * state - 1j * np.sin(angle / 2) * pauli_action(state, op)


def ansatz_state(n, ops, parameters):
    state = np.full(2**n, 2**(-n / 2), dtype=complex)
    for op, angle in zip(ops, parameters):
        state = rotate(state, op, angle)
    return state


def loss_gradient(parameters, n, ops, target):
    state = ansatz_state(n, ops, parameters)
    overlap = np.vdot(target, state)
    adjoint = overlap * target
    gradient = np.empty(len(ops))
    for k in reversed(range(len(ops))):
        gradient[k] = -2 * np.real(np.vdot(adjoint, -.5j * pauli_action(state, ops[k])))
        state = rotate(state, ops[k], -parameters[k])
        adjoint = rotate(adjoint, ops[k], -parameters[k])
    return float(1 - abs(overlap)**2), gradient


def fit(n, edges, layers, target, seeds, maxiter, threshold=.999):
    ops = gate_sequence(n, edges, layers)
    start = time.perf_counter()
    choices, records = [], []
    for seed in seeds:
        initial = np.random.default_rng(seed).normal(0, .05, len(ops))
        initial_fidelity = float(abs(np.vdot(target, ansatz_state(n, ops, initial)))**2)
        result = minimize(loss_gradient, initial, args=(n, ops, target), jac=True,
            method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-7,
                                       "maxls": 25})
        choices.append(result)
        records.append(dict(seed=seed, initial_fidelity=initial_fidelity,
            fidelity=float(1-result.fun), iterations=int(result.nit), evaluations=int(result.nfev),
            success=bool(result.success), message=str(result.message)))
    best = min(choices, key=lambda result: result.fun)
    state = ansatz_state(n, ops, best.x)
    fidelity = float(abs(np.vdot(target, state))**2)
    return best.x, state, dict(fidelity=fidelity, fidelity_threshold_met=fidelity >= threshold,
        training_sec=time.perf_counter()-start, n_parameters=len(ops), layers=layers,
        edges=[[int(q) for q in e] for e in edges], starts=records,
        objective_evaluations=sum(r["evaluations"] for r in records),
        optimizer_iterations=sum(r["iterations"] for r in records),
        target_representation="exact_statevector", gradient="unitary_inverse_adjoint")


def circuit(n, edges, layers, parameters):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(range(n))
    ops = gate_sequence(n, edges, layers)
    if len(parameters) != len(ops):
        raise ValueError("Parameter count mismatch")
    for op, theta in zip(ops, parameters):
        getattr(qc, op[0])(float(theta), *op[1:])
    return qc


def select_patch(target, n):
    """Calibration-only connected patch selection, independent of sample outcomes.

    Try a greedy connected growth from every device qubit. Rank final patches by
    mean CZ error, with a modest perimeter penalty favoring compact patches.
    Return an ordered logical-to-physical map and its full induced coupling graph.
    """
    edge_error = {}
    for edge, properties in target["cz"].items():
        key = tuple(sorted(edge))
        value = properties.error if properties and properties.error is not None else .05
        edge_error[key] = float(value)
    adjacency = {q: set() for q in range(target.num_qubits)}
    for u, v in edge_error:
        adjacency[u].add(v)
        adjacency[v].add(u)
    candidates = []
    for root in adjacency:
        selected = [root]
        while len(selected) < n:
            frontier = set().union(*(adjacency[q] for q in selected)) - set(selected)
            if not frontier:
                break
            def growth_score(q):
                links = [edge_error[tuple(sorted((q, p)))] for p in selected if p in adjacency[q]]
                return (np.mean(links) - .002 * len(links), q)
            selected.append(min(frontier, key=growth_score))
        if len(selected) == n:
            induced = [e for e in edge_error if all(q in selected for q in e)]
            score = np.mean([edge_error[e] for e in induced]) - .001 * len(induced) / n
            candidates.append((score, selected, induced))
    if not candidates:
        raise ValueError("No connected patch of requested size")
    score, selected, induced = min(candidates, key=lambda item: (item[0], item[1]))
    index = {p: i for i, p in enumerate(selected)}
    edges = sorted(tuple(sorted((index[u], index[v]))) for u, v in induced)
    return selected, edges, dict(selection_score=float(score),
        mean_cz_error=float(np.mean([edge_error[e] for e in induced])),
        selected_physical_edges=[list(e) for e in induced])
