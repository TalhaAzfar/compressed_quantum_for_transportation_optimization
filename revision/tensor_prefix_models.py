"""Sparse FLP/TSP models and direct decoding, without 2**n state enumeration."""
from __future__ import annotations

import itertools
import numpy as np


def make_model(spec):
    problem, seed = spec["problem"], spec["seed"]
    rng = np.random.default_rng(seed)
    if problem == "flp":
        c, f = spec["customers"], spec["facilities"]
        n = f*(c+1)
        service = rng.integers(1, 21, (c, f)).astype(float)
        opening = rng.integers(10, 31, f).astype(float)
        penalty = 2*(service.sum()+opening.sum())
        linear, quadratic = np.zeros(n), np.zeros((n, n))
        linear[:f] = opening
        for i in range(c):
            for j in range(f):
                u = f+i*f+j
                linear[u] = service[i, j]
                quadratic[j, u] = -penalty
                for k in range(j+1, f):
                    quadratic[u, f+i*f+k] = 2*penalty
        costs = []
        for mask in range(1, 2**f):
            opened = [j for j in range(f) if mask >> j & 1]
            costs.append(float(opening[opened].sum()+service[:, opened].min(axis=1).sum()))
        extra = dict(customers=c, facilities=f, service=service.tolist(), opening=opening.tolist(),
                     optimum=min(costs), classical_reference_evaluations=len(costs), constant=c*penalty)
    elif problem == "tsp":
        nodes = spec["nodes"]
        m, n = nodes-1, (nodes-1)**2
        coordinates = rng.uniform(0, 10, (nodes, 2))
        distance = np.linalg.norm(coordinates[:, None]-coordinates[None, :], axis=2)
        penalty = 2*distance.sum()
        linear, quadratic = np.zeros(n), np.zeros((n, n))
        for t in range(m):
            for i in range(m):
                u = t*m+i
                linear[u] = -2*penalty
                if t == 0:
                    linear[u] += distance[0, i+1]
                if t == m-1:
                    linear[u] += distance[i+1, 0]
                for j in range(i+1, m):
                    quadratic[u, t*m+j] += 2*penalty
                for s in range(t+1, m):
                    quadratic[u, s*m+i] += 2*penalty
                if t < m-1:
                    for j in range(m):
                        quadratic[u, (t+1)*m+j] += distance[i+1, j+1]
        tours = list(itertools.permutations(range(1, nodes)))
        optimum = min(sum(distance[a, b] for a, b in zip((0,)+tour, tour+(0,))) for tour in tours)
        extra = dict(nodes=nodes, distance=distance.tolist(), coordinates=coordinates.tolist(),
                     optimum=float(optimum), classical_reference_evaluations=len(tours), constant=2*m*penalty)
    else:
        raise ValueError(problem)
    scale = max(1., float(abs(linear).max()), float(abs(quadratic).max()))
    return dict(spec=spec, problem=problem, seed=seed, n_qubits=n,
                instance_id=spec.get("id", f"{problem}_{n}q_seed{seed}"),
                penalty=float(penalty), scale=scale, linear=(linear/scale).tolist(),
                quadratic=(quadratic/scale).tolist(), **extra)


def numpy_model(model):
    return {**model, "linear": np.array(model["linear"]), "quadratic": np.array(model["quadratic"])}


def decode(model, values):
    values = np.asarray(values, dtype=np.uint64)
    bits = ((values[:, None] >> np.arange(model["n_qubits"], dtype=np.uint64)) & 1).astype(float)
    if model["problem"] == "flp":
        c, f = model["customers"], model["facilities"]
        y, x = bits[:, :f], bits[:, f:].reshape(-1, c, f)
        costs = y@np.array(model["opening"])+(x*np.array(model["service"])).sum(axis=(1, 2))
        violations = ((x.sum(axis=2)-1)**2).sum(axis=1)+(x*(1-y[:, None, :])).sum(axis=(1, 2))
    else:
        m = model["nodes"]-1
        x = bits.reshape(-1, m, m)
        distance = np.array(model["distance"])
        costs = x[:, 0, :]@distance[0, 1:]+x[:, -1, :]@distance[1:, 0]
        for t in range(m-1):
            costs += np.einsum("bi,ij,bj->b", x[:, t, :], distance[1:, 1:], x[:, t+1, :])
        violations = ((x.sum(axis=1)-1)**2).sum(axis=1)+((x.sum(axis=2)-1)**2).sum(axis=1)
    feasible = violations == 0
    return dict(cost=costs, violations=violations, feasible=feasible,
                good=feasible & (costs <= 1.05*model["optimum"]+1e-9),
                optimal=feasible & np.isclose(costs, model["optimum"], atol=1e-9, rtol=0),
                energy=(costs+model["penalty"]*violations)/model["scale"])


def count_metrics(model, counts, alpha=.2):
    keys = list(counts)
    values = [int(k.replace(" ", ""), 2) if isinstance(k, str) else int(k) for k in keys]
    weights = np.array([counts[k] for k in keys], dtype=float)
    assert np.all(weights >= 0) and weights.sum() > 0
    decoded = decode(model, values)
    order = np.argsort(decoded["energy"])
    remaining, weighted = alpha*weights.sum(), 0.
    for k in order:
        take = min(remaining, weights[k])
        weighted += take*decoded["energy"][k]
        remaining -= take
        if remaining <= 1e-12:
            break
    shots = int(weights.sum())
    result = dict(shots=shots, cvar=float(weighted/(alpha*shots)),
                  mean_energy=float(np.dot(weights, decoded["energy"])/shots))
    for event in ["feasible", "good", "optimal"]:
        count = int(np.dot(weights, decoded[event]))
        result[event+"_count"] = count
        result["p_"+event] = count/shots
    result["best_feasible_cost"] = float(decoded["cost"][decoded["feasible"]].min()) if decoded["feasible"].any() else None
    return result
