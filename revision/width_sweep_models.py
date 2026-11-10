"""Independently enumerable FLP, fixed-depot TSP and bounded two-vehicle VRP."""
import itertools
import numpy as np
from .tensor_prefix_models import make_model as base_model, decode as base_decode


def route_cost(route, distance):
    return sum(distance[a][b] for a, b in zip([0]+list(route), list(route)+[0]))


def feasible_costs(model):
    result = {}
    if model['problem'] == 'flp':
        c, f = model['customers'], model['facilities']
        for mask in range(1, 1 << f):
            opened = [j for j in range(f) if mask >> j & 1]
            for assignment in itertools.product(opened, repeat=c):
                value = mask + sum(1 << (f+i*f+j) for i, j in enumerate(assignment))
                result[value] = sum(model['opening'][j] for j in opened) + sum(model['service'][i][j] for i, j in enumerate(assignment))
    elif model['problem'] == 'tsp':
        m = model['nodes']-1
        for tour in itertools.permutations(range(1, m+1)):
            result[sum(1 << (t*m+city-1) for t, city in enumerate(tour))] = route_cost(tour, model['distance'])
    else:
        nodes = model['nodes']; arcs = [tuple(a) for a in model['variables']]
        for order in itertools.permutations(range(1, nodes)):
            for cut in range(1, nodes-1):
                routes = [order[:cut], order[cut:]]
                selected = {a for route in routes for a in zip([0]+list(route), list(route)+[0])}
                value = sum(1 << arcs.index(a) for a in selected)
                cost = sum(route_cost(route, model['distance']) for route in routes)
                assert value not in result or abs(result[value]-cost) < 1e-9
                result[value] = cost
    return result


def make_model(spec):
    if spec['problem'] != 'vrp':
        return base_model(spec)
    nodes = spec['nodes']
    if nodes not in [4, 5] or spec['vehicles'] != 2:
        raise ValueError('This encoding is certified only for 3–4 customers and two nonempty vehicles')
    coords = np.random.default_rng(spec['seed']).uniform(0, 10, (nodes, 2))
    distance = np.linalg.norm(coords[:, None]-coords[None, :], axis=2)
    arcs = [(i, j) for i in range(nodes) for j in range(nodes) if i != j]
    index = {a: k for k, a in enumerate(arcs)}; n = len(arcs)
    penalty = 2*distance.sum(); linear = np.array([distance[a] for a in arcs]); quadratic = np.zeros((n, n)); constant = 0.
    for i in range(nodes):
        rhs = 2 if i == 0 else 1
        for qs in [[index[i, j] for j in range(nodes) if j != i], [index[j, i] for j in range(nodes) if j != i]]:
            linear[qs] += penalty*(1-2*rhs); constant += penalty*rhs**2
            for u, v in itertools.combinations(qs, 2):
                quadratic[min(u, v), max(u, v)] += 2*penalty
    # With <=4 customers and two nonempty depot routes, only a disconnected
    # 2-cycle can survive the degree constraints. Forbid every customer 2-cycle.
    for i, j in itertools.combinations(range(1, nodes), 2):
        u, v = index[i, j], index[j, i]
        quadratic[min(u, v), max(u, v)] += penalty
    scale = max(1., abs(linear).max(), abs(quadratic).max())
    model = dict(spec=spec, problem='vrp', instance_id=spec['id'], seed=spec['seed'], nodes=nodes,
                 customers=nodes-1, vehicles=2, n_qubits=n, variables=arcs, coordinates=coords.tolist(),
                 distance=distance.tolist(), penalty=float(penalty), constant=constant, scale=float(scale),
                 linear=(linear/scale).tolist(), quadratic=(quadratic/scale).tolist(),
                 encoding='directed arcs; degree squares and customer two-cycle penalties; valid only for <=4 customers, two nonempty vehicles')
    costs = feasible_costs(model); model['optimum'] = min(costs.values())
    model['classical_reference_evaluations'] = len(costs)
    return model


def decode(model, values):
    if model['problem'] != 'vrp':
        return base_decode(model, values)
    values = np.asarray(values, dtype=np.uint64)
    bits = ((values[:, None] >> np.arange(model['n_qubits'], dtype=np.uint64)) & 1).astype(float)
    arcs = [tuple(a) for a in model['variables']]; index = {a: k for k, a in enumerate(arcs)}
    costs = bits @ np.array([model['distance'][i][j] for i, j in arcs]); violations = np.zeros(len(values))
    for i in range(model['nodes']):
        rhs = 2 if i == 0 else 1
        for qs in [[q for q, a in enumerate(arcs) if a[0] == i], [q for q, a in enumerate(arcs) if a[1] == i]]:
            violations += (bits[:, qs].sum(axis=1)-rhs)**2
    for i, j in itertools.combinations(range(1, model['nodes']), 2):
        violations += bits[:, index[i, j]]*bits[:, index[j, i]]
    feasible = violations == 0
    return dict(cost=costs, violations=violations, feasible=feasible,
                good=feasible & (costs <= 1.05*model['optimum']+1e-9),
                optimal=feasible & np.isclose(costs, model['optimum'], atol=1e-9, rtol=0),
                energy=(costs+model['penalty']*violations)/model['scale'])


def validate_model(model, exhaustive=None):
    expected = feasible_costs(model); n = model['n_qubits']
    assert abs(min(expected.values())-model['optimum']) < 1e-9
    # Enumerate all VRP states to certify the subtour treatment, using chunks.
    exhaustive = model['problem'] == 'vrp' if exhaustive is None else exhaustive
    seen = set(); minimum = float('inf')
    batches = (np.arange(start, min(start+4096, 1 << n), dtype=np.uint64)
               for start in range(0, 1 << n, 4096)) if exhaustive else [
                   np.array(sorted(expected), dtype=np.uint64),
                   np.random.default_rng(model['seed']+1).integers(0, 1 << n, 4096, dtype=np.uint64)]
    linear, quadratic = np.array(model['linear']), np.array(model['quadratic'])
    for values in batches:
        d = decode(model, values)
        bits = ((values[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
        expanded = bits@linear + np.sum((bits@quadratic)*bits, axis=1) + model['constant']/model['scale']
        assert np.allclose(expanded, d['energy'], atol=1e-9, rtol=0)
        wanted = np.array([int(v) in expected for v in values])
        assert np.array_equal(d['feasible'], wanted)
        assert all(abs(d['cost'][i]-expected[int(v)]) < 1e-9 for i, v in enumerate(values) if wanted[i])
        seen.update(int(v) for v in values[wanted]); minimum = min(minimum, float(expanded.min()))
    assert seen == set(expected)
    if exhaustive:
        assert abs(minimum-model['optimum']/model['scale']) < 1e-9
    return dict(exhaustive=exhaustive, feasible_decisions=len(expected), exact_optimum=model['optimum'],
                penalty_expansion_checked=True, independently_enumerated_routes=True)


def metrics(model, counts):
    expected = feasible_costs(model); keys = list(counts)
    assert all(len(k) == model['n_qubits'] and not set(k)-{'0', '1'} and isinstance(counts[k], int) and counts[k] > 0 for k in keys)
    values = np.array([int(k, 2) for k in keys], dtype=np.uint64); weights = np.array([counts[k] for k in keys])
    d = decode(model, values); shots = int(weights.sum()); result = dict(shots=shots)
    for event in ['feasible', 'good', 'optimal']:
        count = int(weights@d[event])
        check = sum(counts[k] for k in keys if int(k, 2) in expected and
                    (event == 'feasible' or (expected[int(k, 2)] <= 1.05*model['optimum']+1e-9 if event == 'good' else abs(expected[int(k, 2)]-model['optimum']) <= 1e-9)))
        assert count == check
        result.update({event+'_count': count, 'p_'+event: count/shots})
    feasible = [expected[int(k, 2)] for k in keys if int(k, 2) in expected]
    result['best_feasible_cost'] = min(feasible) if feasible else None
    return result
