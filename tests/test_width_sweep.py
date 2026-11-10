import numpy as np
import pytest
from revision.width_sweep_models import make_model, feasible_costs, decode, validate_model, metrics


@pytest.mark.parametrize('nodes,expected', [(4, 6), (5, 36)])
def test_vrp_full_encoding(nodes, expected):
    m = make_model(dict(id='test', problem='vrp', nodes=nodes, vehicles=2, seed=100801))
    report = validate_model(m, exhaustive=True)
    assert report['feasible_decisions'] == expected


def test_vrp_disconnected_subtour_rejected():
    m = make_model(dict(id='test', problem='vrp', nodes=5, vehicles=2, seed=100801))
    arcs = [tuple(a) for a in m['variables']]
    selected = [(0, 1), (1, 0), (0, 2), (2, 0), (3, 4), (4, 3)]
    value = sum(1 << arcs.index(a) for a in selected)
    d = decode(m, [value])
    assert d['violations'][0] == 1 and not d['feasible'][0]
    assert value not in feasible_costs(m)


def test_vrp_scope_is_enforced():
    with pytest.raises(ValueError):
        make_model(dict(id='test', problem='vrp', nodes=6, vehicles=2, seed=100801))


@pytest.mark.parametrize('spec', [dict(problem='flp', customers=2, facilities=2), dict(problem='tsp', nodes=4), dict(problem='vrp', nodes=5, vehicles=2)])
def test_independent_count_events_and_order(spec):
    m = make_model(dict(id='test', seed=123, **spec)); costs = feasible_costs(m)
    counts = {format(value, f"0{m['n_qubits']}b"): i+1 for i, value in enumerate(costs)}
    counts['0'*m['n_qubits']] = 17
    r = metrics(m, counts)
    assert r['feasible_count'] == sum(counts.values())-17
    assert r['optimal_count'] == sum(counts[format(v, f"0{m['n_qubits']}b")] for v, cost in costs.items() if abs(cost-m['optimum']) < 1e-9)


def test_tensor_explicit_gradient_and_native_state():
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from revision.tensor_prefix_core import native_ansatz, mps_state
    from revision.full_circuit_explicit import explicit_settings
    from revision.hardware_width_sweep import exact_state
    m = make_model(dict(id='test', problem='flp', customers=2, facilities=2, seed=77))
    from revision.tensor_prefix_core import target_circuit
    source = target_circuit(m, 1., 12, 2.)
    target = mps_state(source, 64); target._psi /= np.sqrt(float(abs(target.psi.H@target.psi)))
    ansatz = native_ansatz(6, [(i, i+1) for i in range(5)], 2)
    objective = MaximizeStateFidelity(target, ansatz, explicit_settings(64))
    x = np.random.default_rng(42).normal(0, .03, ansatz.num_parameters)
    loss, gradient = objective.loss_function(x)
    expected = abs(np.vdot(exact_state(source), exact_state(ansatz.assign_parameters(x))))**2
    # The production MPS target truncates at 1e-10 per factorization; accumulated
    # amplitude error need not be below 1e-9 after the full anneal.
    assert abs(1-loss-expected) < 1e-6
    for q in [0, 5, 12, len(x)-1]:
        delta = np.zeros(len(x)); delta[q] = 1e-5
        numerical = (objective.loss_function(x+delta)[0]-objective.loss_function(x-delta)[0])/2e-5
        assert abs(numerical-gradient[q]) < 1e-6
