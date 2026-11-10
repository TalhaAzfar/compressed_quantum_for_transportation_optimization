"""AQC-Tensor MPS fitting and native prefix/QAOA circuit construction."""
from __future__ import annotations

from functools import partial
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize

from .corrected_study import ising_operator
from .nighthawk_study import evolution
from .tensor_prefix_models import numpy_model


def simulator(bond):
    import quimb.tensor as qtn
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    return QuimbSimulator(partial(qtn.CircuitMPS, max_bond=bond, cutoff=1e-10), autodiff_backend="jax")


def mps_state(qc, bond):
    from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
    return tensornetwork_from_circuit(qc, simulator(bond))


def fidelity(a, b):
    a, b = a.psi, b.psi
    return float(abs(a.H@b)**2/(abs(a.H@a)*abs(b.H@b)))


def native_ansatz(n, edges, layers):
    from .native_aqc import edge_matchings
    theta = ParameterVector("a", layers*(2*n+len(edges))+2*n)
    qc, k = QuantumCircuit(n), 0
    qc.h(range(n))
    for layer in range(layers+1):
        for q in range(n):
            qc.ry(theta[k], q); k += 1
            qc.rz(theta[k], q); k += 1
        if layer < layers:
            for matching in edge_matchings(edges):
                for u, v in matching:
                    # Decompose RZZ so AQC-Tensor and every hardware compilation
                    # use the same concrete gate-level parameterization.
                    qc.cx(u, v)
                    qc.rz(theta[k], v); k += 1
                    qc.cx(u, v)
    assert k == len(theta)
    return qc


def path_patch(target, n, seed=81001):
    """Calibration-only seeded search for a physical path, then induced edges."""
    errors = {tuple(sorted(e)): float(p.error if p and p.error is not None else .05)
              for e, p in target["cz"].items()}
    adj = {q: [] for q in range(target.num_qubits)}
    for u, v in errors:
        adj[u].append(v); adj[v].append(u)
    rng, choices = np.random.default_rng(seed+n), []
    for attempt in range(2500):
        path = [int(rng.integers(target.num_qubits))]
        while len(path) < n:
            candidates = [q for q in adj[path[-1]] if q not in path]
            if not candidates:
                break
            score = [errors[tuple(sorted((path[-1], q)))]+rng.uniform(0, .03) for q in candidates]
            path.append(candidates[int(np.argmin(score))])
        if len(path) == n:
            choices.append((sum(errors[tuple(sorted(e))] for e in zip(path, path[1:]))/(n-1), path))
    if not choices:
        raise ValueError(f"No physical path of length {n}")
    score, physical = min(choices)
    edges = [(i, j) for i in range(n) for j in range(i+1, n) if tuple(sorted((physical[i], physical[j]))) in errors]
    return physical, edges, dict(mean_path_cz_error=score, path_edges=[(i, i+1) for i in range(n-1)])


def target_circuit(model, endpoint, steps=8, duration=2.):
    return evolution(numpy_model(model), duration, steps, end=endpoint)


def qaoa(prefix, model, angles):
    from qiskit.circuit.library import PauliEvolutionGate
    qc = prefix.copy()
    for gamma, beta in np.asarray(angles).reshape(-1, 2):
        qc.append(PauliEvolutionGate(ising_operator(numpy_model(model)), time=float(gamma)), range(qc.num_qubits))
        qc.rx(2*float(beta), range(qc.num_qubits))
    return qc.decompose(reps=2)


def fit_prefix(target, ansatz, bond=32, maxiter=40, seed=81002):
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    settings = simulator(bond)
    norm = float(abs(target.psi.H@target.psi))
    normalized = target.copy()
    normalized._psi /= np.sqrt(norm)
    objective = MaximizeStateFidelity(normalized, ansatz, settings)
    x0 = np.random.default_rng(seed).normal(0, .02, ansatz.num_parameters)
    history, start = [], time.perf_counter()
    def fun(x):
        value, grad = objective.loss_function(x)
        if not np.isfinite(value) or not np.all(np.isfinite(grad)):
            raise ValueError("Nonfinite tensor objective or gradient")
        history.append(float(value))
        return float(value), np.asarray(grad, dtype=float)
    initial, _ = fun(x0)
    result = minimize(fun, x0, jac=True, method="L-BFGS-B", options={"maxiter":maxiter, "maxls":12, "ftol":1e-10})
    loss, _ = fun(result.x)
    bound = ansatz.assign_parameters(result.x)
    check = fidelity(target, mps_state(bound, max(bond, 64)))
    assert abs((1-loss)-check) < 2e-5, (loss, check)
    return bound, dict(parameters=result.x.tolist(), parameter_count=ansatz.num_parameters,
                       initial_fidelity=1-initial, fidelity=check, loss_history=history,
                       iterations=int(result.nit), evaluations=len(history), seconds=time.perf_counter()-start,
                       optimizer_message=str(result.message), target_norm_before_normalization=norm)


def mps_energy(qc, model, bond=64):
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="matrix_product_state", matrix_product_state_max_bond_dimension=bond,
                       matrix_product_state_truncation_threshold=1e-10, max_parallel_threads=1)
    circuit = qc.copy()
    circuit.save_expectation_value(ising_operator(numpy_model(model)), range(qc.num_qubits), label="energy")
    result = sim.run(circuit, shots=1).result()
    if not result.success:
        raise RuntimeError(result.status)
    return float(result.data(0)["energy"])


def train_qaoa(prefix, model, p=1, maxeval=24, bond=64):
    start, history = time.perf_counter(), []
    x0 = np.array([.2/p, -.2]*p)
    def objective(theta):
        value = mps_energy(qaoa(prefix, model, theta), model, bond)
        history.append(dict(parameters=np.asarray(theta).tolist(), energy=value))
        return value
    bounds = [(-.6, .6), (-.8, .8)]*p
    result = minimize(objective, x0, method="COBYLA", bounds=bounds,
                      options={"maxiter":maxeval, "rhobeg":.12, "tol":.005})
    qc = qaoa(prefix, model, result.x)
    low, high = mps_energy(qc, model, bond), mps_energy(qc, model, 2*bond)
    return qc, dict(parameters=result.x.tolist(), energy=high, low_bond_energy=low,
                    energy_bond_difference=abs(high-low), bond=bond, check_bond=2*bond,
                    seconds=time.perf_counter()-start, history=history, objective="ideal mean normalized Ising energy",
                    optimizer_message=str(result.message))


def compile_isa_mps_diagnostic(qc, target, physical, edges, seed=81003, verify=True):
    """Route within the chosen patch; check layout-aware local MPS equivalence."""
    n = qc.num_qubits
    measured = qc.copy(); measured.measure_all()
    cmap = [[u, v] for a, b in edges for u, v in [(a, b), (b, a)]]
    local = transpile(measured, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cmap,
                      initial_layout=list(range(n)), optimization_level=1, seed_transpiler=seed)
    mapping = local.layout.final_index_layout() if local.layout else list(range(n))
    assert sorted(mapping) == list(range(n))
    measurements = {local.find_bit(inst.clbits[0]).index:local.find_bit(inst.qubits[0]).index
                    for inst in local.data if inst.operation.name == "measure"}
    assert [measurements[i] for i in range(n)] == mapping
    for inst in local.data:
        if inst.operation.num_qubits == 2:
            assert tuple(sorted(local.find_bit(q).index for q in inst.qubits)) in edges
    checks = {}
    if verify:
        inverse = np.argsort(mapping).tolist()
        restored = QuantumCircuit(n).compose(local.remove_final_measurements(inplace=False), qubits=inverse)
        if n <= 16:
            from qiskit.quantum_info import Statevector
            expected, actual = Statevector.from_instruction(qc).data, Statevector.from_instruction(restored).data
            check = float(abs(np.vdot(expected, actual))**2)
            assert abs(1-check) < 1e-8, check
            checks = dict(transpilation_fidelity=check, verification="exact_statevector")
        else:
            # Both independent contractions must stabilize before an equivalence
            # discrepancy can be interpreted as a compiler failure.
            a, b = mps_state(qc, 64), mps_state(restored, 64)
            a2, b2 = mps_state(qc, 128), mps_state(restored, 128)
            check, ca, cb = fidelity(a2, b2), fidelity(a, a2), fidelity(b, b2)
            checks = dict(transpilation_fidelity=check, original_bond_convergence=ca,
                          transpiled_bond_convergence=cb, verification="normalized_MPS_64_128")
            assert min(check, ca, cb) > .999, checks
    isa = transpile(local, target=target, initial_layout=physical, routing_method="none", optimization_level=0)
    allowed = {tuple(sorted(e)) for e in target["cz"]}
    error_sum, cz = 0., 0
    for inst in isa.data:
        name = inst.operation.name
        qubits = tuple(isa.find_bit(q).index for q in inst.qubits)
        if name == "barrier":
            continue
        assert all(q in physical for q in qubits)
        if len(qubits) == 2:
            assert name == "cz" and tuple(sorted(qubits)) in allowed
            cz += 1
        prop = target[name].get(qubits)
        if prop and prop.error:
            error_sum += prop.error
    isa_measurements = {isa.find_bit(inst.clbits[0]).index:isa.find_bit(inst.qubits[0]).index
                        for inst in isa.data if inst.operation.name == "measure"}
    assert all(isa_measurements[i] == physical[mapping[i]] for i in range(n))
    return isa, dict(cz_count=cz, cz_depth=isa.depth(lambda inst:inst.operation.name=="cz"),
                     depth=isa.depth(), calibration_error_sum=float(error_sum),
                     physical=physical, final_logical_mapping=mapping, **checks)


def compile_isa(qc, target, physical, edges, seed=81003, verify=True):
    """Gatewise checked native synthesis and explicit permutation-tracked routing.

    Each small decomposition is checked as a unitary, not only on zero input.
    The routing certificate independently reconstructs the logical gate trace.
    This does not depend on a large approximate simulation of the routed state.
    """
    from collections import deque
    from qiskit.quantum_info import Operator, Statevector
    n=qc.num_qubits
    edges={tuple(sorted(e)) for e in edges}
    adjacency={q:sorted(v if u==q else u for u,v in edges if q in (u,v)) for q in range(n)}
    source=qc.decompose(reps=2)
    local=QuantumCircuit(n,n)
    logical_to_wire=list(range(n));wire_to_logical=list(range(n))
    decompositions={};trace=[];max_error=0.
    def native_gate(operation):
        nonlocal max_error
        key=(operation.name,tuple(complex(p) for p in operation.params))
        if key not in decompositions:
            small=QuantumCircuit(operation.num_qubits)
            small.append(operation,range(operation.num_qubits))
            converted=transpile(small,basis_gates=["cz","rz","sx","x"],optimization_level=0)
            expected,actual=Operator(small).data,Operator(converted).data
            error=float(np.max(np.abs(expected-actual)))
            assert error<1e-12,(operation.name,error)
            max_error=max(max_error,error)
            decompositions[key]=converted
        return decompositions[key]
    from qiskit.circuit.library import SwapGate
    swap_native=native_gate(SwapGate())
    def swap(u,v):
        assert tuple(sorted((u,v))) in edges
        local.compose(swap_native,qubits=[u,v],inplace=True)
        a,b=wire_to_logical[u],wire_to_logical[v]
        wire_to_logical[u],wire_to_logical[v]=b,a
        logical_to_wire[a],logical_to_wire[b]=v,u
        trace.append(dict(kind="swap",wires=[u,v]))
    for index,inst in enumerate(source.data):
        op=inst.operation
        if op.name=="barrier":continue
        assert 1<=op.num_qubits<=2 and not inst.clbits and not op.is_parameterized()
        logical=[source.find_bit(q).index for q in inst.qubits]
        if len(logical)==2:
            u,v=[logical_to_wire[q] for q in logical]
            if tuple(sorted((u,v))) not in edges:
                queue=deque([[u]]);seen={u};path=None
                while queue:
                    candidate=queue.popleft()
                    if candidate[-1]==v:path=candidate;break
                    for node in adjacency[candidate[-1]]:
                        if node not in seen:seen.add(node);queue.append(candidate+[node])
                assert path is not None
                for a,b in zip(path[:-2],path[1:-1]):swap(a,b)
        wires=[logical_to_wire[q] for q in logical]
        if len(wires)==2:assert tuple(sorted(wires)) in edges
        local.compose(native_gate(op),qubits=wires,inplace=True)
        trace.append(dict(kind="gate",source_index=index,logical=logical,wires=wires))
    local.global_phase+=source.global_phase
    # Independent replay of the emitted routing certificate.
    replay=list(range(n));gate_count=0
    for event in trace:
        wires=event["wires"]
        if event["kind"]=="swap":
            u,v=wires;replay[u],replay[v]=replay[v],replay[u]
        else:
            inst=source.data[event["source_index"]]
            expected=[source.find_bit(q).index for q in inst.qubits]
            assert [replay[q] for q in wires]==expected
            gate_count+=1
    assert gate_count==sum(inst.operation.name!="barrier" for inst in source.data)
    assert all(replay[logical_to_wire[q]]==q for q in range(n))
    check=None
    if verify and n<=10:
        restored=QuantumCircuit(n).compose(local.remove_final_measurements(inplace=False),qubits=np.argsort(logical_to_wire).tolist())
        check=float(abs(np.vdot(Statevector.from_instruction(qc).data,Statevector.from_instruction(restored).data))**2)
        assert abs(1-check)<1e-8
    for q,wire in enumerate(logical_to_wire):local.measure(wire,q)
    isa=QuantumCircuit(target.num_qubits,n)
    isa.compose(local,qubits=physical,clbits=range(n),inplace=True)
    error_sum=0.;allowed={tuple(sorted(e)) for e in target["cz"]}
    for inst in isa.data:
        name=inst.operation.name;qubits=tuple(isa.find_bit(q).index for q in inst.qubits)
        assert all(q in physical for q in qubits)
        if len(qubits)==2:assert name=="cz" and tuple(sorted(qubits)) in allowed
        assert target.instruction_supported(operation_name=name,qargs=qubits)
        prop=target[name].get(qubits)
        if prop and prop.error:error_sum+=prop.error
    measured={isa.find_bit(inst.clbits[0]).index:isa.find_bit(inst.qubits[0]).index for inst in isa.data if inst.operation.name=="measure"}
    assert all(measured[q]==physical[logical_to_wire[q]] for q in range(n))
    # Runtime collection expects the standard named register.
    from qiskit import ClassicalRegister
    named=QuantumCircuit(target.num_qubits)
    named.add_register(ClassicalRegister(n,"meas"))
    named.compose(isa,qubits=range(target.num_qubits),clbits=range(n),inplace=True)
    return named,dict(cz_count=int(isa.count_ops().get("cz",0)),cz_depth=isa.depth(lambda inst:inst.operation.name=="cz"),
        depth=isa.depth(),calibration_error_sum=float(error_sum),physical=physical,final_logical_mapping=logical_to_wire,
        verification="gatewise_unitary_and_independent_swap_trace",maximum_gate_matrix_error=max_error,
        conservative_accumulated_matrix_error_bound=4*max_error*len(trace),
        decomposition_types=len(decompositions),source_gates=gate_count,swap_count=sum(e["kind"]=="swap" for e in trace),
        routing_certificate=trace,transpilation_fidelity=check)
