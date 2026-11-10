"""Gatewise verified SABRE routing; standalone until pre-hardware validation."""
import numpy as np
from qiskit import QuantumCircuit,transpile

def compile_isa_sabre(qc, target, physical, edges, seed=81003, verify=True):
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
    from qiskit.transpiler import CouplingMap, PassManager
    from qiskit.transpiler.passes import SabreSwap
    routing_input=QuantumCircuit(n)
    for index,inst in enumerate(source.data):
        if inst.operation.name=="barrier":continue
        assert 1<=inst.operation.num_qubits<=2 and not inst.clbits and not inst.operation.is_parameterized()
        operation=inst.operation.to_mutable()
        operation.label=f"aqc_source_{index}"
        routing_input.append(operation,[source.find_bit(q).index for q in inst.qubits])
    coupling=CouplingMap([list(pair) for u,v in edges for pair in [(u,v),(v,u)]])
    routed=PassManager([SabreSwap(coupling_map=coupling,heuristic="decay",seed=seed,trials=8)]).run(routing_input)
    for inst in routed.data:
        wires=[routed.find_bit(q).index for q in inst.qubits]
        label=inst.operation.label
        if label is None and inst.operation.name=="swap":
            swap(*wires);continue
        assert label and label.startswith("aqc_source_")
        index=int(label.removeprefix("aqc_source_"))
        original=source.data[index]
        logical=[source.find_bit(q).index for q in original.qubits]
        assert inst.operation.name==original.operation.name
        assert list(inst.operation.params)==list(original.operation.params)
        assert [wire_to_logical[q] for q in wires]==logical
        if len(wires)==2:assert tuple(sorted(wires)) in edges
        local.compose(native_gate(original.operation),qubits=wires,inplace=True)
        trace.append(dict(kind="gate",source_index=index,logical=logical,wires=wires))
    local.global_phase+=source.global_phase
    # Independent replay of the emitted routing certificate.
    replay=list(range(n));gate_count=0
    expected_order={q:[] for q in range(n)}
    observed_order={q:[] for q in range(n)}
    for index,inst in enumerate(source.data):
        if inst.operation.name!="barrier":
            for q in inst.qubits:expected_order[source.find_bit(q).index].append(index)
    for event in trace:
        wires=event["wires"]
        if event["kind"]=="swap":
            u,v=wires;replay[u],replay[v]=replay[v],replay[u]
        else:
            inst=source.data[event["source_index"]]
            expected=[source.find_bit(q).index for q in inst.qubits]
            assert [replay[q] for q in wires]==expected
            for q in expected:observed_order[q].append(event["source_index"])
            gate_count+=1
    assert observed_order==expected_order
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
        verification="gatewise_unitary_sabre_swap_and_independent_wire_order_trace",maximum_gate_matrix_error=max_error,
        conservative_accumulated_matrix_error_bound=4*max_error*len(trace),
        decomposition_types=len(decompositions),source_gates=gate_count,swap_count=sum(e["kind"]=="swap" for e in trace),
        routing_certificate=trace,transpilation_fidelity=check)
