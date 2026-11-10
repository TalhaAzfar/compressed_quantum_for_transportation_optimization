"""Certify one-qubit block optimization and exact native-CZ cancellations.

The certificates are input-state independent and operate on 2x2 matrices and
commutation, so validation does not require simulating the physical device.
"""
import numpy as np


def wire_blocks(qc):
    from qiskit.quantum_info import Operator
    pending=[np.eye(2,dtype=complex) for _ in range(qc.num_qubits)]
    blocks=[[] for _ in pending]
    for inst in qc.data:
        qubits=tuple(qc.find_bit(q).index for q in inst.qubits);name=inst.operation.name
        if len(qubits)==1 and name not in ['measure','reset','barrier','delay']:
            q=qubits[0];pending[q]=Operator(inst.operation).data@pending[q]
        else:
            assert name in ['cz','measure','barrier','delay'],name
            marker=(name,qubits,tuple(qc.find_bit(c).index for c in inst.clbits))
            for q in qubits:
                blocks[q].append((marker,pending[q]));pending[q]=np.eye(2,dtype=complex)
    for q in range(qc.num_qubits):blocks[q].append((('end',),pending[q]))
    return blocks


def certify_one_qubit_change(before,after):
    assert before.num_qubits==after.num_qubits and before.num_clbits==after.num_clbits
    left,right=wire_blocks(before),wire_blocks(after);phase=float(after.global_phase)-float(before.global_phase)
    errors=[]
    for a,b in zip(left,right):
        assert len(a)==len(b),'Two-qubit/measurement skeleton changed'
        for (marker,u),(other,v) in zip(a,b):
            assert marker==other,'Wire order changed'
            overlap=np.trace(u.conj().T@v)/2
            assert abs(abs(overlap)-1)<1e-10
            p=np.angle(overlap);error=float(np.max(np.abs(v-np.exp(1j*p)*u)))
            assert error<1e-10,error
            errors.append(error);phase+=p
    phase_error=float(abs(np.exp(1j*phase)-1));assert phase_error<1e-8,phase_error
    return dict(blocks=len(errors),maximum_block_error=max(errors,default=0.),
                accumulated_error_bound=sum(errors)+phase_error,global_phase_error=phase_error)


def cancel_commuting_cz(qc):
    from qiskit.quantum_info import Operator
    pending={};deleted=set();pairs=[]
    for i,inst in enumerate(qc.data):
        qubits=tuple(qc.find_bit(q).index for q in inst.qubits);name=inst.operation.name
        if name=='cz':
            edge=tuple(sorted(qubits))
            if edge in pending:
                first=pending.pop(edge);deleted.update([first,i]);pairs.append((first,i,edge))
            else:pending[edge]=i
        else:
            diagonal=False
            if len(qubits)==1 and name not in ['measure','reset','barrier','delay']:
                matrix=Operator(inst.operation).data
                # Only exactly diagonal gates commute in this transformation.
                diagonal=bool(matrix[0,1]==0 and matrix[1,0]==0)
            if not diagonal:
                pending={edge:j for edge,j in pending.items() if not set(edge).intersection(qubits)}
    # Independent replay: every intervening operation must commute with CZ.
    for first,last,edge in pairs:
        for index in [first,last]:
            inst=qc.data[index]
            assert inst.operation.name=='cz' and tuple(sorted(qc.find_bit(q).index for q in inst.qubits))==edge
        for inst in qc.data[first+1:last]:
            touched=set(qc.find_bit(q).index for q in inst.qubits).intersection(edge)
            if not touched or inst.operation.name=='cz':continue
            assert len(inst.qubits)==1 and inst.operation.name not in ['measure','barrier','reset','delay']
            matrix=Operator(inst.operation).data;assert matrix[0,1]==0 and matrix[1,0]==0
    out=qc.copy_empty_like();out.global_phase=qc.global_phase
    for i,inst in enumerate(qc.data):
        if i not in deleted:out.append(inst.operation,inst.qubits,inst.clbits)
    assert int(qc.count_ops().get('cz',0))-int(out.count_ops().get('cz',0))==2*len(pairs)
    return out,dict(cancelled_pairs=len(pairs),certificate=[dict(first=a,last=b,edge=list(e)) for a,b,e in pairs])


def cleanup(qc):
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition
    before=qc.copy();current=qc.copy();rounds=[]
    for iteration in range(4):
        merged=PassManager([Optimize1qGatesDecomposition(basis=['rz','sx','x'])]).run(current)
        certificate=certify_one_qubit_change(current,merged)
        reduced,cancellations=cancel_commuting_cz(merged)
        rounds.append(dict(one_qubit_certificate=certificate,**cancellations))
        current=reduced
        if not cancellations['cancelled_pairs']:break
    if rounds[-1]['cancelled_pairs']:
        merged=PassManager([Optimize1qGatesDecomposition(basis=['rz','sx','x'])]).run(current)
        rounds.append(dict(one_qubit_certificate=certify_one_qubit_change(current,merged),cancelled_pairs=0,certificate=[]))
        current=merged
    result=dict(verified=True,rounds=rounds,before_counts={k:int(v) for k,v in before.count_ops().items()},
                after_counts={k:int(v) for k,v in current.count_ops().items()},
                before_cz_depth=before.depth(lambda i:i.operation.name=='cz'),
                after_cz_depth=current.depth(lambda i:i.operation.name=='cz'),
                accumulated_error_bound=sum(r['one_qubit_certificate']['accumulated_error_bound'] for r in rounds))
    assert result['accumulated_error_bound']<1e-8
    return current,result


def cleaned_resources(qc,resources,target,certificate):
    result=dict(resources)
    result.update(nominal_cz_count=resources['cz_count'],nominal_cz_depth=resources['cz_depth'],
                  cz_count=int(qc.count_ops().get('cz',0)),cz_depth=qc.depth(lambda i:i.operation.name=='cz'),depth=qc.depth())
    error=0.
    for inst in qc.data:
        qubits=tuple(qc.find_bit(q).index for q in inst.qubits)
        assert target.instruction_supported(operation_name=inst.operation.name,qargs=qubits)
        prop=target[inst.operation.name].get(qubits)
        if prop and prop.error:error+=prop.error
    result['calibration_error_sum']=float(error)
    result['conservative_accumulated_matrix_error_bound']+=certificate['accumulated_error_bound']
    result['cleanup_certificate']=certificate
    result['verification']+='; certified one-qubit blocks and commuting-CZ cancellation'
    return result
