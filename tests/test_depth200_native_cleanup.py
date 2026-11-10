import numpy as np
from qiskit import QuantumCircuit,transpile
from qiskit.quantum_info import Operator
from revision.depth200_native_cleanup import cleanup,cancel_commuting_cz


def test_full_unitary_preserved_under_native_cleanup():
    rng=np.random.default_rng(881)
    qc=QuantumCircuit(4)
    for layer in range(3):
        for q in range(4):qc.rz(rng.normal(),q);qc.ry(rng.normal(),q);qc.rz(rng.normal(),q)
        for q in range(3):qc.rzz(rng.normal(),q,q+1)
    qc.global_phase=.37
    native=transpile(qc,basis_gates=['rz','sx','x','cz'],optimization_level=0)
    cleaned,certificate=cleanup(native)
    assert certificate['verified']
    assert np.max(abs(Operator(native).data-Operator(cleaned).data))<1e-11
    assert cleaned.count_ops()['sx']<native.count_ops()['sx']


def test_zero_identity_extension_is_removed():
    qc=QuantumCircuit(3);qc.h(range(3));qc.cx(0,1);qc.rz(0.,1);qc.cx(0,1)
    native=transpile(qc,basis_gates=['rz','sx','x','cz'],optimization_level=0)
    cleaned,certificate=cleanup(native)
    assert cleaned.count_ops().get('cz',0)==0
    assert np.max(abs(Operator(native).data-Operator(cleaned).data))<1e-11


def test_noncommuting_gate_prevents_cancellation():
    qc=QuantumCircuit(2);qc.cz(0,1);qc.sx(0);qc.cz(0,1)
    result,certificate=cancel_commuting_cz(qc)
    assert result.count_ops()['cz']==2 and certificate['cancelled_pairs']==0


def test_diagonal_interleavings_cancel_and_measurements_stay_ordered():
    qc=QuantumCircuit(3,3);qc.cz(0,1);qc.rz(.2,0);qc.cz(1,2);qc.cz(0,1);qc.measure([2,0,1],[0,1,2])
    cleaned,certificate=cleanup(qc)
    assert cleaned.count_ops()['cz']==1
    assert np.max(abs(Operator(qc.remove_final_measurements(inplace=False)).data-
                      Operator(cleaned.remove_final_measurements(inplace=False)).data))<1e-11
    measured=lambda c:{c.find_bit(i.clbits[0]).index:c.find_bit(i.qubits[0]).index for i in c.data if i.operation.name=='measure'}
    assert measured(qc)==measured(cleaned)
