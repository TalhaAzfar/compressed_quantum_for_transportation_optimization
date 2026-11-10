import numpy as np
import pytest
from revision.flp_mps_events import probabilities

@pytest.mark.parametrize('facilities,customers,seed',[(2,2,98011),(3,2,98012),(3,3,98013)])
def test_mps_constraint_contraction_matches_full_exact_distribution(facilities,customers,seed):
    from revision.tensor_prefix_models import make_model,decode
    from revision.tensor_prefix_core import target_circuit,mps_state
    from revision.full_circuit_compression import exact_state,mps_dense_qiskit
    from revision.full_circuit_explicit import explicit_settings
    from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
    model=make_model(dict(id='test',problem='flp',facilities=facilities,customers=customers,seed=seed))
    qc=target_circuit(model,1.,steps=4,duration=2.)
    state=mps_state(qc,64);got=probabilities(model,state)
    # The contraction is exact for its supplied MPS. Separate this invariant
    # from the simulator's deliberate singular-value truncation error.
    dense=mps_dense_qiskit(state,model['n_qubits']);p=np.abs(dense)**2;p/=p.sum()
    decoded=decode(model,np.arange(len(p)))
    for event in ['feasible','good','optimal']:
        assert got['p_'+event]==pytest.approx(float(p@decoded[event]),abs=1e-11)
        assert got[event+'_count']==int(decoded[event].sum())
    untruncated=tensornetwork_from_circuit(qc,explicit_settings(64,cutoff=0.))
    actual=probabilities(model,untruncated);exact=np.abs(exact_state(qc))**2
    for event in ['feasible','good','optimal']:
        assert actual['p_'+event]==pytest.approx(float(exact@decoded[event]),abs=1e-10)

def test_42_qubit_product_reference_without_exponential_vector():
    import math
    from qiskit import QuantumCircuit
    from revision.tensor_prefix_models import make_model
    from revision.tensor_prefix_core import mps_state
    model=make_model(dict(id='test',problem='flp',facilities=6,customers=6,seed=98014))
    qc=QuantumCircuit(42);qc.h(range(42));got=probabilities(model,mps_state(qc,2))
    count=sum(math.comb(6,k)*k**6 for k in range(1,7))
    assert got['feasible_count']==count
    assert got['p_feasible']==pytest.approx(count/2**42,rel=1e-10)
    assert got['p_good']==pytest.approx(got['good_count']/2**42,rel=1e-10)
