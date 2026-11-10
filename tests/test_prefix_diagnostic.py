import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from revision.prefix_diagnostic import feasible_assignments,energy_diagonal,continue_state,incumbent
from revision.tensor_prefix_models import make_model,decode
from revision.tensor_prefix_core import qaoa,mps_state


def test_feasible_generator_agrees_with_full_bit_space_and_incumbent():
    model=make_model(dict(problem="flp",customers=2,facilities=2,seed=93003))
    values,costs=feasible_assignments(model)
    d=decode(model,np.arange(64))
    assert set(values)==set(np.flatnonzero(d["feasible"]))
    assert np.allclose(costs,d["cost"][values])
    answer=incumbent(model);result=decode(model,[answer["value"]])
    assert result["feasible"][0] and result["cost"][0]==answer["cost"]


def test_crossed_continuation_matches_qiskit_on_arbitrary_input():
    model=make_model(dict(problem="flp",customers=2,facilities=2,seed=93004))
    rng=np.random.default_rng(93005)
    state=rng.normal(size=64)+1j*rng.normal(size=64);state/=np.linalg.norm(state)
    for angles in [[.37,-.28],[-.21,.61],[0.,0.]]:
        actual=continue_state(state,energy_diagonal(model),angles)
        expected=Statevector(state).evolve(qaoa(QuantumCircuit(6),model,angles)).data
        assert abs(abs(np.vdot(expected,actual))**2-1)<1e-12


def test_incumbent_uses_service_costs_to_choose_the_initial_singleton():
    answer=incumbent(dict(opening=[1,4],service=[[10,0],[10,0]]))
    assert answer["opened"]==[1] and answer["cost"]==4


def test_mps_sample_bit_order_on_asymmetric_basis_state():
    qc=QuantumCircuit(6);qc.x(0);qc.x(4)
    for bits,p in mps_state(qc,8).psi.sample(4,seed=93006):
        assert int("".join(str(int(b)) for b in reversed(bits)),2)==17
        assert abs(p-1)<1e-12
