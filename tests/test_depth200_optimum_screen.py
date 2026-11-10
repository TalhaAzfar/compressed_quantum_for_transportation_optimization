import itertools
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from revision.depth200_optimum_screen import feasible_values, metrics_from_amplitudes, mps_metrics
from revision.tensor_prefix_models import make_model, decode
from revision.flp42_fidelity_parallel import state


@pytest.mark.parametrize('spec', [dict(problem='tsp',nodes=4,seed=101),
                                  dict(problem='flp',facilities=2,customers=2,seed=102)])
def test_sparse_events_against_independent_full_distribution(spec):
    model=make_model(spec);n=model['n_qubits']
    values=feasible_values(model)
    decoded=decode(model,np.arange(1<<n,dtype=np.uint64))
    assert sorted(values)==np.flatnonzero(decoded['feasible']).tolist()
    qc=QuantumCircuit(n);rng=np.random.default_rng(123)
    for q in range(n):qc.ry(rng.normal(),q);qc.rz(rng.normal(),q)
    for q in range(n-1):qc.cx(q,q+1);qc.ry(rng.normal(),q+1)
    exact=np.asarray(Statevector.from_instruction(qc));p=abs(exact)**2
    a=metrics_from_amplitudes(model,values,exact[values])
    b=mps_metrics(model,state(qc,32,1e-13))
    for event in ['feasible','good','optimal']:
        expected=p[decoded[event]].sum()
        assert a['p_'+event]==pytest.approx(expected,abs=1e-12)
        assert b['p_'+event]==pytest.approx(expected,abs=1e-12)


def test_tsp_optimum_matches_direct_tour_enumeration():
    model=make_model(dict(problem='tsp',nodes=6,seed=83022))
    d=np.asarray(model['distance'])
    costs=[sum(d[a,b] for a,b in zip((0,)+t,t+(0,)))
           for t in itertools.permutations(range(1,6))]
    decoded=decode(model,feasible_values(model))
    assert len(costs)==120 and np.allclose(costs,decoded['cost'])
    assert min(costs)==pytest.approx(model['optimum'])
    assert decoded['optimal'].sum()==2
