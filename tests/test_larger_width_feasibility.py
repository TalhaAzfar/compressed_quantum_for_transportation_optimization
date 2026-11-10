import numpy as np
import pytest


@pytest.mark.parametrize('engine',['aer','quimb'])
def test_direct_mps_target_preserves_state_events_and_aqc_gradient(engine):
    from qiskit.quantum_info import Statevector
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from revision.tensor_prefix_models import make_model, decode
    from revision.tensor_prefix_core import target_circuit, native_ansatz
    from revision.full_circuit_explicit import explicit_settings
    from revision.flp_mps_events import probabilities
    from revision.larger_width_feasibility import aer_state, fuse_zz
    model = make_model(dict(problem='flp', facilities=2, customers=2, seed=73))
    source = target_circuit(model, 1., 12, 2.)
    fused, count = fuse_zz(source)
    assert count > 0
    assert abs(np.vdot(Statevector(source).data,Statevector(fused).data))**2 == pytest.approx(1.,abs=1e-10)
    if engine == 'aer':
        target = aer_state(fused, 64, 0., 2)
    else:
        from revision.flp42_fidelity_parallel import state
        target = state(fused, 64, 0.)
    exact = Statevector(source).data
    dense = np.asarray(target.psi.to_dense()).reshape((2,)*6).transpose(tuple(reversed(range(6)))).reshape(-1)
    assert abs(np.vdot(exact, dense))**2 == pytest.approx(1., abs=1e-10)
    events = probabilities(model, target)
    decoded = decode(model, np.arange(64, dtype=np.uint64))
    for event in ['feasible', 'good', 'optimal']:
        assert events['p_'+event] == pytest.approx(float((abs(exact)**2)[decoded[event]].sum()), abs=1e-8)
    ansatz = native_ansatz(6, [(i,i+1) for i in range(5)], 2)
    objective = MaximizeStateFidelity(target, ansatz, explicit_settings(64, 0.))
    x = np.random.default_rng(99).normal(0,.1,ansatz.num_parameters)
    loss, gradient = objective.loss_function(x)
    expected = 1-abs(np.vdot(exact, Statevector(ansatz.assign_parameters(x)).data))**2
    assert float(loss) == pytest.approx(expected, abs=1e-8)
    for i in [0,5,len(x)-1]:
        dx = np.zeros_like(x); dx[i]=1e-6
        a = objective.loss_function(x+dx)[0]; b = objective.loss_function(x-dx)[0]
        assert gradient[i] == pytest.approx((a-b)/2e-6, abs=1e-7)


def test_screening_uses_feasibility_without_a_fidelity_threshold():
    from revision.larger_width_feasibility import choose_screen, best_per_depth
    rows = [dict(method='shallow', layers=2, fidelity=.95, candidate_id='s', resources=dict(cz_depth=14)),
            dict(method='deeper', layers=6, fidelity=.7, candidate_id='d', resources=dict(cz_depth=38)),
            dict(method='product', layers=0, fidelity=.99, candidate_id='p', resources=dict(cz_depth=0))]
    pooled = [dict(method=r['method'], feasible_count=k, shots=1000) for r,k in zip(rows,[2,10,20])]
    assert choose_screen(rows,pooled)['method']=='deeper'
    records=[dict(layers=2,index=0,verified=False,fidelity=.99),
             dict(layers=2,index=1,verified=True,fidelity=.6),
             dict(layers=0,index=0,verified=True,fidelity=.5),
             dict(status='worker_failed',verified=False)]
    chosen=best_per_depth(records)
    assert [r['fidelity'] for r in chosen]==[.5,.6]


def test_zz_fusion_is_a_unitary_identity_including_reversed_wires():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from revision.larger_width_feasibility import fuse_zz
    qc=QuantumCircuit(3);qc.global_phase=.17
    qc.h(1)
    for u,v,t in [(0,2,.73),(2,1,-1.13),(1,0,.07)]:
        qc.cx(u,v);qc.rz(t,v);qc.cx(u,v)
    fused,count=fuse_zz(qc)
    assert count==3
    assert np.max(abs(Operator(qc).data-Operator(fused).data))<1e-12
