import numpy as np
import pytest


def test_block_factorization_and_cached_input_gradient_match_exact_full_state():
    from qiskit.quantum_info import Statevector
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation import apply_circuit_to_state
    from revision.tensor_prefix_core import native_ansatz
    from revision.flp42_fidelity_parallel import state
    from revision.full_circuit_explicit import explicit_settings
    from revision.flp42_depth_extension import split_block,bind_subset,block_settings
    n,edges,layers = 4,[(0,1),(1,2),(2,3)],6
    ansatz = native_ansatz(n,edges,layers)
    rng = np.random.default_rng(741)
    x = rng.normal(0,.4,ansatz.num_parameters)
    target_circuit = native_ansatz(n,edges,4).assign_parameters(
        rng.normal(0,.7,native_ansatz(n,edges,4).num_parameters))
    exact_target = Statevector(target_circuit).data
    target = state(target_circuit,64,0.)
    for first,last in [(0,2),(2,4),(4,6)]:
        prefix,block,suffix = split_block(ansatz,n,len(edges),first,last)
        p,b,s = [bind_subset(q,ansatz,x) for q in [prefix,block,suffix]]
        combined = p.compose(b).compose(s)
        assert abs(np.vdot(Statevector(ansatz.assign_parameters(x)).data,
                           Statevector(combined).data))**2 == pytest.approx(1.,abs=1e-11)
        initial = state(p,64,0.)
        effective = apply_circuit_to_state(s.inverse(),target,explicit_settings(64,0.))
        objective = MaximizeStateFidelity(effective,block,block_settings(initial,64,0.))
        mapping = {p:i for i,p in enumerate(ansatz.parameters)}
        indices = [mapping[p] for p in block.parameters]
        loss,gradient = objective.loss_function(x[indices])
        exact_loss = 1-abs(np.vdot(exact_target,Statevector(ansatz.assign_parameters(x)).data))**2
        assert float(loss) == pytest.approx(exact_loss,abs=1e-10)
        for j in [0,len(indices)//2,len(indices)-1]:
            a,b = x.copy(),x.copy()
            a[indices[j]] += 1e-6
            b[indices[j]] -= 1e-6
            fa = 1-abs(np.vdot(exact_target,Statevector(ansatz.assign_parameters(a)).data))**2
            fb = 1-abs(np.vdot(exact_target,Statevector(ansatz.assign_parameters(b)).data))**2
            assert float(gradient[j]) == pytest.approx((fa-fb)/2e-6,abs=2e-8)


def test_identity_extension_preserves_state_and_depth_is_bounded_on_saved_patch():
    import pickle
    from qiskit.quantum_info import Statevector
    from revision.contribution_hardware_core import ROOT,read
    from revision.tensor_prefix_core import native_ansatz
    from revision.steps_native_sweep import extend_parameters
    from revision.tensor_prefix_sabre import compile_isa_sabre
    edges = [(0,1),(1,2)]
    small = native_ansatz(3,edges,2)
    x = np.random.default_rng(5).normal(0,.2,small.num_parameters)
    deep = native_ansatz(3,edges,6)
    extended = extend_parameters(x,3,len(edges),2,6)
    assert abs(np.vdot(Statevector(small.assign_parameters(x)).data,
                       Statevector(deep.assign_parameters(extended)).data))**2 == pytest.approx(1.,abs=1e-11)
    parent = ROOT/'outputs/larger_width_feasibility_2026_10'
    patch = read(parent/'preflight/patch.json')
    target = pickle.loads((parent/'preflight/target.pkl').read_bytes())
    config = read(ROOT/'configs/flp42_depth_extension_2026_10.json')
    depths = []
    for layers in config['layers']:
        qc = native_ansatz(42,patch['edges'],layers)
        params = np.random.default_rng(layers).normal(0,.1,qc.num_parameters)
        _,res = compile_isa_sabre(qc.assign_parameters(params),target,
                                  patch['physical'],patch['edges'],seed=config['transpile_seed'])
        assert res['swap_count'] == 0
        assert res['cz_depth'] <= 200
        depths.append(res['cz_depth'])
    assert depths == [18,36,54,90,126,162,198]
