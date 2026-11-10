"""Guarded numerical-backend timing before the depth-refinement protocol."""
import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from .contribution_hardware_core import ROOT, read, save, now

PARENT = ROOT/'outputs/hardware_width_sweep_2026'
OUT = ROOT/'outputs/native_depth_refinement_2026_10'


def load_target(cid):
    import numpy as np
    from .hardware_width_sweep import exact_state, dense_target
    from .nighthawk_study import qpy_read
    from .full_circuit_compression import mps_dense_qiskit
    case = read(PARENT/'cases'/cid/'complete.json')
    model = case['model']; n = model['n_qubits']
    folder = OUT/'targets'/cid; folder.mkdir(parents=True, exist_ok=True)
    vector = folder/'reference.npy'; mps = folder/'target.pkl'
    if vector.exists() and mps.exists():
        return case, np.load(vector, mmap_mode='r'), pickle.loads(mps.read_bytes())
    key = hashlib.sha256(json.dumps(dict(model=model, steps=12, duration=2.0), sort_keys=True).encode()).hexdigest()
    cache = Path('/tmp/aqc_width_sweep_state_cache')/key
    # The old configuration stored integer 2; accept only a matching model hash.
    if not cache.exists():
        key = hashlib.sha256(json.dumps(dict(model=model, steps=12, duration=2), sort_keys=True).encode()).hexdigest()
        cache = Path('/tmp/aqc_width_sweep_state_cache')/key
    reference = np.load(cache/'reference.npy') if (cache/'reference.npy').exists() else exact_state(qpy_read(PARENT/'cases'/cid/'source.qpy'))
    np.save(vector, reference)
    state = pickle.loads((cache/'target256.pkl').read_bytes()) if (cache/'target256.pkl').exists() else dense_target(reference, n, 256)
    state._psi /= np.sqrt(float(abs(state.psi.H@state.psi)))
    dense = mps_dense_qiskit(state, n)
    fidelity = float(abs(np.vdot(reference, dense))**2)
    assert fidelity >= .999, fidelity
    mps.write_bytes(pickle.dumps(state))
    save(folder/'validation.json', dict(created_utc=now(), target_exact_fidelity=fidelity,
         source='same complete 12-step annealing state as the frozen width sweep',
         construction='exact-reference-assisted MPS', bond=256))
    return case, reference, state


def worker(cid, layers, backend):
    import numpy as np
    import quimb.tensor as qtn
    from functools import partial
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    from .tensor_prefix_core import native_ansatz
    from .steps_native_sweep import extend_parameters
    t0 = time.monotonic(); print('loading', cid, layers, backend, flush=True)
    case, reference, target = load_target(cid)
    n = case['model']['n_qubits']; ansatz = native_ansatz(n, case['edges'], layers)
    prior = max([r for r in case['rows'] if r['fit'] and r['fit']['layers'] <= layers], key=lambda r:r['fit']['exact_fidelity'])['fit']
    x = extend_parameters(prior['parameters'], n, len(case['edges']), prior['layers'], layers)
    settings = QuimbSimulator(partial(qtn.CircuitMPS, max_bond=128, cutoff=1e-12), autodiff_backend=backend)
    print('creating objective', time.monotonic()-t0, flush=True)
    diagnostic = {}
    if backend == 'bounded':
        from .native_depth_objective import make_objective
        objective, diagnostic = make_objective(target, ansatz)
        print('bounded contraction', {k:v for k,v in diagnostic.items() if k not in ['path','sliced_indices']}, flush=True)
    else:
        objective = MaximizeStateFidelity(target, ansatz, settings)
    if backend not in ['explicit', 'bounded']:
        import cotengra as ctg
        from qiskit_addon_aqc_tensor.simulation.quimb import maximize_state_fidelity_loss_function
        context = objective._preprocessed
        # Convert fixed target arrays to the same autodiff backend as the gates.
        context.tnopt = qtn.TNOptimizer(context.quimb_ansatz,
            loss_fn=maximize_state_fidelity_loss_function,
            loss_constants={'target': target.psi}, autodiff_backend=backend, progbar=False)
        network = target.psi.H & objective._preprocessed.quimb_ansatz.psi
        optimizer = ctg.ReusableHyperOptimizer(methods=['greedy'], max_repeats=16, parallel=False, minimize='combo')
        tree = network.contraction_tree(optimize=optimizer)
        diagnostic = dict(largest_elements=int(tree.max_size()), total_write_elements=float(tree.total_write()), contraction_flops=float(tree.total_flops()))
        print('contraction estimate', diagnostic, flush=True)
        save(OUT/'benchmarks'/f'{cid}_l{layers}_{backend}_path.json', diagnostic)
        # Reverse-mode differentiation retains intermediates; use conservative headroom.
        if tree.max_size()*16 > 1e9 or tree.total_write()*16 > 2e9:
            raise RuntimeError('Contraction exceeds the diagnostic memory allowance')
        qtn.set_contract_strategy(optimizer)
    times = []; losses = []
    for i in range(3):
        start = time.monotonic(); loss, grad = objective.loss_function(x)
        assert np.isfinite(loss) and np.isfinite(grad).all()
        times.append(time.monotonic()-start); losses.append(float(loss))
        print('evaluation', i+1, 'seconds', times[-1], 'fidelity', 1-float(loss), flush=True)
        x = x - 1e-4*np.asarray(grad)
    save(OUT/'benchmarks'/f'{cid}_l{layers}_{backend}.json', dict(case_id=cid, layers=layers, backend=backend,
         timings=times, losses=losses, total_seconds=time.monotonic()-t0, contraction=diagnostic))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('case'); p.add_argument('layers', type=int); p.add_argument('backend'); a = p.parse_args()
    worker(a.case, a.layers, a.backend)
