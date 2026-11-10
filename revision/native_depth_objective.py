"""Memory-bounded contraction gradients for the AQC-Tensor fidelity objective.

The public AQC-Tensor parameter conversion and objective interface are retained.
Only the contraction path and the storage of sliced reverse-mode intermediates
are controlled here. Slices are summed as complex amplitudes before fidelity.
"""
import time


def streamed_overlap(tree):
    import torch

    class SlicedOverlap(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *arrays):
            ctx.save_for_backward(*arrays)
            value = None
            for i in range(tree.nslices):
                term = tree.contract_slice(arrays, i, backend='torch')
                value = term if value is None else value + term
            return value

        @staticmethod
        def backward(ctx, output_gradient):
            arrays = ctx.saved_tensors
            wanted = [i for i, needed in enumerate(ctx.needs_input_grad) if needed]
            accumulated = [None]*len(arrays)
            for slice_id in range(tree.nslices):
                with torch.enable_grad():
                    local = [a.detach().requires_grad_(i in wanted) for i, a in enumerate(arrays)]
                    value = tree.contract_slice(local, slice_id, backend='torch')
                    gradients = torch.autograd.grad(value, [local[i] for i in wanted],
                        grad_outputs=output_gradient, allow_unused=True)
                for index, grad in zip(wanted, gradients):
                    if grad is not None:
                        grad = grad.detach()
                        if accumulated[index] is None: accumulated[index] = grad
                        else: accumulated[index].add_(grad)
                del local, value, gradients
            return tuple(accumulated)

    return SlicedOverlap.apply


def make_objective(target, ansatz, *, repeats=16, max_tensor_bytes=268435456,
                   max_slice_write_bytes=1500000000, max_slices=1024, seed=100101,
                   minimum_slices=1):
    import numpy as np
    import torch
    import quimb.tensor as qtn
    import cotengra as ctg
    from functools import partial
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    started = time.monotonic()
    settings = QuimbSimulator(partial(qtn.CircuitMPS, max_bond=128, cutoff=1e-12), autodiff_backend='torch')
    objective = MaximizeStateFidelity(target, ansatz, settings)
    context = objective._preprocessed
    network = target.psi.H & context.quimb_ansatz.psi
    optimizer = ctg.HyperOptimizer(methods=['greedy'], max_repeats=repeats,
                                  parallel=False, minimize='combo')
    tree = network.contraction_tree(optimize=optimizer)
    original = dict(max_tensor_elements=int(tree.max_size()), total_write_elements=float(tree.total_write()),
                    contraction_flops=float(tree.total_flops()))
    if minimum_slices > 1:
        tree.slice_(target_slices=minimum_slices, allow_outer=False, seed=seed)
    size = min(int(tree.max_size()), int(max_tensor_bytes//16))
    while tree.max_size()*16 > max_tensor_bytes or tree.total_write()/tree.nslices*16 > max_slice_write_bytes:
        # Split binary circuit indices. Slicing a target MPS bond of dimension
        # 128/256 in one step can needlessly create tens of thousands of tasks.
        finder = ctg.SliceFinder(tree, target_size=max(2,size), allow_outer=False,
                                seed=seed, minimize='combo')
        finder.forbidden.update(ix for ix,d in tree.size_dict.items() if d > 2)
        indices, _ = finder.search(max_repeats=8)
        for ix in indices: tree.remove_ind_(ix)
        if tree.nslices > max_slices: raise MemoryError('Required slicing exceeds the declared contraction budget')
        size //= 2
        if size < 2: raise MemoryError('Cannot meet the contraction memory estimate')
    if tree.nslices > max_slices: raise MemoryError('Required slicing exceeds the declared contraction budget')
    signature = tuple(tuple(a.shape) for a in network.arrays)
    streamed = streamed_overlap(tree) if tree.nslices > 1 else None

    def loss_function(circuit, target):
        arrays = (target.H & circuit.psi).arrays
        assert tuple(tuple(a.shape) for a in arrays) == signature
        overlap = streamed(*arrays) if streamed is not None else tree.contract(arrays, backend='torch')
        return 1 - torch.abs(overlap)**2

    context.tnopt = qtn.TNOptimizer(context.quimb_ansatz, loss_fn=loss_function,
         loss_constants={'target': target.psi}, autodiff_backend='torch', progbar=False)
    diagnostic = dict(backend='AQC-Tensor Quimb/Torch contraction gradient', setup_seconds=time.monotonic()-started,
         target_mps_bond=target.psi.max_bond(), original=original, slices=int(tree.nslices),
         largest_tensor_bytes=int(tree.max_size())*16,
         estimated_write_bytes_per_slice=float(tree.total_write()/tree.nslices)*16,
         contraction_flops=float(tree.total_flops()), streamed_reverse_mode=bool(streamed),
         path=[list(x) for x in tree.get_path()], sliced_indices={str(k):str(v) for k,v in tree.sliced_inds.items()},
         note='No fitting-state MPS truncation; contractions target the fixed validated MPS. All complex slice amplitudes are summed before squaring.')
    return objective, diagnostic
