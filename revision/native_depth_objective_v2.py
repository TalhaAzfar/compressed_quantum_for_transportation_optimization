"""Choose contraction trees by bounded execution cost, before fitting begins.

The fidelity objective and streamed derivative are unchanged. Unlike v1,
candidate trees are ranked after slicing, and excessively sliced trees are
rejected before evaluating them. Every search trial is recorded.
"""
import random
import time
from .native_depth_objective import streamed_overlap


def bounded_tree(tree, max_tensor_bytes, max_slice_write_bytes, max_slices, seed):
    import cotengra as ctg
    tree=tree.copy();size=min(int(tree.max_size()),int(max_tensor_bytes//16))
    while tree.max_size()*16>max_tensor_bytes or tree.total_write()/tree.nslices*16>max_slice_write_bytes:
        finder=ctg.SliceFinder(tree,target_size=max(2,size),allow_outer=False,seed=seed,minimize='combo')
        finder.forbidden.update(ix for ix,d in tree.size_dict.items() if d>2)
        remaining=max_slices//int(tree.nslices)
        if remaining<2:raise MemoryError('Slice budget cannot satisfy storage bounds')
        for _ in range(8):finder.trial(target_slices=remaining)
        valid=[(ix,cost) for ix,cost in finder.costs.items() if cost.size<=max(2,size) and cost.nslices<=remaining]
        if not valid:raise MemoryError('Slice budget cannot satisfy the next storage target')
        indices,_=min(valid,key=lambda x:(x[1].total_flops,x[1].nslices,x[1].size))
        for ix in indices:tree.remove_ind_(ix)
        if tree.nslices>max_slices:raise MemoryError('Excessive slice count rejected before evaluation')
        size//=2
        if size<2:raise MemoryError('Cannot satisfy contraction storage bounds')
    return tree


def make_objective(target,ansatz,*,repeats=16,max_tensor_bytes=268435456,
                   max_slice_write_bytes=1500000000,max_slices=1024,seed=100101,
                   minimum_slices=1,warm_path_file=None):
    import numpy as np
    import torch
    import quimb.tensor as qtn
    import cotengra as ctg
    from functools import partial
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    started=time.monotonic();random.seed(seed);np.random.seed(seed)
    simulator=QuimbSimulator(partial(qtn.CircuitMPS,max_bond=128,cutoff=1e-12),autodiff_backend='torch')
    objective=MaximizeStateFidelity(target,ansatz,simulator);context=objective._preprocessed
    network=target.psi.H & context.quimb_ansatz.psi
    trials=[];candidates=[];slice_cap=min(32,max_slices)
    # Reuse an already validated topology when tensor count matches. A path is
    # only a contraction order: it cannot change the represented state. Fresh
    # index names and dimensions come from the current network.
    from .native_depth_benchmark import OUT
    from .contribution_hardware_core import read,digest
    warm_paths=[warm_path_file] if warm_path_file else [OUT/'benchmarks'/'tsp25_r1_l8_bounded.json',
        OUT/'cases'/'flp20_r1'/'d64_contraction.json']
    warm_hashes={}
    for warm_path in warm_paths:
        if not warm_path.exists():continue
        loaded=read(warm_path);warm=loaded.get('contraction',loaded)
        if len(warm['path'])==len(network.tensors)-1:
            warm_hashes[str(warm_path)]=digest(warm_path)
            original=network.contraction_tree(optimize=[tuple(x) for x in warm['path']])
            for trial_seed in [100101,seed]:
                record=dict(source='archived benchmark path',seed=trial_seed)
                try:
                    tree=bounded_tree(original,max_tensor_bytes,max_slice_write_bytes,slice_cap,trial_seed)
                    score=float(tree.total_flops())*(3 if tree.nslices>1 else 2)+64*float(tree.total_write())
                    score+=1e6*tree.nslices*len(network.tensors)
                    record.update(accepted=True,slices=int(tree.nslices),flops=float(tree.total_flops()),
                        write_bytes_per_slice=float(tree.total_write()/tree.nslices)*16,
                        largest_tensor_bytes=int(tree.max_size())*16,execution_score=score)
                    candidates.append((score,tree,len(trials)))
                except MemoryError as error:record.update(accepted=False,reason=str(error))
                trials.append(record)
    class Recorder(ctg.HyperOptimizer):
        def _maybe_report_result(self,setting,trial):
            super()._maybe_report_result(setting,trial)
            original=trial.get('tree')
            if original is None:return
            record=dict(original_max_tensor_elements=int(original.max_size()),
                original_write_elements=float(original.total_write()),original_flops=float(original.total_flops()))
            try:
                tree=bounded_tree(original,max_tensor_bytes,max_slice_write_bytes,slice_cap,seed+len(trials))
                if minimum_slices>1:tree.slice_(target_slices=minimum_slices,allow_outer=False,seed=seed)
                # Reverse-mode work plus a storage/dispatch proxy; purely a
                # numerical execution choice, independent of objective values.
                score=float(tree.total_flops())*(3 if tree.nslices>1 else 2)+64*float(tree.total_write())
                score+=1e6*tree.nslices*len(network.tensors)
                record.update(accepted=True,slices=int(tree.nslices),flops=float(tree.total_flops()),
                    write_bytes_per_slice=float(tree.total_write()/tree.nslices)*16,
                    largest_tensor_bytes=int(tree.max_size())*16,execution_score=score)
                candidates.append((score,tree,len(trials)))
            except MemoryError as error:record.update(accepted=False,reason=str(error))
            trials.append(record)
    # Search up to three batches. An inexpensive first-batch plan is enough;
    # difficult networks receive more planning before any gradient evaluation.
    for batch in range(3):
        if candidates and min(candidates,key=lambda x:x[0])[1].nslices<=8:break
        optimizer=Recorder(methods=['greedy'],max_repeats=repeats,parallel=False,minimize='combo',
            optlib='random',seed=seed+batch,reconf_opts={'seed':seed+batch})
        network.contraction_tree(optimize=optimizer)
        if candidates and min(candidates,key=lambda x:x[0])[1].nslices<=8:break
    if not candidates:raise MemoryError('No adequately bounded contraction tree found in three planning batches')
    _,tree,selected=min(candidates,key=lambda x:x[0])
    signature=tuple(tuple(a.shape) for a in network.arrays)
    streamed=streamed_overlap(tree) if tree.nslices>1 else None
    def loss_function(circuit,target):
        arrays=(target.H & circuit.psi).arrays
        assert tuple(tuple(a.shape) for a in arrays)==signature
        overlap=streamed(*arrays) if streamed is not None else tree.contract(arrays,backend='torch')
        return 1-torch.abs(overlap)**2
    context.tnopt=qtn.TNOptimizer(context.quimb_ansatz,loss_fn=loss_function,
        loss_constants={'target':target.psi},autodiff_backend='torch',progbar=False)
    diagnostic=dict(backend='AQC-Tensor Quimb/Torch; execution-aware contraction planning v2',
        setup_seconds=time.monotonic()-started,target_mps_bond=target.psi.max_bond(),slices=int(tree.nslices),
        largest_tensor_bytes=int(tree.max_size())*16,
        estimated_write_bytes_per_slice=float(tree.total_write()/tree.nslices)*16,
        contraction_flops=float(tree.total_flops()),streamed_reverse_mode=bool(streamed),
        selected_trial=selected,trials=trials,planner_seed=seed,archived_benchmark_path_sha256=warm_hashes or None,
        path=[list(x) for x in tree.get_path()],
        input_indices=[list(t.inds) for t in network.tensors],size_dict=dict(network.ind_sizes()),
        sliced_indices=list(tree.sliced_inds),array_shapes=[list(s) for s in signature],
        note='Same complete-state fidelity under the supplied storage bounds. Candidate paths are ranked after binary slicing; at most 32 slices are admitted before fitting. No fitting-state MPS truncation.')
    return objective,diagnostic
