import numpy as np


def test_nontrivial_identity_block_preserves_complex_preparation():
    from qiskit.quantum_info import Statevector
    from revision.tensor_prefix_core import native_ansatz
    from revision.native_depth_identity import identity_extension
    n=5;edges=[(0,1),(1,2),(2,3),(3,4),(0,4),(1,3)]
    old=native_ansatz(n,edges,2)
    x=np.random.default_rng(17).normal(0,.8,old.num_parameters)
    reference=Statevector(old.assign_parameters(x)).data
    for layers in [4,6,8,10]:
        expanded=identity_extension(x,n,len(edges),2,layers,31+layers)
        assert np.count_nonzero(expanded[len(x):])>0
        actual=Statevector(native_ansatz(n,edges,layers).assign_parameters(expanded)).data
        assert abs(np.vdot(reference,actual))**2>1-1e-12
