"""Direct Aer MPS references and independently testable tensor conversion."""
import numpy as np

def from_aer_mps(data):
    import quimb.tensor as qtn
    from types import SimpleNamespace
    gammas,lambdas=data;n=len(gammas);assert n>=2 and len(lambdas)==n-1
    arrays=[]
    for q,gamma in enumerate(gammas):
        a=np.stack([np.asarray(gamma[0]),np.asarray(gamma[1])],axis=1)
        if q<n-1:a=a*np.asarray(lambdas[q])[None,None,:]
        if q==0:a=a[0,:,:]
        if q==n-1:a=a[:,:,0]
        arrays.append(a)
    return SimpleNamespace(psi=qtn.MatrixProductState(arrays,shape='lpr'))

def reference(qc,model,bond,cutoff=1e-13):
    from qiskit_aer import AerSimulator
    from qiskit_aer.library import save_matrix_product_state
    from .corrected_study import ising_operator
    from .tensor_prefix_models import numpy_model
    from .placement_qaoa_core import energy_offset
    assert qc.num_qubits==model['n_qubits']
    sim=AerSimulator(method='matrix_product_state',matrix_product_state_max_bond_dimension=bond,
        matrix_product_state_truncation_threshold=cutoff,max_parallel_threads=1,max_parallel_experiments=1)
    saved=qc.copy();saved.save_matrix_product_state()
    saved.save_expectation_value(ising_operator(numpy_model(model)),range(qc.num_qubits),label='energy')
    result=sim.run(saved,shots=1).result();assert result.success
    data=result.data(0);state=from_aer_mps(data['matrix_product_state']);norm=float(abs(state.psi.H@state.psi))
    assert abs(norm-1)<1e-7,norm
    return state,float(data['energy'])+energy_offset(model)

def verify(qc,model,config):
    from .tensor_prefix_core import fidelity
    from .flp_mps_events import probabilities
    low,ea=reference(qc,model,128,config['check_cutoff']);high,eb=reference(qc,model,256,config['check_cutoff'])
    overlap=fidelity(low,high);a=probabilities(model,low);b=probabilities(model,high)
    differences={key:abs(a[key]-b[key])/max(b[key],1e-300) for key in ['p_feasible','p_good','p_optimal']}
    ok=overlap>=config['convergence_min'] and abs(ea-eb)<=config['qaoa_energy_tolerance'] and max(differences.values())<=config['event_relative_tolerance']
    return dict(verified=bool(ok),verification='Secondary direct Aer bond-128/256 check with independently tested MPS conversion; not an exact certificate',
        events=b,low_events=a,energy=eb,low_energy=ea,state_overlap=overlap,energy_bond_difference=abs(ea-eb),event_relative_differences=differences)
