"""Cached p=1 continuation, independent numerical checks and prospective power."""
import time
import numpy as np

def energy_offset(model):
    return float(np.asarray(model['linear']).sum()/2+np.asarray(model['quadratic']).sum()/4+model['constant']/model['scale'])

class CachedEnergy:
    def __init__(self,prefix,model,bond=64,diagonal=None):
        from .full_circuit_compression import exact_state
        self.model=model;self.n=prefix.num_qubits;self.bond=bond
        if self.n<=20:
            from .prefix_diagnostic import energy_diagonal
            self.initial=exact_state(prefix);self.diagonal=energy_diagonal(model) if diagonal is None else diagonal
        else:
            from qiskit_aer import AerSimulator
            from qiskit_aer.library import save_matrix_product_state,set_matrix_product_state
            self.sim=AerSimulator(method='matrix_product_state',matrix_product_state_max_bond_dimension=bond,
                matrix_product_state_truncation_threshold=1e-12,max_parallel_threads=1,max_parallel_experiments=1)
            qc=prefix.copy();qc.save_matrix_product_state();result=self.sim.run(qc,shots=1).result();assert result.success
            self.initial=result.data(0)['matrix_product_state']
    def evaluate(self,angles):
        if self.n<=20:
            from .prefix_diagnostic import continue_state
            vector=continue_state(self.initial,self.diagonal,angles)
            return float((np.abs(vector)**2)@self.diagonal)
        from qiskit import QuantumCircuit
        from .tensor_prefix_core import qaoa
        from .corrected_study import ising_operator
        from .tensor_prefix_models import numpy_model
        qc=QuantumCircuit(self.n);qc.set_matrix_product_state(self.initial)
        qc.compose(qaoa(QuantumCircuit(self.n),self.model,angles),inplace=True)
        qc.save_expectation_value(ising_operator(numpy_model(self.model)),range(self.n),label='energy')
        result=self.sim.run(qc,shots=1).result();assert result.success
        return float(result.data(0)['energy'])+energy_offset(self.model)

def train_continuation(prefix,model,config,checkpoint=None,diagonal=None):
    from scipy.optimize import minimize
    from .flp42_fidelity_parallel import save,now
    started=time.perf_counter();history=[];cache=CachedEnergy(prefix,model,config['qaoa_train_bond'],diagonal)
    preparation_seconds=time.perf_counter()-started;best=dict(energy=float('inf'),angles=None)
    class BudgetStop(Exception):pass
    def objective(theta):
        if history and (len(history)>=config['qaoa_max_evaluations'] or time.perf_counter()-started>=config['qaoa_seconds']):raise BudgetStop()
        value=cache.evaluate(theta);assert np.isfinite(value)
        feasible=abs(float(theta[0]))<=.6+1e-9 and abs(float(theta[1]))<=.8+1e-9
        history.append(dict(angles=np.asarray(theta).tolist(),energy=value,within_bounds=feasible,seconds=time.perf_counter()-started))
        if feasible and value<best['energy']:best.update(energy=value,angles=np.asarray(theta).tolist())
        if checkpoint:save(checkpoint,dict(**best,evaluations=len(history),history=history,updated_utc=now()))
        return value
    try:
        result=minimize(objective,[.2,-.2],method='COBYLA',bounds=[(-.6,.6),(-.8,.8)],
                        options=dict(maxiter=config['qaoa_max_evaluations'],rhobeg=.12,tol=.002))
        status=str(result.message)
    except BudgetStop:status='budget_best_completed_evaluation'
    assert history and best['angles'] is not None
    return dict(**best,evaluations=len(history),history=history,status=status,seconds=time.perf_counter()-started,
                preparation_seconds=preparation_seconds,objective='mean normalized penalized energy',train_bond=None if prefix.num_qubits<=20 else config['qaoa_train_bond'])

def verify_circuit(qc,model,config,diagonal=None):
    from .flp42_fidelity_parallel import state
    from .tensor_prefix_core import fidelity,mps_energy
    from .flp_mps_events import probabilities as mps_probabilities
    if qc.num_qubits<=20:
        from .full_circuit_compression import exact_state
        from .prefix_diagnostic import feasible_assignments,probabilities,energy_diagonal
        exact=exact_state(qc);feasible,costs=feasible_assignments(model)
        if diagonal is None:diagonal=energy_diagonal(model)
        events=probabilities(exact,model,feasible,costs,diagonal)
        return dict(verified=True,verification='exact_statevector',events=events,energy=events['mean_penalized_energy'])
    low_bond,high_bond=config['qaoa_check_bonds']
    low=state(qc,low_bond,config['check_cutoff']);high=state(qc,high_bond,config['check_cutoff'])
    overlap=fidelity(low,high);a=mps_probabilities(model,low);b=mps_probabilities(model,high)
    differences={k:abs(a[k]-b[k])/max(b[k],1e-300) for k in ['p_feasible','p_good','p_optimal']}
    ea=mps_energy(qc,model,low_bond)+energy_offset(model);eb=mps_energy(qc,model,high_bond)+energy_offset(model)
    verified=overlap>=config['convergence_min'] and abs(ea-eb)<=config['qaoa_energy_tolerance'] and max(differences.values())<=config['event_relative_tolerance']
    return dict(verified=bool(verified),verification='independent MPS bond and event sensitivity, not exact certification',
        events=b,low_events=a,energy=eb,low_energy=ea,energy_bond_difference=abs(ea-eb),state_overlap=overlap,event_relative_differences=differences)

def exact_fisher_power(p_treatment,p_control,n,alpha=.05/6,tail=1e-10):
    """Independent binomials and the one-sided Fisher rejection region.

    Truncate only negligible binomial tails and report the excluded mass.
    Chunk the rejection matrix to keep memory bounded.
    """
    from scipy.stats import binom,hypergeom
    assert 0<=p_treatment<=1 and 0<=p_control<=1 and n>0 and 0<alpha<1
    def support(p):
        lo=int(binom.ppf(tail/4,n,p));hi=int(binom.ppf(1-tail/4,n,p));x=np.arange(lo,hi+1)
        return x,binom.pmf(x,n,p)
    a,pa=support(p_treatment);b,pb=support(p_control);power=0.
    for start in range(0,len(a),64):
        aa=a[start:start+64,None];pvalue=hypergeom.sf(aa-1,2*n,aa+b[None,:],n)
        power+=float(np.sum((pvalue<=alpha)*pa[start:start+64,None]*pb[None,:]))
    return dict(power_lower_bound=power,omitted_mass=max(0.,float(1-pa.sum()*pb.sum())),n_per_arm=n,alpha=alpha,
                expected_treatment=n*p_treatment,expected_control=n*p_control)
