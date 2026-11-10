"""Contract FLP feasibility and low-cost events directly from an MPS.

The largest tensors are density matrices on MPS bonds, not 2**n vectors.
Facility masks are enumerated; each customer block projects onto an allowed
one-hot assignment. Cost pruning enumerates only potentially useful choices.
"""
import numpy as np

def site_arrays(state):
    psi=state.psi if hasattr(state,'psi') else state
    arrays=[]
    for q in range(psi.L):
        physical=psi.site_ind(q);order=[]
        if q:order.append(psi.bond(q-1,q))
        order.append(physical)
        if q+1<psi.L:order.append(psi.bond(q,q+1))
        a=np.asarray(psi[q].transpose(*order).data)
        if q==0:a=np.expand_dims(a,0)
        if q+1==psi.L:a=np.expand_dims(a,-1)
        assert a.ndim==3 and a.shape[1]==2
        arrays.append(a)
    return arrays,float(abs(psi.H@psi))

def one_hot_blocks(arrays):
    prefix=[np.eye(arrays[0].shape[0],dtype=complex)]
    for a in arrays:prefix.append(prefix[-1]@a[:,0,:])
    suffix=[None]*(len(arrays)+1);suffix[-1]=np.eye(arrays[-1].shape[-1],dtype=complex)
    for q in reversed(range(len(arrays))):suffix[q]=arrays[q][:,0,:]@suffix[q+1]
    return [prefix[q]@a[:,1,:]@suffix[q+1] for q,a in enumerate(arrays)]

def probabilities(model,state,gap=.05):
    f,c=model['facilities'],model['customers'];arrays,norm=site_arrays(state)
    assert len(arrays)==f*(c+1) and norm>0
    blocks=[one_hot_blocks(arrays[f+i*f:f+(i+1)*f]) for i in range(c)]
    opening=np.asarray(model['opening']);service=np.asarray(model['service']);limit=(1+gap)*model['optimum']+1e-9
    feasible=good=optimal=0.;feasible_count=good_count=optimal_count=0
    for mask in range(1,1<<f):
        opened=[j for j in range(f) if mask>>j&1];v=np.ones(1,dtype=complex)
        for q in range(f):v=v@arrays[q][:,(mask>>q)&1,:]
        rho=np.outer(v.conj(),v)
        for i in range(c):
            next_rho=np.zeros((blocks[i][0].shape[1],)*2,dtype=complex)
            for j in opened:
                b=blocks[i][j];next_rho+=b.conj().T@rho@b
            rho=next_rho
        assert rho.shape==(1,1)
        p=float(rho[0,0].real);assert p>=-1e-10 and abs(rho[0,0].imag)<1e-9
        feasible+=max(0.,p);feasible_count+=len(opened)**c
        lower=[0.]*(c+1)
        for i in reversed(range(c)):lower[i]=lower[i+1]+min(service[i,j] for j in opened)
        def visit(i,vector,cost):
            nonlocal good,optimal,good_count,optimal_count
            if cost+lower[i]>limit:return
            if i==c:
                probability=float(abs(vector[0])**2);good+=probability;good_count+=1
                if cost<=model['optimum']+1e-9:optimal+=probability;optimal_count+=1
                return
            for j in opened:
                newcost=cost+service[i,j]
                if newcost+lower[i+1]<=limit:visit(i+1,vector@blocks[i][j],newcost)
        visit(0,v,float(opening[opened].sum()))
    result=dict(p_feasible=feasible/norm,p_good=good/norm,p_optimal=optimal/norm,
        feasible_count=feasible_count,good_count=good_count,optimal_count=optimal_count,norm=norm)
    assert -1e-10<=result['p_good']<=result['p_feasible']+1e-9<=1+1e-8
    return result
