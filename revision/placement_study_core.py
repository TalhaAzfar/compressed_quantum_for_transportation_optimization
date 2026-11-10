"""Native placement ablations with unchanged logical targets and decoding."""
import collections
import time
import numpy as np

def distances(n,edges):
    adjacency=[[] for _ in range(n)]
    for u,v in edges:adjacency[u].append(v);adjacency[v].append(u)
    result=np.full((n,n),n,dtype=int)
    for s in range(n):
        result[s,s]=0;queue=collections.deque([s])
        while queue:
            u=queue.popleft()
            for v in adjacency[u]:
                if result[s,v]==n:result[s,v]=result[s,u]+1;queue.append(v)
    assert result.max()<n
    return result

def interaction_weights(model):
    q=np.asarray(model['quadratic'],float);w=np.abs(q+q.T);np.fill_diagonal(w,0.)
    return w

def placement_cost(weights,distance,logical_to_wire):
    p=np.asarray(logical_to_wire,int)
    return float(np.sum(weights*distance[np.ix_(p,p)])/2)

def optimize_placement(model,edges,seed=100001,restarts=8,proposals=4000):
    started=time.perf_counter();n=model['n_qubits'];w=interaction_weights(model);d=distances(n,edges)
    rng=np.random.default_rng(seed);best=np.arange(n);best_cost=placement_cost(w,d,best);initial=best_cost
    for restart in range(restarts):
        p=np.arange(n) if restart==0 else rng.permutation(n);cost=placement_cost(w,d,p)
        for k in range(proposals):
            u,v=rng.choice(n,2,replace=False);a,b=p[u],p[v]
            delta=float(np.dot(w[u]-w[v],d[b,p]-d[a,p])+2*w[u,v]*d[a,b])
            temperature=max(.002,1.5*(1-k/proposals)**2)
            if delta<=0 or rng.random()<np.exp(-delta/temperature):
                p[u],p[v]=b,a;cost+=delta
                if cost<best_cost-1e-10:best=p.copy();best_cost=cost
        assert abs(cost-placement_cost(w,d,p))<1e-8
    assert best_cost<=initial+1e-8
    return best.tolist(),dict(initial_cost=initial,cost=best_cost,seconds=time.perf_counter()-started,
                             seed=seed,restarts=restarts,proposals_per_restart=proposals)

def mapped_patch(patch,logical_to_wire):
    p=list(logical_to_wire);n=len(p);assert sorted(p)==list(range(n))
    inverse=np.argsort(p).tolist()
    edges=sorted(tuple(sorted((inverse[u],inverse[v]))) for u,v in patch['edges'])
    return dict(physical=[patch['physical'][p[q]] for q in range(n)],edges=[list(e) for e in edges],
                logical_to_wire=p,wire_to_logical=inverse)

def native_euler_ansatz(n,edges,layers):
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from .native_aqc import edge_matchings
    theta=ParameterVector('a',3*n*(layers+1)+len(edges)*layers);qc=QuantumCircuit(n);qc.h(range(n));k=0
    for layer in range(layers+1):
        for q in range(n):
            qc.rz(theta[k],q);qc.ry(theta[k+1],q);qc.rz(theta[k+2],q);k+=3
        if layer<layers:
            for matching in edge_matchings(edges):
                for u,v in matching:qc.cx(u,v);qc.rz(theta[k],v);qc.cx(u,v);k+=1
    assert k==len(theta)
    return qc

def projected_initialization(model,edges,layers,endpoint=.5,duration=2.):
    """Exact embedding of short native-projected annealing, up to global phase.

    All original Ising fields are retained. Only nonnative ZZ terms are
    omitted in this INITIALIZER; the fitting target always stays unchanged.
    """
    from .native_aqc import edge_matchings
    n=model['n_qubits'];q=np.asarray(model['quadratic']);linear=np.asarray(model['linear'])
    h=-linear/2-(q.sum(0)+q.sum(1))/4;j=(q+q.T)/4
    dt=duration*endpoint/layers;values=[]
    for layer in range(layers):
        s=endpoint*(layer+.5)/layers
        for site in range(n):values.extend([np.pi/2,-2*dt*(1-s),-np.pi/2+2*dt*s*h[site]])
        for matching in edge_matchings(edges):
            for u,v in matching:values.append(2*dt*s*j[u,v])
    values.extend([0.]*(3*n))
    return np.asarray(values)

def projected_circuit(model,edges,layers,endpoint=.5,duration=2.):
    """Independent gate construction for initializer equivalence tests."""
    from qiskit import QuantumCircuit
    n=model['n_qubits'];q=np.asarray(model['quadratic']);linear=np.asarray(model['linear'])
    h=-linear/2-(q.sum(0)+q.sum(1))/4;j=(q+q.T)/4;dt=duration*endpoint/layers
    qc=QuantumCircuit(n);qc.h(range(n))
    for layer in range(layers):
        s=endpoint*(layer+.5)/layers;qc.rx(-2*dt*(1-s),range(n))
        for site in range(n):qc.rz(2*dt*s*h[site],site)
        for u,v in edges:qc.rzz(2*dt*s*j[u,v],u,v)
    return qc

def design_ansatz(model,patch,architecture,layers):
    from .tensor_prefix_core import native_ansatz
    return (native_ansatz if architecture=='ryrz' else native_euler_ansatz)(model['n_qubits'],patch['edges'],layers)

def initialize_parameters(model,patch,architecture,initialization,layers,seed,start):
    ansatz=design_ansatz(model,patch,architecture,layers);rng=np.random.default_rng(seed+start)
    if initialization=='random':return rng.normal(0,.02,ansatz.num_parameters)
    assert architecture=='euler' and initialization=='projected'
    x=projected_initialization(model,patch['edges'],layers)
    return x if start==0 else x+rng.normal(0,.005,len(x))

def placement_summary(model,patch):
    w=interaction_weights(model);e={tuple(sorted(x)) for x in patch['edges']};n=len(w)
    pairs=[(u,v) for u in range(n) for v in range(u+1,n) if w[u,v]>0]
    total=sum(w[u,v] for u,v in pairs);d=distances(n,patch['edges'])
    return dict(interaction_pairs=len(pairs),direct_pairs=sum((u,v) in e for u,v in pairs),
                direct_weight_fraction=sum(w[u,v] for u,v in pairs if (u,v) in e)/total,
                weighted_distance=sum(w[u,v]*d[u,v] for u,v in pairs)/total)
