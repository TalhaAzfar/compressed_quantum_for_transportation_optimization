"""Local assignment-matrix correction of diagonal event indicators.

Matrices have rows=observed bit, columns=prepared bit. Qubit zero is the
least-significant bit. No quasi-probability is clipped or relabeled a sample.
"""
import numpy as np


def count_arrays(counts, n):
    values=np.array([int(k.replace(" ",""),2) for k in counts],dtype=np.int64)
    weights=np.array(list(counts.values()),dtype=np.int64)
    if (values<0).any() or (values>=1<<n).any() or (weights<0).any() or weights.sum()<=0:
        raise ValueError("Invalid counts")
    return values,weights


def assignment_matrices(zero,one,n):
    result=np.empty((n,2,2))
    for prepared,counts in enumerate([zero,one]):
        values,weights=count_arrays(counts,n)
        for q in range(n):
            p=float(weights@((values>>q)&1)/weights.sum())
            result[q,:,prepared]=[1-p,p]
    if np.min(np.abs(np.linalg.det(result)))<.1:
        raise ValueError("Readout calibration is too ill-conditioned")
    return result


def apply_local(vector, matrices):
    """Apply a Kronecker product without constructing a 2^n by 2^n matrix."""
    result=np.array(vector,dtype=float,copy=True)
    n=len(matrices)
    if result.shape[0]!=1<<n: raise ValueError("Dimension mismatch")
    trailing=result.shape[1:]
    for q,m in enumerate(matrices):
        view=result.reshape((-1,2,1<<q)+trailing)
        zero,one=view[:,0].copy(),view[:,1].copy()
        view[:,0]=m[0,0]*zero+m[0,1]*one
        view[:,1]=m[1,0]*zero+m[1,1]*one
    return result


def event_weights(indicators,matrices):
    return apply_local(indicators,np.linalg.inv(matrices).transpose(0,2,1))


def weighted_estimate(counts,weights):
    n=(len(weights)-1).bit_length()
    values,counts=count_arrays(counts,n)
    nshots=int(counts.sum()); selected=weights[values]
    mean=counts@selected/nshots
    second=counts@(selected*selected)/nshots
    variance=np.maximum(0.,second-mean*mean)/(nshots-1) if nshots>1 else np.full_like(mean,np.nan)
    return mean,np.sqrt(variance)


def resample_counts(counts,rng):
    keys=list(counts); values=np.array(list(counts.values()),dtype=int); shots=int(values.sum())
    draw=rng.multinomial(shots,values/shots)
    return {k:int(v) for k,v in zip(keys,draw) if v}


def bit_mapping(circuit,physical):
    """Indices in physical patch order for each returned classical bit."""
    measured={circuit.find_bit(inst.clbits[0]).index:circuit.find_bit(inst.qubits[0]).index
              for inst in circuit.data if inst.operation.name=="measure"}
    if set(measured)!=set(range(len(physical))): raise ValueError("Incomplete measurements")
    mapping=[physical.index(measured[q]) for q in range(len(physical))]
    if sorted(mapping)!=list(range(len(physical))): raise ValueError("Non-bijective measurements")
    return mapping
