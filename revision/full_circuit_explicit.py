"""Pre-hardware amendment: explicit MPS gradients for deeper full-state fits."""
import argparse
import json
import pickle
import signal
import sys
import time

from .full_circuit_compression import ROOT,EXTRA,context,settings,save,exact_state,mps_dense_qiskit,ideal_metrics


def explicit_settings(bond=128,cutoff=1e-10):
    from functools import partial
    import quimb.tensor as qtn
    from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
    return QuimbSimulator(partial(qtn.CircuitMPS,max_bond=bond,cutoff=cutoff),autodiff_backend="explicit")


def worker(source,layers,index):
    import numpy as np
    from scipy.optimize import minimize
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from .steps_native_sweep import extend_parameters,alarm_handler,BudgetExpired
    from .tensor_prefix_core import native_ansatz,mps_state,fidelity
    from .tensor_prefix_sabre import compile_isa_sabre
    from .nighthawk_study import qpy_write
    from .corrected_study import file_hash
    c,out,parent,case,rows=context();n=case["model"]["n_qubits"];edges=case["patch"]["edges"]
    previous=[json.loads(p.read_text()) for p in (out/"fits"/source).glob("*.json")]
    previous += [json.loads(p.read_text()) for p in (out/"explicit_fits"/source).glob("*.json")]
    previous=[r for r in previous if r.get("verified") and r["layers"]<layers]
    old=max(previous,key=lambda r:r["exact_fidelity"])
    ansatz=native_ansatz(n,edges,layers)
    x0=extend_parameters(old["parameters"],n,len(edges),old["layers"],layers)
    if index: x0+=np.random.default_rng(94102).normal(0,.02,len(x0))
    target=pickle.loads((out/"targets"/(source+".pkl")).read_bytes())
    target._psi/=np.sqrt(float(abs(target.psi.H@target.psi)))
    objective=MaximizeStateFidelity(target,ansatz,explicit_settings(128))
    best=dict(loss=float("inf"),parameters=x0);history=[];start=time.perf_counter()
    def fun(theta):
        loss,gradient=objective.loss_function(theta);loss=float(loss);gradient=np.asarray(gradient,float)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):raise ValueError("Nonfinite MPS gradient")
        history.append(loss)
        if loss<best["loss"]:best.update(loss=loss,parameters=np.array(theta))
        return loss,gradient
    signal.signal(signal.SIGALRM,alarm_handler);signal.alarm(90)
    try:
        result=minimize(fun,x0,jac=True,method="L-BFGS-B",options=dict(maxiter=120,maxls=12,ftol=1e-10))
        status=str(result.message);iterations=int(result.nit)
    except BudgetExpired:status="wall_time_cap_best_completed_evaluation";iterations=None
    finally:signal.alarm(0)
    assert history,"No MPS-gradient evaluation completed"
    bound=ansatz.assign_parameters(best["parameters"])
    exact=exact_state(bound);reference=np.load(out/"targets"/(source+".npy"))
    target_dense=mps_dense_qiskit(target,n);target_dense/=np.linalg.norm(target_dense)
    f=float(abs(np.vdot(reference,exact))**2);fmps=float(abs(np.vdot(target_dense,exact))**2)
    low=mps_state(bound,128);high=mps_state(bound,256)
    convergence=fidelity(low,high);difference=abs(1-best["loss"]-fmps)
    dense=mps_dense_qiskit(high,n)
    exactcheck=float(abs(np.vdot(exact,dense))**2/np.vdot(dense,dense).real)
    verified=convergence>=c["convergence_min"] and exactcheck>=c["convergence_min"] and difference<2e-5
    backend=pickle.loads((parent/"preflight/target_snapshot.pkl").read_bytes())
    isa,resources=compile_isa_sabre(bound,backend,case["patch"]["physical"],edges)
    assert resources["swap_count"]==0 and resources["cz_count"]==2*len(edges)*layers
    name=f"explicit_{source}_l{layers}_r{index}"
    logical=out/"circuits"/(name+".qpy");hardware=out/"circuits"/(name+"_isa.qpy")
    qpy_write(logical,bound);qpy_write(hardware,isa)
    row=dict(source=source,layers=layers,index=index,backend="aqc_tensor_explicit_mps_gradient",ansatz_bond=128,
        exact_fidelity=f,fidelity=fmps,verified=bool(verified),objective_check_difference=difference,
        ansatz_bond_convergence=convergence,ansatz_exact_check=exactcheck,
        status=status,iterations=iterations,evaluations=len(history),loss_history=history,
        parameters=best["parameters"].tolist(),seconds=time.perf_counter()-start,resources=resources,
        ideal=ideal_metrics(exact,case["model"]),logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=file_hash(logical),
        isa_qpy=str(hardware.relative_to(ROOT)),isa_sha256=file_hash(hardware),
        initialization="extension_of_best_shallower_fit" if index==0 else "perturbed_extension_of_best_shallower_fit")
    save(out/"explicit_fits"/source/f"l{layers}_r{index}.json",row)
    print(name,"F",f,"verified",verified,"objective mismatch",difference,flush=True)


def run():
    from .corrected_study import file_hash
    from .nighthawk_study import now
    from .memory_guard import guarded_run
    c,out=settings();assert (out/"preparation_complete.json").exists()
    assert not (out/"jobs.json").exists(),"Do not amend fitting after seeing new hardware"
    source="revision/full_circuit_explicit.py"
    record=dict(source_sha256=file_hash(ROOT/source),layers=[4,5],starts=2,bond=128,check_bond=256,
        fit_seconds_cap=90,maxiter=120,targets=c["targets"],
        amendment="Original automatic-differentiation contraction requested an oversized intermediate at layers 4/5 and was stopped by the hard cap. Before hardware, add equal-budget explicit AQC-Tensor MPS-gradient fits for both sources at both depths. Keep all initial failures. Hardware selection includes all verified fits, using the originally fixed exact-fidelity ranking and layer budgets.")
    p=out/"explicit_gradient_amendment.json"
    if p.exists():
        old=json.loads(p.read_text());assert all(old[k]==v for k,v in record.items())
    else:
        save(p,dict(frozen_utc=now(),new_hardware_outcomes_seen=False,**record))
        (out/"source_snapshot/full_circuit_explicit.py").write_bytes((ROOT/source).read_bytes())
    for source in c["targets"]:
        for layers in [4,5]:
            for index in range(2):
                path=out/"explicit_fits"/source/f"l{layers}_r{index}.json"
                if path.exists():continue
                name=f"explicit_{source}_l{layers}_r{index}"
                cmd=[sys.executable,"-m","revision.local_runner","--extra-site",EXTRA,
                     "revision.full_circuit_explicit","worker",source,str(layers),str(index)]
                result=guarded_run(cmd,out/"logs"/(name+".log"),address_gib=4,rss_gib=3,reserve_gib=12,seconds=180)
                save(out/"logs"/(name+"_process.json"),result)
                if result["status"]=="system_memory_reserve":raise RuntimeError(result)
                if not path.exists():save(path,dict(source=source,layers=layers,index=index,verified=False,status="worker_failed",process=result))
                row=json.loads(path.read_text());print(name,row.get("exact_fidelity"),row["status"],flush=True)
    save(out/"explicit_complete.json",dict(completed_utc=now(),amendment_sha256=file_hash(p)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["run","worker"])
    p.add_argument("source",nargs="?");p.add_argument("layers",nargs="?",type=int);p.add_argument("index",nargs="?",type=int)
    a=p.parse_args()
    if a.action=="run":run()
    else:worker(a.source,a.layers,a.index)
