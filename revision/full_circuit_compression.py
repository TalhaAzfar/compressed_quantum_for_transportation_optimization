"""Bounded full-state AQC-Tensor recompilation at fixed QAOA angles."""
import argparse
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/full_circuit_mitigation.json"
EXTRA = ""


def settings():
    c = json.loads(CONFIG.read_text())
    return c, ROOT/c["output_dir"]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")


def context():
    c, out = settings()
    parent = ROOT/c["parent_dir"]
    case = json.loads((parent/"hardware/flp20.json").read_text())
    rows = {r["method"]:r for r in case["methods"]}
    return c, out, parent, case, rows


def freeze():
    from .corrected_study import file_hash
    from .nighthawk_study import now
    c, out, parent, case, rows = context()
    names = [str(CONFIG.relative_to(ROOT)), "revision/full_circuit_compression.py",
             "revision/memory_guard.py", "revision/tensor_prefix_core.py", "revision/steps_native_sweep.py",
             "revision/tensor_prefix_sabre.py", "revision/prefix_diagnostic.py"]
    inputs = [parent/"hardware/flp20.json", parent/"preflight/target_snapshot.pkl"]
    inputs += [ROOT/rows[s]["logical_qpy"] for s in c["targets"]]
    record = dict(config=c, source_sha256={n:file_hash(ROOT/n) for n in names},
                  input_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in inputs})
    path = out/"protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert all(old[k] == record[k] for k in record), "Frozen inputs or preparation source changed"
    else:
        for name in names:
            dst=out/"source_snapshot"/Path(name).name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, dst)
        save(path, dict(frozen_utc=now(), previous_hardware_outcomes_known=True,
                        new_hardware_outcomes_seen=False, **record))
    for name in ["targets", "fits", "circuits", "logs"]: (out/name).mkdir(exist_ok=True)


def exact_state(qc):
    import numpy as np
    from qiskit_aer import AerSimulator
    sim=AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1)
    circuit=qc.copy(); circuit.save_statevector()
    result=sim.run(circuit,shots=1).result(); assert result.success
    return np.asarray(result.get_statevector(), dtype=complex)


def ideal_metrics(state, model):
    from .prefix_diagnostic import feasible_assignments, energy_diagonal, probabilities
    feasible, costs = feasible_assignments(model)
    return probabilities(state,model,feasible,costs,energy_diagonal(model))


def mps_dense_qiskit(state,n):
    import numpy as np
    # Quimb site zero is the leftmost tensor index; Qiskit q0 is the LSB.
    return np.asarray(state.psi.to_dense()).reshape([2]*n).transpose(list(reversed(range(n)))).reshape(-1)


def target_worker(source):
    import numpy as np
    from .nighthawk_study import qpy_read
    from .tensor_prefix_core import mps_state, fidelity
    c,out,parent,case,rows=context()
    qc=qpy_read(ROOT/rows[source]["logical_qpy"])
    exact=exact_state(qc); np.save(out/"targets"/(source+".npy"),exact)
    record=dict(source=source, ideal=ideal_metrics(exact,case["model"]),convergence=[])
    previous=None; chosen=None; start=time.perf_counter()
    for bond in c["target_bonds"]:
        state=mps_state(qc,bond)
        dense=mps_dense_qiskit(state,qc.num_qubits)
        f=float(abs(np.vdot(exact,dense))**2/np.vdot(dense,dense).real)
        row=dict(bond=bond,exact_fidelity=f,achieved_bond=state.psi.max_bond())
        if previous is not None: row["successive_fidelity"]=fidelity(previous,state)
        record["convergence"].append(row); print(source,row,flush=True)
        if f>=c["convergence_min"] and previous is not None and row["successive_fidelity"]>=c["convergence_min"]:
            chosen=state; break
        previous=state
    record.update(status="converged" if chosen is not None else "unconverged",seconds=time.perf_counter()-start)
    if chosen is not None: (out/"targets"/(source+".pkl")).write_bytes(pickle.dumps(chosen))
    save(out/"targets"/(source+".json"),record)


def fit_worker(source,layers,index):
    import numpy as np
    from .steps_native_sweep import fit,extend_parameters
    from .tensor_prefix_core import native_ansatz, mps_state
    from .tensor_prefix_sabre import compile_isa_sabre
    from .nighthawk_study import qpy_write
    from .corrected_study import file_hash
    c,out,parent,case,rows=context()
    folder=out/"fits"/source; folder.mkdir(exist_ok=True)
    n=case["model"]["n_qubits"]; edges=case["patch"]["edges"]
    ansatz=native_ansatz(n,edges,layers)
    prior=[json.loads(p.read_text()) for p in folder.glob("l*_r*.json")]
    prior=[r for r in prior if r.get("verified") and r["layers"]<layers]
    if prior and index==0:
        depth=max(r["layers"] for r in prior)
        old=max([r for r in prior if r["layers"]==depth],key=lambda r:r["exact_fidelity"])
        x0=extend_parameters(old["parameters"],n,len(edges),depth,layers)
        initialization="identity_extension"
    else:
        x0=np.random.default_rng(c["fit_seeds"][index]).normal(0,.02 if index==0 else .2,ansatz.num_parameters)
        initialization="random_normal"
    target=pickle.loads((out/"targets"/(source+".pkl")).read_bytes())
    bound,row=fit(target,ansatz,x0,c)
    exact=exact_state(bound); reference=np.load(out/"targets"/(source+".npy"))
    row["exact_fidelity"]=float(abs(np.vdot(reference,exact))**2)
    check=mps_state(bound,c["ansatz_check_bond"])
    dense=mps_dense_qiskit(check,n)
    row["ansatz_exact_check"]=float(abs(np.vdot(exact,dense))**2/np.vdot(dense,dense).real)
    row["verified"]=bool(row["verified"] and row["ansatz_exact_check"]>=c["convergence_min"])
    row["ideal"]=ideal_metrics(exact,case["model"])
    backend=pickle.loads((parent/"preflight/target_snapshot.pkl").read_bytes())
    isa,resources=compile_isa_sabre(bound,backend,case["patch"]["physical"],edges)
    assert resources["swap_count"]==0 and resources["cz_count"]==2*len(edges)*layers
    name=f"{source}_l{layers}_r{index}"
    logical=out/"circuits"/(name+".qpy"); hardware=out/"circuits"/(name+"_isa.qpy")
    qpy_write(logical,bound); qpy_write(hardware,isa)
    row.update(source=source,layers=layers,index=index,seed=c["fit_seeds"][index],initialization=initialization,
        resources=resources,logical_qpy=str(logical.relative_to(ROOT)),logical_sha256=file_hash(logical),
        isa_qpy=str(hardware.relative_to(ROOT)),isa_sha256=file_hash(hardware))
    save(folder/f"l{layers}_r{index}.json",row)
    print(name,"fidelity",row["exact_fidelity"],"feasible",row["ideal"]["p_feasible"],"CZ",resources["cz_count"],flush=True)


def run():
    from .memory_guard import guarded_run
    freeze(); c,out=settings()
    def launch(action,args,name,seconds):
        command=[sys.executable,"-m","revision.local_runner","--extra-site",EXTRA,
                 "revision.full_circuit_compression",action,*map(str,args)]
        result=guarded_run(command,out/"logs"/(name+".log"),address_gib=c["worker_address_space_gib"],
            rss_gib=c["worker_rss_gib"],reserve_gib=c["minimum_available_ram_gib"],seconds=seconds)
        save(out/"logs"/(name+"_process.json"),result)
        if result["status"]=="system_memory_reserve": raise RuntimeError(result)
        return result
    for source in c["targets"]:
        path=out/"targets"/(source+".json")
        if not path.exists():
            process=launch("target",[source],source+"_target",c["target_seconds_cap"])
            if not path.exists(): save(path,dict(status="worker_failed",process=process))
        if json.loads(path.read_text())["status"]!="converged": continue
        for layers in c["layers"]:
            for index in range(len(c["fit_seeds"])):
                path=out/"fits"/source/f"l{layers}_r{index}.json"
                if path.exists(): continue
                process=launch("fit",[source,layers,index],f"{source}_l{layers}_r{index}",c["fit_seconds_cap"]+90)
                if not path.exists(): save(path,dict(source=source,layers=layers,index=index,verified=False,status="worker_failed",process=process))
                r=json.loads(path.read_text()); print(source,layers,index,r.get("exact_fidelity"),r["status"],flush=True)
    save(out/"preparation_complete.json",dict(targets=c["targets"],protocol_sha256=__import__("hashlib").sha256((out/"protocol.json").read_bytes()).hexdigest()))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["run","target","fit"])
    p.add_argument("source",nargs="?");p.add_argument("layers",type=int,nargs="?");p.add_argument("index",type=int,nargs="?")
    a=p.parse_args()
    if a.action=="run":run()
    elif a.action=="target":target_worker(a.source)
    else:fit_worker(a.source,a.layers,a.index)
