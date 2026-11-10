"""Prepare the bounded larger-instance AQC-Tensor/QAOA pilot offline."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import pickle
import resource
import shutil
import time
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from .corrected_study import ROOT, file_hash, write_json
from .nighthawk_study import now, qpy_read, qpy_write
from .tensor_prefix_models import make_model
from .tensor_prefix_core import (path_patch, target_circuit, mps_state, fidelity, native_ansatz,
                                fit_prefix, train_qaoa, compile_isa)

CONFIG = ROOT/"configs/aqc_tensor_qaoa_v1.json"


def settings():
    config = json.loads(CONFIG.read_text())
    return config, ROOT/config["output_dir"]


def freeze():
    config, out = settings()
    for folder in ["cases","circuits","source_snapshot","targets"]:
        (out/folder).mkdir(exist_ok=True)
    names = [str(CONFIG.relative_to(ROOT)), "revision/tensor_prefix_models.py", "revision/tensor_prefix_core.py",
             "revision/tensor_prefix_prepare.py", "revision/nighthawk_study.py", "revision/corrected_study.py",
             "revision/native_aqc.py"]
    hashes = {p:file_hash(ROOT/p) for p in names}
    path = out/"protocol.json"
    if path.exists():
        existing = json.loads(path.read_text())
        assert hashes == existing["source_sha256"], "Frozen preparation sources changed"
        assert file_hash(out/"preflight/target_snapshot.pkl") == existing["target_sha256"]
        return
    for name in names:
        shutil.copy2(ROOT/name,out/"source_snapshot"/Path(name).name)
    write_json(path,dict(frozen_utc=now(),config=config,source_sha256=hashes,
                         target_sha256=file_hash(out/"preflight/target_snapshot.pkl"),hardware_outcomes_seen=False,
                         preparation_amendments=["preflight/attempt_2/PREPARATION_AMENDMENT.md", "preflight/attempt_3/PREPARATION_AMENDMENT.md"],
                         packages={p:importlib.metadata.version(p) for p in ["qiskit","qiskit-aer","qiskit-addon-aqc-tensor","quimb","jax","scipy"]}))


def prepare(case_id):
    freeze()
    config, out = settings()
    spec = next(s for s in config["cases"] if s["id"] == case_id)
    path = out/"cases"/(case_id+".json")
    if path.exists():
        print("Already completed", case_id, flush=True); return
    start = time.perf_counter()
    model = make_model(spec)
    target = pickle.loads((out/"preflight/target_snapshot.pkl").read_bytes())
    physical, edges, patch = path_patch(target, model["n_qubits"])
    patch.update(physical=physical, edges=edges)
    candidates, methods, target_records = [], {}, {}
    def save_method(name, logical, extra):
        isa, resources = compile_isa(logical,target,physical,edges)
        logical_file, isa_file = out/"circuits"/f"{case_id}_{name}_logical.qpy", out/"circuits"/f"{case_id}_{name}_isa.qpy"
        qpy_write(logical_file,logical); qpy_write(isa_file,isa)
        methods[name] = dict(method=name,logical_qpy=str(logical_file.relative_to(ROOT)),
            isa_qpy=str(isa_file.relative_to(ROOT)),logical_sha256=file_hash(logical_file),
            isa_sha256=file_hash(isa_file),resources=resources,**extra)
        print(case_id,name,"CZ depth",resources["cz_depth"],flush=True)
    def trained(name,prefix,p=1):
        path=out/"cases"/f"{case_id}_{name}_training.json"
        from .tensor_prefix_core import qaoa
        if path.exists():
            record=json.loads(path.read_text())
            assert record["prefix_sha256"]==file_hash(out/"circuits"/f"{case_id}_{name}_prefix.qpy")
            return qaoa(prefix,model,record["parameters"]),record
        circuit,record=train_qaoa(prefix,model,p=p,maxeval=config["qaoa_train_maxeval"],bond=config["qaoa_train_bond"])
        record["prefix_sha256"]=file_hash(out/"circuits"/f"{case_id}_{name}_prefix.qpy")
        write_json(path,record)
        return circuit,record
    for endpoint in config["target_endpoints"]:
        tag = "half" if endpoint == .5 else "full"
        refined = target_circuit(model,endpoint,config["target_steps"],config["target_duration"])
        low, chosen, convergence = None, None, []
        t = time.perf_counter()
        for bond in config["target_bonds"]:
            state = mps_state(refined,bond)
            if low is not None:
                f = fidelity(low,state)
                convergence.append(dict(low_bond=previous,high_bond=bond,fidelity=f))
                if f >= config["target_convergence_fidelity_minimum"]:
                    chosen = state; break
            low, previous = state,bond
        if chosen is None:
            target_records[tag] = dict(endpoint=endpoint,status="excluded_unconverged",convergence=convergence,
                                       seconds=time.perf_counter()-t)
            print(case_id,tag,"EXCLUDED: unconverged MPS",convergence,flush=True)
            continue
        target_records[tag] = dict(endpoint=endpoint,selected_bond=bond,achieved_bond=chosen.psi.max_bond(),
            norm=float(abs(chosen.psi.H@chosen.psi)),convergence=convergence,seconds=time.perf_counter()-t)
        (out/"targets"/f"{case_id}_{tag}.pkl").write_bytes(pickle.dumps(chosen))
        qpy_write(out/"circuits"/f"{case_id}_{tag}_target.qpy",refined)
        print(case_id,tag,"target",target_records[tag],flush=True)
        endpoint_candidates = []
        for architecture in config["prefix_candidates"]:
            graph,layers = architecture["graph"],architecture["layers"]
            name = f"{tag}_{graph}_l{layers}"
            cache = out/"cases"/f"{case_id}_{name}_fit.json"
            circuit_file = out/"circuits"/f"{case_id}_{name}_fit.qpy"
            if cache.exists():
                record = json.loads(cache.read_text()); compiled=qpy_read(circuit_file)
                _, resources = compile_isa(compiled,target,physical,edges)
                record.update(resources=resources,compiler_protocol_sha256=file_hash(out/"protocol.json"))
                write_json(cache,record)
            else:
                active_edges = [(i,i+1) for i in range(model["n_qubits"]-1)] if graph=="path" else edges
                compiled,record = fit_prefix(chosen,native_ansatz(model["n_qubits"],active_edges,layers),
                    bond=bond,maxiter=config["fit_maxiter"],seed=config["fit_seed"])
                _, resources = compile_isa(compiled,target,physical,edges)
                record.update(name=name,graph=graph,layers=layers,endpoint=endpoint,resources=resources,
                              completed_utc=now(),source_protocol_sha256=file_hash(out/"protocol.json"))
                qpy_write(circuit_file,compiled);write_json(cache,record)
            endpoint_candidates.append((record,compiled))
            candidates.append(record)
            print(case_id,name,"fidelity",record["fidelity"],"fit sec",record["seconds"],flush=True)
        qualified = [p for p in endpoint_candidates if p[0]["fidelity"] >= config["fit_target_fidelity"]]
        if qualified:
            best,prep = min(qualified,key=lambda p:(p[0]["resources"]["cz_depth"],-p[0]["fidelity"]))
        else:
            best,prep = max(endpoint_candidates,key=lambda p:(p[0]["fidelity"],-p[0]["resources"]["cz_depth"]))
        save_method(f"aqc_{tag}_prefix",prep,dict(endpoint=endpoint,selected_fit=best["name"],fidelity_threshold_met=bool(qualified)))
        for name,prefix in [(f"aqc_{tag}",prep),(f"shallow_{tag}",target_circuit(model,endpoint,2,config["target_duration"])),(f"refined_{tag}",refined)]:
            qpy_write(out/"circuits"/f"{case_id}_{name}_prefix.qpy",prefix)
            circuit,training = trained(name,prefix)
            save_method(name,circuit,dict(endpoint=endpoint,training=training,
                prefix_qpy=str((out/"circuits"/f"{case_id}_{name}_prefix.qpy").relative_to(ROOT))))
    if not any(name.startswith("aqc_") for name in methods):
        raise ValueError(f"No converged endpoint for {case_id}; no hardware candidate may be prepared")
    plus = QuantumCircuit(model["n_qubits"]);plus.h(range(model["n_qubits"]))
    for p in [1,2]:
        prefix_file=out/"circuits"/f"{case_id}_standard_p{p}_prefix.qpy";qpy_write(prefix_file,plus)
        circuit,training=trained(f"standard_p{p}",plus,p=p)
        save_method(f"standard_p{p}",circuit,dict(training=training,prefix_qpy=str(prefix_file.relative_to(ROOT))))
    write_json(path,dict(completed_utc=now(),model=model,patch=patch,targets=target_records,
                        candidates=candidates,methods=methods,preparation_sec=time.perf_counter()-start,
                        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    print("Completed",case_id,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    prepare(parser.parse_args().case)
