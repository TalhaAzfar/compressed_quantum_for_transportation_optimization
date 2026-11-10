"""Frozen fixed-duration 8/10/12-step targets and native AQC depth sweeps."""
import argparse
import concurrent.futures
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import resource
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
from scipy.optimize import minimize
from .corrected_study import ROOT, file_hash, write_json
from .nighthawk_study import now, qpy_read, qpy_write
from .tensor_prefix_core import target_circuit, native_ansatz, mps_state, fidelity, simulator
from .tensor_prefix_sabre import compile_isa_sabre

CONFIG = ROOT/"configs/aqc_steps_10_12.json"


def settings():
    config = json.loads(CONFIG.read_text())
    return config, ROOT/config["output_dir"]


def task_id(cid, steps, endpoint):
    return f"{cid}_s{steps}_{'half' if endpoint == .5 else 'full'}"


def freeze():
    config, out = settings()
    for name in ["preflight", "source_snapshot", "tasks", "logs", "circuits", "targets"]:
        (out/name).mkdir(parents=True, exist_ok=True)
    parent = ROOT/config["parent_dir"]
    names = [str(CONFIG.relative_to(ROOT)), "revision/steps_native_sweep.py", "revision/tensor_prefix_core.py",
             "revision/tensor_prefix_models.py", "revision/tensor_prefix_sabre.py", "revision/native_aqc.py",
             "revision/nighthawk_study.py", "revision/corrected_study.py", "tests/test_steps_native.py"]
    hashes = {n: file_hash(ROOT/n) for n in names}
    parents = {cid: file_hash(parent/"cases"/(cid+".json")) for cid in config["cases"]}
    path = out/"protocol.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old["source_sha256"] == hashes, "Frozen sweep code changed"
        assert old["parent_case_sha256"] == parents
        assert old["target_sha256"] == file_hash(out/"preflight/target_snapshot.pkl")
        return
    # Same saved device graph/physical patches isolate step-count effects.
    # Live compatibility and calibration are checked again before new hardware.
    for name in ["target_snapshot.pkl", "backend_snapshot.json", "calibration.json"]:
        shutil.copy2(parent/"preflight"/name, out/"preflight"/name)
    for name in names:
        shutil.copy2(ROOT/name, out/"source_snapshot"/Path(name).name)
    write_json(path, dict(frozen_utc=now(), config=config, source_sha256=hashes,
        parent_case_sha256=parents, target_sha256=file_hash(out/"preflight/target_snapshot.pkl"),
        prior_hardware_outcomes_known=True, new_hardware_outcomes_seen=False,
        packages={p: importlib.metadata.version(p) for p in ["qiskit", "qiskit-addon-aqc-tensor", "quimb", "jax", "scipy"]}))


def extend_parameters(old, n, edge_count, previous_layers, layers):
    """Insert identity blocks after the preceding circuit, preserving its state."""
    assert layers >= previous_layers
    expected = previous_layers*(2*n+edge_count)+2*n
    assert len(old) == expected
    result = np.zeros(layers*(2*n+edge_count)+2*n)
    result[:len(old)] = old
    return result


class BudgetExpired(Exception):
    pass


def alarm_handler(signum, frame):
    raise BudgetExpired("Declared wall-time cap reached")


def fit(target, ansatz, x0, config):
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    normalized = target.copy()
    normalized._psi /= np.sqrt(float(abs(normalized.psi.H@normalized.psi)))
    objective = MaximizeStateFidelity(normalized, ansatz, simulator(config["ansatz_bond"]))
    start = time.perf_counter()
    history = []
    best = {"loss": float("inf"), "parameters": np.array(x0)}
    def fun(theta):
        loss, gradient = objective.loss_function(theta)
        loss, gradient = float(loss), np.asarray(gradient, dtype=float)
        if not np.isfinite(loss) or not np.all(np.isfinite(gradient)):
            raise ValueError("Nonfinite fit objective")
        history.append(loss)
        if loss < best["loss"]:
            best.update(loss=loss, parameters=np.array(theta))
        return loss, gradient
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(config["fit_seconds_cap"])
    try:
        result = minimize(fun, x0, jac=True, method="L-BFGS-B",
                          options={"maxiter": config["fit_maxiter"], "maxls": 12, "ftol": 1e-10})
        status, iterations = str(result.message), int(result.nit)
    except BudgetExpired:
        status, iterations = "wall_time_cap_best_completed_evaluation", None
    finally:
        signal.alarm(0)
    assert history, "No objective evaluation completed within budget"
    bound = ansatz.assign_parameters(best["parameters"])
    low = mps_state(bound, config["ansatz_bond"])
    high = mps_state(bound, config["ansatz_check_bond"])
    f_low, f_high = fidelity(target, low), fidelity(target, high)
    check = fidelity(low, high)
    objective_difference = abs(1-best["loss"]-f_high)
    verified = check >= config["convergence_min"] and objective_difference < 2e-5
    return bound, dict(parameters=best["parameters"].tolist(), fidelity=f_high,
        low_bond_fidelity=f_low, ansatz_bond_convergence=check, objective_check_difference=objective_difference,
        verified=verified, status=status, iterations=iterations, evaluations=len(history),
        initial_fidelity=1-history[0], loss_history=history, seconds=time.perf_counter()-start)


def prepare(cid, steps, endpoint):
    freeze()
    config, out = settings()
    tid = task_id(cid, steps, endpoint)
    folder = out/"tasks"/tid
    folder.mkdir(exist_ok=True)
    result_path = folder/"result.json"
    if result_path.exists():
        print(tid, "already completed", flush=True)
        return
    start = time.perf_counter()
    parent = json.loads((ROOT/config["parent_dir"]/"cases"/(cid+".json")).read_text())
    model, patch = parent["model"], parent["patch"]
    n, edges = model["n_qubits"], [tuple(e) for e in patch["edges"]]
    target_path = out/"targets"/(tid+".pkl")
    target_qpy = out/"circuits"/(tid+"_target.qpy")
    record = dict(task_id=tid, case_id=cid, steps=steps, endpoint=endpoint, duration=config["duration"],
                  simulated_duration=config["duration"]*endpoint, model=model, patch=patch,
                  started_utc=now(), protocol_sha256=file_hash(out/"protocol.json"))
    target_record_path = folder/"target.json"
    if target_record_path.exists():
        tr = json.loads(target_record_path.read_text())
    else:
        qc = target_circuit(model, endpoint, steps, config["duration"])
        qpy_write(target_qpy, qc)
        previous = None
        convergence = []
        state = None
        t = time.perf_counter()
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(config["target_seconds_cap"])
        try:
            for bond in config["target_bonds"]:
                current = mps_state(qc, bond)
                if previous is not None:
                    f = fidelity(previous, current)
                    convergence.append(dict(low_bond=old_bond, high_bond=bond, fidelity=f))
                    if f >= config["convergence_min"]:
                        state = current
                        break
                previous, old_bond = current, bond
            status = "converged" if state is not None else "excluded_unconverged"
        except BudgetExpired:
            status = "excluded_target_time_cap"
        finally:
            signal.alarm(0)
        tr = dict(status=status, convergence=convergence, seconds=time.perf_counter()-t,
                  target_qpy=str(target_qpy.relative_to(ROOT)), target_qpy_sha256=file_hash(target_qpy))
        if state is not None:
            target_path.write_bytes(pickle.dumps(state))
            tr.update(selected_bond=bond, achieved_bond=state.psi.max_bond(),
                      target_pickle_sha256=file_hash(target_path), norm=float(abs(state.psi.H@state.psi)))
        write_json(target_record_path, tr)
    record["target"] = tr
    print(tid, "target", tr["status"], tr["convergence"], flush=True)
    fits = []
    if tr["status"] == "converged":
        assert file_hash(target_path) == tr["target_pickle_sha256"]
        target = pickle.loads(target_path.read_bytes())
        backend_target = pickle.loads((out/"preflight/target_snapshot.pkl").read_bytes())
        previous_best, previous_layers = None, 0
        for layers in config["layers"]:
            ansatz = native_ansatz(n, edges if layers else [], layers)
            layer_fits = []
            for index, seed in enumerate(config["fit_seeds"]):
                name = f"l{layers}_r{index}"
                fit_path = folder/(name+".json")
                if fit_path.exists():
                    row = json.loads(fit_path.read_text())
                else:
                    rng = np.random.default_rng(seed)
                    if index == 0 and previous_best is not None:
                        x0 = extend_parameters(previous_best["parameters"], n, len(edges), previous_layers, layers)
                        initialization = "identity_extension_of_previous_depth_best"
                    else:
                        x0 = rng.normal(0., .02 if index < 2 else .2, ansatz.num_parameters)
                        initialization = "random_normal_0.02" if index < 2 else "random_normal_0.2"
                    bound, row = fit(target, ansatz, x0, config)
                    logical_path = out/"circuits"/(tid+"_"+name+".qpy")
                    isa_path = out/"circuits"/(tid+"_"+name+"_isa.qpy")
                    isa, resources = compile_isa_sabre(bound, backend_target, patch["physical"], edges)
                    assert resources["swap_count"] == 0, "Native ansatz must not route"
                    qpy_write(logical_path, bound)
                    qpy_write(isa_path, isa)
                    row.update(name=name, layers=layers, seed=seed, initialization=initialization,
                        parameters_count=ansatz.num_parameters, resources=resources, completed_utc=now(),
                        logical_qpy=str(logical_path.relative_to(ROOT)), logical_sha256=file_hash(logical_path),
                        isa_qpy=str(isa_path.relative_to(ROOT)), isa_sha256=file_hash(isa_path))
                    write_json(fit_path, row)
                fits.append(row)
                layer_fits.append(row)
                print(tid, name, "fidelity", round(row["fidelity"], 8), "verified", row["verified"],
                      "CZ depth", row["resources"]["cz_depth"], "seconds", round(row["seconds"], 1), flush=True)
            acceptable = [r for r in layer_fits if r["verified"]]
            if acceptable:
                previous_best = max(acceptable, key=lambda r: r["fidelity"])
                previous_layers = layers
    record.update(fits=fits, completed_utc=now(), seconds=time.perf_counter()-start,
                  peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    write_json(result_path, record)
    print(tid, "COMPLETE", flush=True)


def run():
    freeze()
    config, out = settings()
    jobs = [(cid, steps, endpoint) for steps in config["steps"] for endpoint in config["endpoints"] for cid in config["cases"]]
    extra = ""
    def launch(job):
        cid, steps, endpoint = job
        tid = task_id(*job)
        cmd = [sys.executable, "-m", "revision.local_runner", "--extra-site", extra,
               "revision.steps_native_sweep", "prepare", cid, str(steps), str(endpoint)]
        start = time.perf_counter()
        with (out/"logs"/(tid+".log")).open("a") as log:
            try:
                result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=config["task_seconds_cap"], env={**os.environ, "OPENBLAS_NUM_THREADS":"1", "OMP_NUM_THREADS":"1"})
                status = result.returncode
            except subprocess.TimeoutExpired:
                status = "task_time_cap"
        item = dict(task_id=tid, exit_status=status, elapsed_seconds=time.perf_counter()-start, ended_utc=now())
        write_json(out/"tasks"/tid/"process.json", item)
        print(json.dumps(item), flush=True)
        return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=config["parallel_workers"]) as executor:
        results = list(executor.map(launch, jobs))
    write_json(out/"sweep_processes.json", results)
    print("All target processes finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "prepare"])
    parser.add_argument("case", nargs="?")
    parser.add_argument("steps", nargs="?", type=int)
    parser.add_argument("endpoint", nargs="?", type=float)
    args = parser.parse_args()
    if args.action == "freeze": freeze()
    elif args.action == "run": run()
    else: prepare(args.case, args.steps, args.endpoint)
