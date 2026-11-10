"""Frozen native-connectivity validation, with separate prepare/submit/collect.

Preparation is offline using an archived live IBM target. Submission is explicit
and idempotent at block level; job IDs are persisted immediately after submission.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import pickle
import resource
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from . import native_aqc
from .corrected_study import (ROOT, anneal, classical_baselines, file_hash,
    ising_operator, make_qaoa_circuit, metrics, qaoa_state, write_csv, write_json)


def now():
    return datetime.now(timezone.utc).isoformat()


def instance(customers, facilities, seed, config):
    """O(n*2**n) exact model enumeration without a dense basis-by-variable matrix."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, customers, facilities]))
    slo, shi = config["service_cost_range_inclusive"]
    olo, ohi = config["opening_cost_range_inclusive"]
    service = rng.integers(slo, shi+1, (customers, facilities)).astype(float)
    opening = rng.integers(olo, ohi+1, facilities).astype(float)
    n = facilities * (customers+1)
    penalty = 2. * (service.sum()+opening.sum())
    linear, quadratic = np.zeros(n), np.zeros((n, n))
    linear[:facilities] = opening
    for i in range(customers):
        indices = [facilities+i*facilities+j for j in range(facilities)]
        for j, q in enumerate(indices):
            linear[q] = service[i, j]
            quadratic[j, q] = -penalty
        for a, u in enumerate(indices):
            for v in indices[a+1:]:
                quadratic[u, v] = 2*penalty
    scale = max(1., abs(linear).max(), abs(quadratic).max())
    basis = np.arange(2**n, dtype=np.uint32)
    costs, violations = np.zeros(2**n), np.zeros(2**n)
    feasible = np.ones(2**n, dtype=bool)
    for j in range(facilities):
        costs += ((basis >> j) & 1) * opening[j]
    for i in range(customers):
        assigned = np.zeros(2**n, dtype=np.int8)
        for j in range(facilities):
            bit = ((basis >> (facilities+i*facilities+j)) & 1).astype(np.int8)
            assigned += bit
            costs += bit * service[i, j]
            violations += bit * (1-((basis >> j) & 1))
        violations += (1-assigned)**2
        feasible &= assigned == 1
    feasible &= violations == 0
    energy = (costs + penalty*(violations-customers)) / scale
    optimum = min(float(opening[[j for j in range(facilities) if mask & (1 << j)]].sum()
        + service[:, [j for j in range(facilities) if mask & (1 << j)]].min(axis=1).sum())
        for mask in range(1, 2**facilities))
    good = feasible & (costs <= optimum*(1+config["primary_gap_tolerance"])+1e-10)
    optimal = feasible & np.isclose(costs, optimum, rtol=0, atol=1e-10)
    ground = np.isclose(energy, energy.min(), rtol=0, atol=1e-10)
    assert np.all(optimal[ground])
    assert np.isclose(costs[feasible].min(), optimum)
    return dict(instance_id=f"flp_c{customers}_f{facilities}_seed{seed}",
        customers=customers, facilities=facilities, seed=seed, n_qubits=n,
        service=service, opening=opening, penalty=penalty, scale=float(scale),
        linear=linear/scale, quadratic=quadratic/scale, energy=energy,
        feasible=feasible, good=good, optimal=optimal, costs=costs, optimum=optimum,
        gap_tolerance=config["primary_gap_tolerance"])


def evolution(ins, duration, steps, start=0., end=1., prepare=True):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    qc = QuantumCircuit(ins["n_qubits"])
    if prepare:
        qc.h(range(qc.num_qubits))
    hc = ising_operator(ins)
    dt = duration*(end-start)/steps
    for k in range(steps):
        s = start+(end-start)*(k+.5)/steps
        qc.rx(-2*dt*(1-s), range(qc.num_qubits))
        qc.append(PauliEvolutionGate(hc, time=float(dt*s)), range(qc.num_qubits))
    return qc.decompose(reps=2)


def qpy_write(path, qc):
    from qiskit import qpy
    with Path(path).open("wb") as f:
        qpy.dump(qc, f)


def qpy_read(path):
    from qiskit import qpy
    with Path(path).open("rb") as f:
        return qpy.load(f)[0]


def transpile_and_check(qc, target, physical, edges, seed, native=False):
    """Confine every circuit to the same patch; preserve classical bit decoding."""
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    n = qc.num_qubits
    measured = qc.copy()
    measured.measure_all()
    cmap = [list(e) for u, v in edges for e in [(u, v), (v, u)]]
    local = transpile(measured, basis_gates=["cz", "rz", "sx", "x"], coupling_map=cmap,
        initial_layout=list(range(n)), routing_method="none" if native else "sabre",
        optimization_level=3, seed_transpiler=seed)
    for instruction in local.data:
        if instruction.operation.num_qubits == 2:
            edge = tuple(sorted(local.find_bit(q).index for q in instruction.qubits))
            assert edge in edges, ("Illegal local edge", edge)
    mapping = local.layout.final_index_layout() if local.layout else list(range(n))
    # Exact verification excludes idle physical wires and applies the final layout.
    if n <= 14:
        expected = Statevector.from_instruction(qc).data
        actual = Statevector.from_instruction(local.remove_final_measurements(inplace=False)).data
        basis = np.arange(2**n, dtype=np.uint32)
        mapped = np.zeros(2**n, dtype=np.uint32)
        for q, p in enumerate(mapping):
            mapped |= ((basis >> q) & 1) << p
        check_fidelity = float(abs(np.vdot(expected, actual[mapped]))**2)
        assert np.isclose(check_fidelity, 1., atol=1e-8), check_fidelity
    else:
        check_fidelity = None
    # Local circuit has already been routed. Embed it without any further routing.
    isa = transpile(local, target=target, initial_layout=physical, routing_method="none",
                    optimization_level=0, seed_transpiler=seed)
    allowed = {tuple(sorted(e)) for e in target["cz"]}
    for instruction in isa.data:
        if instruction.operation.num_qubits == 2:
            edge = tuple(sorted(isa.find_bit(q).index for q in instruction.qubits))
            assert edge in allowed and all(p in physical for p in edge)
    finish = np.zeros(target.num_qubits)
    for instruction in isa.data:
        name = instruction.operation.name
        qubits = tuple(isa.find_bit(q).index for q in instruction.qubits)
        if name == "barrier":
            continue
        props = target[name].get(qubits)
        duration = props.duration if props and props.duration else 0.
        finish[list(qubits)] = max(finish[list(qubits)], default=0.)+duration
    return isa, dict(cz_count=int(isa.count_ops().get("cz", 0)),
        two_qubit_depth=int(isa.depth(lambda instruction: instruction.operation.num_qubits == 2)),
        total_depth=int(isa.depth()), estimated_gate_critical_path_sec=float(finish.max()),
        transpilation_fidelity=check_fidelity, patch_only=True,
        routing_disabled=native, final_local_layout=mapping)


def prepare(config_path, only=None):
    config = json.loads(Path(config_path).read_text())
    out = ROOT/config["output_dir"]
    for d in [out/"cases", out/"circuits", out/"source_snapshot"]:
        d.mkdir(parents=True, exist_ok=True)
    with (out/"target_snapshot.pkl").open("rb") as f:
        target = pickle.load(f)
    protocol_path = out/"protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text())["config"] != config:
            raise ValueError("Frozen protocol differs; use a new output directory")
    else:
        write_json(protocol_path, dict(frozen_utc=now(), config=config,
            config_sha256=file_hash(config_path), backend_snapshot_sha256=file_hash(out/"backend_snapshot.json"),
            source_sha256={name: file_hash(ROOT/name) for name in
                ["revision/nighthawk_study.py", "revision/native_aqc.py", "revision/corrected_study.py"]},
            packages={p: importlib.metadata.version(p) for p in ["qiskit", "qiskit-ibm-runtime", "numpy", "scipy"]},
            independence="Distinct seeds and SeedSequence([seed,customers,facilities]); new cases independent of prior pilot",
            hardware_outcomes_seen=False))
        for source in [Path(config_path), Path(__file__), ROOT/"revision/native_aqc.py", ROOT/"revision/corrected_study.py"]:
            shutil.copy2(source, out/"source_snapshot"/source.name)
    for group in config["cases"]:
        for seed in group["seeds"]:
            case_id = f"flp_c{group['customers']}_f{group['facilities']}_seed{seed}"
            if only and only != case_id:
                continue
            destination = out/"cases"/f"{case_id}.json"
            if destination.exists():
                print("Already prepared", case_id, flush=True)
                continue
            print("Preparing", case_id, flush=True)
            start = time.perf_counter()
            ins = instance(group["customers"], group["facilities"], seed, config)
            n, T, fraction = ins["n_qubits"], config["annealing_time"], config["prefix_fraction"]
            physical, edges, selection = native_aqc.select_patch(target, n)
            target_start = time.perf_counter()
            prefix_state = anneal(ins, T, config["target_prefix_steps"], end_fraction=fraction)
            full_state = anneal(ins, T, config["tail_steps"], state=prefix_state, start_fraction=fraction)
            target_time = time.perf_counter()-target_start
            tail = evolution(ins, T, config["tail_steps"], fraction, 1., prepare=False)
            circuits = {
                "coarse_uncompressed": evolution(ins, T, config["num_steps"]),
                "refined_target": evolution(ins, T, config["target_prefix_steps"], end=fraction).compose(tail),
                "shallow_unoptimized": evolution(ins, T, config["shallow_prefix_steps"], end=fraction).compose(tail),
            }
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
            uniform = QuantumCircuit(n)
            uniform.h(range(n))
            circuits["uniform_bitstrings"] = uniform
            training, prefixes = {}, {}
            problem_edges = sorted(zip(*np.nonzero(ins["quadratic"])))
            for method, graph, layers, target_state, maxiter in [
                ("routed_graph_aqc", problem_edges, config["prefix_ansatz_layers"], prefix_state, config["prefix_maxiter"]),
                ("native_prefix_aqc", edges, config["prefix_ansatz_layers"], prefix_state, config["prefix_maxiter"]),
                ("native_full_aqc", edges, config["full_ansatz_layers"], full_state, config["full_maxiter"]),
            ]:
                params, state, record = native_aqc.fit(n, graph, layers, target_state,
                    config["initialization_seeds"], maxiter, config["fidelity_threshold"])
                record["parameters"] = params.tolist()
                training[method] = record
                prep = native_aqc.circuit(n, graph, layers, params)
                assert np.isclose(abs(np.vdot(Statevector.from_instruction(prep).data, state))**2, 1., atol=1e-8)
                prefixes[method] = prep
                circuits[method] = prep if method == "native_full_aqc" else prep.compose(tail)
                print(method, "F", record["fidelity"], "seconds", record["training_sec"], flush=True)
            choices, qstart = [], time.perf_counter()
            for initial_seed in config["qaoa_initialization_seeds"]:
                x0 = np.random.default_rng(initial_seed).uniform(-np.pi, np.pi, 2*config["qaoa_depth"])
                def objective(params):
                    return float(np.dot(abs(qaoa_state(ins, params))**2, ins["energy"]))
                choices.append(minimize(objective, x0, method="COBYLA", options={
                    "maxiter": config["qaoa_maxfev_per_start"], "rhobeg": .5, "tol": 1e-5}))
            best = min(choices, key=lambda r: r.fun)
            training["standard_qaoa_p2"] = dict(parameters=best.x.tolist(), training_sec=time.perf_counter()-qstart,
                objective="exact_mean_penalized_energy", energy=float(best.fun),
                objective_evaluations=sum(int(r.nfev) for r in choices))
            circuits["standard_qaoa_p2"] = make_qaoa_circuit(ins, best.x).decompose(reps=2)
            rows = []
            for method in config["hardware_methods"]:
                qc = circuits[method]
                qc.name = f"{case_id}_{method}"
                qc.metadata = dict(study=config["schema_version"], instance_id=case_id, method=method,
                                   n_model_qubits=n, physical_qubits=physical)
                state = Statevector.from_instruction(qc).data
                if method == "refined_target":
                    assert np.isclose(abs(np.vdot(state, full_state))**2, 1, atol=1e-8)
                isa, resources = transpile_and_check(qc, target, physical, edges,
                    config["transpiler_seed"], native=method in ["native_full_aqc", "uniform_bitstrings"])
                qpy_write(out/"circuits"/f"{case_id}_{method}_logical.qpy", qc)
                isa_path = out/"circuits"/f"{case_id}_{method}_isa.qpy"
                qpy_write(isa_path, isa)
                row = dict(instance_id=case_id, method=method, n_qubits=n,
                    full_ideal_fidelity_to_refined=float(abs(np.vdot(full_state, state))**2),
                    **metrics(ins, abs(state)**2), **resources,
                    isa_qpy=str(isa_path.relative_to(ROOT)), isa_sha256=file_hash(isa_path))
                if method in prefixes:
                    _, prefix_resources = transpile_and_check(prefixes[method], target, physical, edges,
                        config["transpiler_seed"], native=method.startswith("native"))
                    row["prefix_resources"] = prefix_resources
                    row["compilation_fidelity"] = training[method]["fidelity"]
                rows.append(row)
            result = dict(instance_id=case_id, customers=ins["customers"], facilities=ins["facilities"], seed=seed,
                n_qubits=n, service_costs=ins["service"].tolist(), opening_costs=ins["opening"].tolist(),
                penalty=ins["penalty"], scale=ins["scale"], optimum=ins["optimum"],
                physical_qubits=physical, local_edges=edges, patch_selection=selection,
                target_build_sec=target_time, training=training, rows=rows,
                classical_baselines=classical_baselines(ins),
                preparation_wall_sec=time.perf_counter()-start, prepared_utc=now())
            write_json(destination, result)
            print("Completed", case_id, "seconds", result["preparation_wall_sec"], flush=True)
    aggregate(out, config)


def aggregate(out, config):
    cases = [json.loads(p.read_text()) for p in sorted((out/"cases").glob("*.json"))]
    rows = [row for c in cases for row in c["rows"]]
    write_csv(out/"ideal_and_resources.csv", [{k:v for k,v in row.items() if not isinstance(v, (dict,list))} for row in rows])
    write_json(out/"circuit_manifest.json", dict(config_sha256=file_hash(out/"source_snapshot/nighthawk_validation.json"),
        complete=len(cases)==sum(len(g["seeds"]) for g in config["cases"]), rows=rows,
        prepared_instances=len(cases), generated_utc=now()))


def connect(config):
    logging.getLogger("qiskit_ibm_runtime").setLevel(logging.ERROR)
    from qiskit_ibm_runtime import QiskitRuntimeService
    account = os.environ.get("IBM_QUANTUM_ACCOUNT") or config.get("account")
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    instance = os.environ.get("IBM_QUANTUM_INSTANCE") or config.get("instance")
    instance_name = config.get("instance_name")
    if token:
        options = {"channel": "ibm_quantum_platform", "token": token}
        if instance:
            options["instance"] = instance
        service = QiskitRuntimeService(**options)
    else:
        service = QiskitRuntimeService(**({"name": account} if account else {}))
    if instance_name and not instance:
        matches = [i for i in service.instances() if i.get("name") == instance_name]
        if len(matches) != 1:
            raise ValueError("Set IBM_QUANTUM_INSTANCE to your accessible instance CRN")
        instance = matches[0]["crn"]
    backend_name = os.environ.get("IBM_QUANTUM_BACKEND") or config["backend"]
    backend = service.backend(backend_name, **({"instance": instance} if instance else {}))
    processor = backend.configuration().processor_type
    if processor.get("family") != config["processor_family_required"]:
        raise ValueError(f"Wrong processor family: {processor}")
    return service, backend


def submit(config_path):
    from qiskit_ibm_runtime import SamplerV2
    config = json.loads(Path(config_path).read_text())
    out = ROOT/config["output_dir"]
    manifest = json.loads((out/"circuit_manifest.json").read_text())
    if not manifest["complete"]:
        raise ValueError("Prepare every planned instance before submission")
    if manifest["config_sha256"] != file_hash(config_path):
        raise ValueError("Configuration changed after preparation")
    service, backend = connect(config)
    live_edges = {tuple(sorted(e)) for e in backend.target["cz"]}
    for path in (out/"cases").glob("*.json"):
        case = json.loads(path.read_text())
        if any(tuple(sorted(e)) not in live_edges for e in case["patch_selection"]["selected_physical_edges"]):
            raise ValueError("Backend connectivity changed; preparation needs review")
    jobs_path = out/"jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = config["shots_per_circuit_per_block"]
    sampler.options.max_execution_time = config["max_execution_seconds_per_job"]
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    for block in range(config["execution_blocks"]):
        if any(job["block"]==block for job in jobs):
            print("Already submitted block", block, flush=True)
            continue
        rng = np.random.default_rng(config["randomization_seed"]+block)
        order = rng.permutation(len(manifest["rows"])).tolist()
        ordered = [manifest["rows"][i] for i in order]
        circuits = []
        for row in ordered:
            path = ROOT/row["isa_qpy"]
            if file_hash(path) != row["isa_sha256"]:
                raise ValueError("Circuit changed after manifest creation")
            circuits.append(qpy_read(path))
        calibration = backend.properties(refresh=True)
        # Calibration files may include datetimes, but contain no credentials.
        (out/f"calibration_block_{block}.json").write_text(json.dumps(calibration.to_dict(), indent=2, default=str)+"\n")
        sampler.options.environment.job_tags = [config["schema_version"], f"block-{block}"]
        job = sampler.run(circuits, shots=config["shots_per_circuit_per_block"])
        record = dict(block=block, job_id=job.job_id(), submitted_utc=now(),
            backend=backend.name, backend_version=backend.backend_version,
            processor_type=backend.configuration().processor_type,
            calibration_timestamp=str(calibration.last_update_date),
            shots_per_pub=config["shots_per_circuit_per_block"],
            pubs=[dict(instance_id=row["instance_id"], method=row["method"], isa_sha256=row["isa_sha256"]) for row in ordered])
        jobs.append(record)
        write_json(jobs_path, jobs)
        print("Submitted", job.job_id(), "block", block, "circuits", len(circuits), flush=True)


def collect(config_path):
    config = json.loads(Path(config_path).read_text())
    out = ROOT/config["output_dir"]
    (out/"raw_counts").mkdir(exist_ok=True)
    service, backend = connect(config)
    jobs = json.loads((out/"jobs.json").read_text())
    for record in jobs:
        path = out/"raw_counts"/f"block_{record['block']}.json"
        if path.exists():
            print("Already collected", record["job_id"], flush=True)
            continue
        job = service.job(record["job_id"])
        status = str(job.status())
        print(record["job_id"], status, flush=True)
        if status not in ("DONE", "JobStatus.DONE"):
            continue
        result = job.result()
        rows = []
        for metadata, pub in zip(record["pubs"], result):
            counts = pub.data.meas.get_counts()
            if sum(counts.values()) != record["shots_per_pub"]:
                raise ValueError("Unexpected shot total")
            rows.append(dict(metadata, counts=counts, metadata=pub.metadata))
        if len(rows) != len(record["pubs"]):
            raise ValueError("PUB count mismatch")
        payload = dict(block=record["block"], job_id=record["job_id"], collected_utc=now(),
            backend=record["backend"], processor_type=record["processor_type"], rows=rows,
            job_metrics=job.metrics(), result_metadata=result.metadata)
        path.write_text(json.dumps(payload, indent=2, default=str)+"\n")
        print("Collected", len(rows), "PUBs", flush=True)


def scaling(config_path, case_index):
    config = json.loads(Path(config_path).read_text())
    out = ROOT/config["output_dir"]
    (out/"scaling").mkdir(exist_ok=True)
    c, f, seed = config["scaling_cases"][case_index]
    destination = out/"scaling"/f"case_{case_index}.json"
    if destination.exists():
        print("Already measured", destination, flush=True)
        return
    start = time.perf_counter()
    ins = instance(c, f, seed, config)
    model_sec = time.perf_counter()-start
    with (out/"target_snapshot.pkl").open("rb") as handle:
        target = pickle.load(handle)
    physical, edges, selection = native_aqc.select_patch(target, ins["n_qubits"])
    start_target = time.perf_counter()
    state = anneal(ins, config["annealing_time"], config["target_prefix_steps"], end_fraction=config["prefix_fraction"])
    target_sec = time.perf_counter()-start_target
    params, compiled, record = native_aqc.fit(ins["n_qubits"], edges, config["scaling_layers"], state,
        [config["scaling_initialization_seed"]], config["scaling_maxiter"], config["fidelity_threshold"])
    qc = native_aqc.circuit(ins["n_qubits"], edges, config["scaling_layers"], params)
    _, resources = transpile_and_check(qc, target, physical, edges, config["transpiler_seed"], native=True)
    reference = evolution(ins, config["annealing_time"], config["target_prefix_steps"], end=config["prefix_fraction"])
    _, target_resources = transpile_and_check(reference, target, physical, edges, config["transpiler_seed"])
    full_native = anneal(ins, config["annealing_time"], config["tail_steps"], state=compiled, start_fraction=config["prefix_fraction"])
    full_reference = anneal(ins, config["annealing_time"], config["tail_steps"], state=state, start_fraction=config["prefix_fraction"])
    result = dict(case_index=case_index, customers=c, facilities=f, seed=seed, n_qubits=ins["n_qubits"],
        model_enumeration_sec=model_sec, target_build_sec=target_sec, training=record,
        native_resources=resources, refined_prefix_resources=target_resources,
        compiled_final_metrics=metrics(ins, abs(full_native)**2),
        reference_final_metrics=metrics(ins, abs(full_reference)**2),
        classical_baselines=classical_baselines(ins),
        peak_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        memory_measurement="Linux ru_maxrss, fresh process for this one case, includes imports and exact model enumeration",
        physical_qubits=physical, completed_utc=now(),
        exact_statevector_bytes=16*2**ins["n_qubits"], hardware_executed=False)
    write_json(destination, result)
    print(json.dumps(dict(n=ins["n_qubits"], fidelity=record["fidelity"], training_sec=record["training_sec"],
                          peak_rss_mib=result["peak_process_rss_mib"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "submit", "collect", "scaling"])
    parser.add_argument("--config", default=str(ROOT/"configs/nighthawk_validation.json"))
    parser.add_argument("--only")
    parser.add_argument("--case-index", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.config, args.only)
    elif args.action == "scaling":
        scaling(args.config, args.case_index)
    else:
        globals()[args.action](args.config)


if __name__ == "__main__":
    main()
