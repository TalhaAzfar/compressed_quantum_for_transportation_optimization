"""Reproducible, noiseless FLP pilot with corrected initial-state semantics.

This module never connects to quantum hardware. Numerical probabilities are
exact statevector probabilities, not shot estimates. Existing submitted results
are inputs to a separate retrospective analysis and are never overwritten here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_instance(customers, seed, config):
    """Build FLP coefficients in little-endian model-variable order.

    Q = objective + A sum_i(1-sum_j x_ij)^2 + A sum_ij x_ij(1-y_j).
    The constant A*customers is dropped; it only changes global phase.
    """
    facilities = int(config["facility_count"])
    rng = np.random.default_rng(seed)
    slo, shi = config["service_cost_range_inclusive"]
    olo, ohi = config["opening_cost_range_inclusive"]
    service = rng.integers(slo, shi + 1, size=(customers, facilities)).astype(float)
    opening = rng.integers(olo, ohi + 1, size=facilities).astype(float)
    n = facilities * (customers + 1)
    penalty = 2.0 * (service.sum() + opening.sum())
    linear = np.zeros(n)
    quadratic = np.zeros((n, n))
    linear[:facilities] = opening
    for i in range(customers):
        indices = [facilities + i * facilities + j for j in range(facilities)]
        for j, index in enumerate(indices):
            # -A*x from one-hot plus +A*x from open-if-assigned cancel.
            linear[index] += service[i, j]
            quadratic[j, index] -= penalty
        for a, u in enumerate(indices):
            for v in indices[a + 1:]:
                quadratic[u, v] += 2 * penalty
    scale = max(float(np.abs(linear).max()), float(np.abs(quadratic).max()), 1.0)
    basis = np.arange(2**n, dtype=np.uint64)
    bits = ((basis[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(float)
    assignment = bits[:, facilities:].reshape(-1, customers, facilities)
    feasible = (assignment.sum(axis=2) == 1).all(axis=1)
    feasible &= (assignment <= bits[:, None, :facilities]).all(axis=(1, 2))
    costs = bits[:, :facilities] @ opening + (assignment * service).sum(axis=(1, 2))
    energy = (bits @ linear + np.einsum("bi,ij,bj->b", bits, quadratic, bits)) / scale
    optimum = float(costs[feasible].min())
    good = feasible & (costs <= optimum * (1 + config["primary_gap_tolerance"]) + 1e-10)
    optimal = feasible & np.isclose(costs, optimum, atol=1e-10, rtol=0)
    ground = np.isclose(energy, energy.min(), atol=1e-10, rtol=0)
    if not np.all(optimal[ground]):
        raise AssertionError("Penalized Hamiltonian ground states must be optimal feasible states")
    return dict(
        instance_id=f"flp_c{customers}_f{facilities}_seed{seed}",
        customers=customers, facilities=facilities, seed=seed, n_qubits=n,
        service=service, opening=opening, penalty=penalty, scale=scale,
        linear=linear / scale, quadratic=quadratic / scale, energy=energy,
        qubo_constant_dropped=float(penalty * customers / scale),
        ising_identity_dropped=float((linear.sum() / 2 + quadratic.sum() / 4) / scale),
        feasible=feasible, good=good, optimal=optimal, costs=costs, optimum=optimum,
        gap_tolerance=float(config["primary_gap_tolerance"]),
    )


def apply_driver(state, angle, n_qubits):
    """Apply exp(-i angle sum X) in Qiskit's little-endian order."""
    state = np.asarray(state, dtype=complex).copy()
    c, s = np.cos(angle), -1j * np.sin(angle)
    for q in range(n_qubits):
        view = state.reshape(-1, 2, 2**q)
        lo, hi = view[:, 0, :].copy(), view[:, 1, :].copy()
        view[:, 0, :], view[:, 1, :] = c * lo + s * hi, s * lo + c * hi
    return state


def anneal(instance, duration, num_steps, driver=-1.0, *, state=None,
           start_fraction=0.0, end_fraction=1.0):
    """Midpoint first-order driver-then-cost digitization over an interval."""
    n = instance["n_qubits"]
    if state is None:
        state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    else:
        state = np.asarray(state, dtype=complex).copy()
    dt = duration * (end_fraction - start_fraction) / num_steps
    for k in range(num_steps):
        s = start_fraction + (end_fraction - start_fraction) * (k + .5) / num_steps
        state = apply_driver(state, dt * (1 - s) * driver, n)
        state *= np.exp(-1j * dt * s * instance["energy"])
    return state


def metrics(instance, probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
        raise AssertionError("Distribution is not normalized")
    feasible = instance["feasible"]
    p = float(probabilities[feasible].sum())
    return dict(
        p_feasible=p,
        p_good=float(probabilities[instance["good"]].sum()),
        p_optimal=float(probabilities[instance["optimal"]].sum()),
        mean_feasible_gap=(float(np.dot(probabilities[feasible],
            (instance["costs"][feasible] - instance["optimum"]) / instance["optimum"])) / p
            if p > 1e-15 else None),
    )


def qaoa_state(instance, parameters):
    n = instance["n_qubits"]
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    for gamma, beta in np.asarray(parameters).reshape(-1, 2):
        state *= np.exp(-1j * gamma * instance["energy"])
        state = apply_driver(state, beta, n)
    return state


def ising_operator(instance):
    from qiskit.quantum_info import SparsePauliOp
    n = instance["n_qubits"]
    linear, quadratic = instance["linear"], instance["quadratic"]
    zcoeff = -linear / 2 - (quadratic.sum(axis=0) + quadratic.sum(axis=1)) / 4
    terms = [("Z", [q], float(c)) for q, c in enumerate(zcoeff) if c]
    for i, j in zip(*np.nonzero(quadratic)):
        terms.append(("ZZ", [int(i), int(j)], float(quadratic[i, j] / 4)))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n)


def make_qaoa_circuit(instance, parameters):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    qc = QuantumCircuit(instance["n_qubits"])
    qc.h(range(qc.num_qubits))
    hc = ising_operator(instance)
    for gamma, beta in np.asarray(parameters).reshape(-1, 2):
        qc.append(PauliEvolutionGate(hc, time=float(gamma)), range(qc.num_qubits))
        qc.rx(2 * float(beta), range(qc.num_qubits))
    return qc


def circuit_resources(circuit, config):
    from qiskit import transpile
    start = time.perf_counter()
    transpiled = transpile(circuit, basis_gates=config["circuit_resource_basis"],
                           optimization_level=3, seed_transpiler=config["transpiler_seed"])
    return dict(logical_cx_depth=transpiled.depth(lambda inst: inst.operation.num_qubits == 2),
                logical_cx_count=int(transpiled.count_ops().get("cx", 0)),
                resource_compilation_sec=time.perf_counter() - start)


def identity_row(instance):
    return {k: instance[k] for k in ["instance_id", "customers", "seed", "n_qubits"]}


def classical_baselines(instance):
    start = time.perf_counter()
    values = []
    for mask in range(1, 2**instance["facilities"]):
        opened = [j for j in range(instance["facilities"]) if mask & (1 << j)]
        value = float(instance["opening"][opened].sum() +
                      instance["service"][:, opened].min(axis=1).sum())
        values.append(value)
    solve_sec = time.perf_counter() - start
    if not np.isclose(min(values), instance["optimum"]):
        raise AssertionError("Constructive FLP reference and full enumeration disagree")
    uniform = np.full(len(instance["energy"]), 1 / len(instance["energy"]))
    feasible_uniform = instance["feasible"] / instance["feasible"].sum()
    threshold = instance["optimum"] * (1 + instance["gap_tolerance"]) + 1e-10
    rows = [dict(identity_row(instance), method="uniform_bitstrings", **metrics(instance, uniform)),
            dict(identity_row(instance), method="uniform_feasible", **metrics(instance, feasible_uniform)),
            dict(identity_row(instance), method="random_open_subset_greedy_assignment",
                 p_feasible=1.0, p_good=float(np.mean(np.asarray(values) <= threshold)),
                 p_optimal=float(np.mean(np.isclose(values, instance["optimum"]))),
                 mean_feasible_gap=float(np.mean(values) / instance["optimum"] - 1)),
            dict(identity_row(instance), method="enumerate_open_subsets_greedy_assignment",
                 p_feasible=1.0, p_good=1.0, p_optimal=1.0, mean_feasible_gap=0.0,
                 solve_sec=solve_sec, evaluated_open_subsets=len(values))]
    return rows


def compression_controls(instance, config, out):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_addon_aqc_tensor.ansatz_generation import generate_ansatz_from_circuit
    from .aqc_workflow import AQCSettings, build_annealing_circuit_with_aqc_prefix, save_qpy

    c = config["compression"]
    settings = AQCSettings(maxiter=c["maxiter"], stopping_fidelity=c["stopping_fidelity"],
                           mps_max_bond_dimension=c["mps_max_bond_dimension"],
                           aqc_target_refinement=c["target_refinement"],
                           aqc_good_steps=c["shallow_prefix_steps"])
    start = time.perf_counter()
    full, prefixes, remaining, record = build_annealing_circuit_with_aqc_prefix(
        instance["n_qubits"], ising_operator(instance), num_steps=c["num_steps"],
        alpha=c["prefix_steps"], delta_t=c["annealing_time"] / c["num_steps"],
        aqc_target_refinement=c["target_refinement"], aqc_good_steps=c["shallow_prefix_steps"],
        aqc_settings=settings, measure=False)
    wall = time.perf_counter() - start
    prep = QuantumCircuit(instance["n_qubits"])
    prep.h(range(prep.num_qubits))
    target_state = Statevector.from_instruction(prep.compose(prefixes["target_prefix"]).decompose(reps=4))
    rows, distributions = [], {}
    variants = {"refined_target": prefixes["target_prefix"],
                "shallow_unoptimized": prefixes["good_prefix"],
                "aqc_compressed": prefixes["compressed_prefix"]}
    initial_ansatz, initial_parameters = generate_ansatz_from_circuit(
        prefixes["good_prefix"].decompose(reps=4))
    variants["initial_ansatz"] = initial_ansatz.assign_parameters(initial_parameters)
    for name, prefix in variants.items():
        prefix_preparation = prep.compose(prefix)
        sv = Statevector.from_instruction(prefix_preparation.decompose(reps=4))
        fidelity = float(abs(np.vdot(target_state.data, sv.data))**2)
        circuit = prefix_preparation
        if remaining["tail"] is not None:
            circuit = circuit.compose(remaining["tail"])
        final = Statevector.from_instruction(circuit.decompose(reps=4))
        probs = np.abs(final.data)**2
        distributions[name] = probs
        resources = circuit_resources(circuit, config)
        row = dict(identity_row(instance), method=name, annealing_time=c["annealing_time"],
                   num_steps=c["num_steps"], prefix_steps=c["prefix_steps"],
                   actual_input_prefix_fidelity=fidelity, **resources, **metrics(instance, probs))
        if name == "aqc_compressed":
            if abs(fidelity - record.final_fidelity) > 5e-6:
                raise AssertionError("Reported AQC fidelity disagrees with actual input state")
            direct_full = Statevector.from_instruction(full.decompose(reps=4))
            if not np.isclose(abs(np.vdot(direct_full.data, final.data))**2, 1, atol=1e-9):
                raise AssertionError("Builder and assembled corrected state differ")
            row.update(compression_wall_sec=wall, target_build_sec=record.target_build_sec,
                       compression_optimizer_sec=record.compression_sec,
                       optimizer_iterations=record.optimizer_iterations,
                       optimizer_evaluations=record.objective_evaluations,
                       optimizer_status=record.optimizer_status_code,
                       reported_fidelity=record.final_fidelity,
                       fidelity_threshold_met=bool(fidelity >= c["stopping_fidelity"]))
        circuit.metadata = {**(circuit.metadata or {}), "study": config["schema_version"],
                            "instance_id": instance["instance_id"], "method": name,
                            "hardware_executed": False}
        save_qpy(out / "circuits" / f"{instance['instance_id']}_{name}.qpy", circuit)
        rows.append(row)
    if not np.allclose(distributions["initial_ansatz"], distributions["shallow_unoptimized"],
                       atol=1e-10, rtol=1e-8):
        raise AssertionError("Initial ansatz must reproduce the shallow circuit distribution")
    coarse, _, _, _ = build_annealing_circuit_with_aqc_prefix(
        instance["n_qubits"], ising_operator(instance), num_steps=c["num_steps"],
        alpha=0, delta_t=c["annealing_time"] / c["num_steps"], measure=False)
    coarse_sv = Statevector.from_instruction(coarse.decompose(reps=4))
    expected = anneal(instance, c["annealing_time"], c["num_steps"])
    if not np.isclose(abs(np.vdot(coarse_sv.data, expected))**2, 1, atol=1e-8):
        raise AssertionError("NumPy simulator and Qiskit evolution disagree")
    distributions["coarse_uncompressed"] = np.abs(coarse_sv.data)**2
    rows.append(dict(identity_row(instance), method="coarse_uncompressed",
                     annealing_time=c["annealing_time"], num_steps=c["num_steps"],
                     prefix_steps=0, **circuit_resources(coarse, config),
                     **metrics(instance, distributions["coarse_uncompressed"])))
    save_qpy(out / "circuits" / f"{instance['instance_id']}_coarse_uncompressed.qpy", coarse)
    np.savez_compressed(out / "distributions" / f"{instance['instance_id']}.npz", **distributions)
    return rows


def run(config_path, *, skip_compression=False, instance_limit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    out = ROOT / config["output_dir"]
    for directory in [out, out / "circuits", out / "distributions"]:
        directory.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ["numpy", "scipy", "qiskit", "qiskit-addon-aqc-tensor", "quimb", "jax"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not_installed"
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      config=config, config_sha256=file_hash(config_path),
                      source_sha256=file_hash(__file__), python=platform.python_version(),
                      packages=packages, exact_probabilities=True, hardware_executed=False,
                      skip_compression=skip_compression, instance_limit=instance_limit,
                      scope="exploratory synthetic-instance noiseless pilot, not device validation")
    dependencies = [Path(__file__), ROOT / "revision/aqc_workflow.py", config_path]
    snapshots = out / "source_snapshot"
    snapshots.mkdir(exist_ok=True)
    provenance["source_files"] = {}
    for source in dependencies:
        target = snapshots / source.name
        shutil.copy2(source, target)
        provenance["source_files"][str(source.relative_to(ROOT))] = file_hash(source)
    provenance["instance_dependence"] = (
        "Five seed families reused across three sizes; random streams share prefixes. "
        "Do not treat all 15 cases as independent population replicates.")
    # Record the full protocol before any experiment; no outcome-dependent settings.
    write_json(out / "protocol_and_provenance.json", provenance)
    instances = [generate_instance(c, s, config) for c in config["customer_counts"]
                 for s in config["instance_seeds"]]
    if instance_limit is not None:
        instances = instances[:instance_limit]
    write_json(out / "instances.json", [dict(identity_row(i), service_costs=i["service"].tolist(),
        opening_costs=i["opening"].tolist(), penalty=i["penalty"], normalization_scale=i["scale"],
        normalized_qubo_constant_dropped=i["qubo_constant_dropped"],
        normalized_ising_identity_dropped=i["ising_identity_dropped"],
        optimum=i["optimum"], feasible_states=int(i["feasible"].sum()),
        good_states=int(i["good"].sum())) for i in instances])
    baseline_rows, duration_rows, qaoa_rows, compression_rows, qaoa_parameters = [], [], [], [], []
    for instance in instances:
        print("Instance", instance["instance_id"], flush=True)
        baseline_rows.extend(classical_baselines(instance))
        for duration in config["annealing_times"]:
            for steps in config["trotter_steps"]:
                for driver in config["driver_coefficients"]:
                    start = time.perf_counter()
                    state = anneal(instance, duration, steps, driver)
                    duration_rows.append(dict(identity_row(instance), annealing_time=duration,
                        num_steps=steps, driver_coefficient=driver,
                        simulation_sec=time.perf_counter() - start,
                        **metrics(instance, abs(state)**2)))
        for depth in config["qaoa"]["depths"]:
            start = time.perf_counter()
            choices = []
            total_evaluations = 0
            for seed in config["qaoa"]["initialization_seeds"]:
                x0 = np.random.default_rng(seed).uniform(-np.pi, np.pi, 2 * depth)
                def objective(params):
                    probabilities = abs(qaoa_state(instance, params))**2
                    return float(np.dot(probabilities, instance["energy"]))
                result = minimize(objective, x0, method="COBYLA", options={
                    "maxiter": config["qaoa"]["max_function_evaluations_per_start"],
                    "rhobeg": 0.5, "tol": 1e-5})
                total_evaluations += int(result.nfev)
                choices.append(result)
            best = min(choices, key=lambda r: r.fun)
            training_sec = time.perf_counter() - start
            circuit = make_qaoa_circuit(instance, best.x)
            probs = abs(qaoa_state(instance, best.x))**2
            qaoa_rows.append(dict(identity_row(instance), method=f"standard_qaoa_p{depth}",
                qaoa_depth=depth, training_sec=training_sec, energy_evaluations=total_evaluations,
                training_energy=float(best.fun), **circuit_resources(circuit, config),
                **metrics(instance, probs)))
            qaoa_parameters.append(dict(instance_id=instance["instance_id"], depth=depth,
                parameters=best.x.tolist(), training_energy=float(best.fun),
                optimizer_success=bool(best.success), optimizer_message=str(best.message)))
        if not skip_compression:
            compression_rows.extend(compression_controls(instance, config, out))
        # Checkpoint only in the new study folder; an interrupted run is auditable.
        write_csv(out / "classical_baselines.csv", baseline_rows)
        write_csv(out / "duration_sweep.csv", duration_rows)
        write_csv(out / "qaoa_results.csv", qaoa_rows)
        write_csv(out / "compression_controls.csv", compression_rows)
        write_json(out / "qaoa_parameters.json", qaoa_parameters)
    summary = dict(instances=len(instances), duration_rows=len(duration_rows),
                   compression_rows=len(compression_rows), qaoa_rows=len(qaoa_rows),
                   complete=True, hardware_executed=False,
                   completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(out / "run_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/corrected_study.json"))
    parser.add_argument("--skip-compression", action="store_true")
    parser.add_argument("--instance-limit", type=int)
    args = parser.parse_args()
    run(args.config, skip_compression=args.skip_compression, instance_limit=args.instance_limit)


if __name__ == "__main__":
    main()
