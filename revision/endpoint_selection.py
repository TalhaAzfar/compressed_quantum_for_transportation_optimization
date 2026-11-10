"""Offline, frozen split experiment for compression endpoint selection.

The proposal is a testable heuristic, not a new general compilation principle.
Gate-local stochastic noise is deliberately different from the scalar global
depolarizing surrogate used for selection. No IBM service is contacted.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import pickle
import shutil
import time
from pathlib import Path

import numpy as np

from . import native_aqc
from .corrected_study import ROOT, anneal, file_hash, metrics, write_csv, write_json
from .nighthawk_study import evolution, instance, now, qpy_read, qpy_write


def event_interval(p, fidelity):
    """Binary-measurement fidelity corollary; not a circuit-noise guarantee."""
    if not (0 <= p <= 1 and -1e-12 <= fidelity <= 1+1e-12):
        raise ValueError("Expected probabilities and squared fidelity")
    theta = np.arcsin(np.sqrt(p))
    delta = np.arccos(np.sqrt(np.clip(fidelity, 0., 1.)))
    return float(np.sin(max(0., theta-delta))**2), float(np.sin(min(np.pi/2, theta+delta))**2)


def greedy_incumbent(service, opening):
    """Best singleton, then strictly improving additions; never uses C*."""
    def cost(selected):
        return float(opening[list(selected)].sum()+service[:, list(selected)].min(axis=1).sum())
    selected = (min(range(len(opening)), key=lambda j: (cost((j,)), j)),)
    while len(selected) < len(opening):
        proposed = min((tuple(sorted((*selected, j))) for j in range(len(opening)) if j not in selected),
                       key=lambda subset: (cost(subset), subset))
        if cost(proposed) >= cost(selected)-1e-12:
            break
        selected = proposed
    return cost(selected), list(selected)


def readout_event_weights(mask, errors):
    """Pull a symmetric independent readout channel back onto an event mask."""
    values = np.asarray(mask, dtype=float).copy()
    for q, error in enumerate(errors):
        view = values.reshape(-1, 2, 2**q)
        low, high = view[:, 0, :].copy(), view[:, 1, :].copy()
        view[:, 0, :] = (1-error)*low + error*high
        view[:, 1, :] = error*low + (1-error)*high
    return values


def circuit_error(target, physical, name, local_qubits):
    qubits = tuple(physical[q] for q in local_qubits)
    prop = target[name].get(qubits)
    if prop is None and len(qubits) == 2:
        prop = target[name].get(tuple(reversed(qubits)))
    if prop is None or prop.error is None:
        raise ValueError(f"Missing calibration for {name} {qubits}")
    return float(prop.error)


def local_compile(qc, target, physical, edges, seed, native=False):
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    local = transpile(qc, basis_gates=["cz", "rz", "sx", "x"],
        coupling_map=[list(e) for u, v in edges for e in [(u, v), (v, u)]],
        initial_layout=list(range(qc.num_qubits)), optimization_level=3,
        routing_method="none" if native else "sabre", seed_transpiler=seed)
    mapping = local.layout.final_index_layout() if local.layout else list(range(qc.num_qubits))
    basis = np.arange(2**qc.num_qubits, dtype=np.uint32)
    permuted = np.zeros_like(basis)
    for q, wire in enumerate(mapping):
        permuted |= ((basis >> q) & 1) << wire
    expected = Statevector.from_instruction(qc).data
    actual = Statevector.from_instruction(local).data
    overlap = float(abs(np.vdot(expected, actual[permuted]))**2)
    assert abs(overlap-1) < 1e-8, overlap
    error_sum = 0.
    for item in local.data:
        wires = tuple(local.find_bit(q).index for q in item.qubits)
        if len(wires) == 2:
            assert tuple(sorted(wires)) in edges
        if item.operation.name in ["cz", "sx", "x"]:
            error_sum += circuit_error(target, physical, item.operation.name, wires)
    # RZ is an ideal virtual frame update in this deliberately limited model.
    readout_errors = [circuit_error(target, physical, "measure", (q,)) for q in range(qc.num_qubits)]
    return local, dict(final_layout=mapping, transpilation_fidelity=overlap,
        cz_count=int(local.count_ops().get("cz", 0)),
        cz_depth=int(local.depth(lambda item: item.operation.num_qubits == 2)),
        calibration_error_sum=error_sum+sum(readout_errors), readout_errors=readout_errors)


def settings(path):
    config = json.loads(Path(path).read_text())
    return config, ROOT/config["output_dir"]


def freeze(config_path):
    config, out = settings(config_path)
    out.mkdir(parents=True, exist_ok=True)
    for d in ["cases", "circuits", "noise", "source_snapshot"]:
        (out/d).mkdir(exist_ok=True)
    names = ["revision/endpoint_selection.py", "revision/native_aqc.py", "revision/nighthawk_study.py",
             "revision/corrected_study.py", "configs/endpoint_selection.json"]
    hashes = {name: file_hash(ROOT/name) for name in names}
    protocol = out/"protocol.json"
    if protocol.exists():
        old = json.loads(protocol.read_text())
        assert old["config"] == config, "Use a new output directory for changed settings"
        assert old["source_sha256"] == hashes, "Frozen study source changed"
    else:
        write_json(protocol, dict(frozen_utc=now(), config=config, source_sha256=hashes,
            archived_target_sha256=file_hash(ROOT/config["target_snapshot"]),
            packages={p: importlib.metadata.version(p) for p in ["numpy", "scipy", "qiskit", "qiskit-aer"]},
            test_outcomes_seen=False, preceding_hardware_outcomes_seen=True,
            registration="Locally timestamped before this experiment; not an external preregistration"))
        for name in names:
            shutil.copy2(ROOT/name, out/"source_snapshot"/Path(name).name)
    return config, out


def cases(config, split):
    for family in config["cases"]:
        for seed in family[split]:
            yield family["customers"], family["facilities"], seed


def case_id(c, f, seed):
    return f"flp_c{c}_f{f}_seed{seed}"


def prepare(config_path, split):
    config, out = freeze(config_path)
    if split == "test":
        assert (out/"selection_rule.json").exists(), "Freeze rule before generating test cases"
    with (ROOT/config["target_snapshot"]).open("rb") as stream:
        target = pickle.load(stream)
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    for c, f, seed in cases(config, split):
        cid = case_id(c, f, seed)
        dest = out/"cases"/(cid+".json")
        if dest.exists():
            continue
        print("Preparing", split, cid, flush=True)
        started = time.perf_counter()
        ins = instance(c, f, seed, config)
        n, duration, fraction = ins["n_qubits"], config["annealing_time"], config["prefix_fraction"]
        physical, edges, patch = native_aqc.select_patch(target, n)
        incumbent, opened = greedy_incumbent(ins["service"], ins["opening"])
        selection_mask = ins["feasible"] & (ins["costs"] <= (1+config["incumbent_tolerance"])*incumbent+1e-10)
        prefix = anneal(ins, duration, config["target_prefix_steps"], end_fraction=fraction)
        full = anneal(ins, duration, config["tail_steps"], state=prefix, start_fraction=fraction)
        tail = evolution(ins, duration, config["tail_steps"], fraction, 1., prepare=False)
        circuits = {"refined": evolution(ins, duration, config["target_prefix_steps"], end=fraction).compose(tail)}
        for steps in config["shallow_full_steps"]:
            circuits[f"shallow_s{steps}"] = evolution(ins, duration, steps)
        training = {}
        for endpoint in ["prefix", "full"]:
            for layers in config["native_layers"]:
                method = f"native_{endpoint}_l{layers}"
                params, state, record = native_aqc.fit(n, edges, layers, prefix if endpoint == "prefix" else full,
                    config["fit_seeds"], config["fit_maxiter"])
                prep = native_aqc.circuit(n, edges, layers, params)
                assert abs(abs(np.vdot(Statevector.from_instruction(prep).data, state))**2-1) < 1e-8
                circuits[method] = prep.compose(tail) if endpoint == "prefix" else prep
                training[method] = {**record, "parameters": params.tolist()}
        method = "routed_prefix_l3"
        graph = sorted(zip(*np.nonzero(ins["quadratic"])))
        params, state, record = native_aqc.fit(n, graph, config["routed_prefix_layers"], prefix,
            config["fit_seeds"], config["fit_maxiter"])
        circuits[method] = native_aqc.circuit(n, graph, config["routed_prefix_layers"], params).compose(tail)
        training[method] = {**record, "parameters": params.tolist()}
        uniform = QuantumCircuit(n)
        uniform.h(range(n))
        circuits["uniform"] = uniform
        rows = []
        for method, qc in circuits.items():
            ideal = Statevector.from_instruction(qc).data
            prob = abs(ideal)**2
            fidelity = float(np.clip(abs(np.vdot(full, ideal))**2, 0., 1.))
            p_reference = float(abs(full[selection_mask]) @ abs(full[selection_mask]))
            low, high = event_interval(p_reference, fidelity)
            p_event = float(prob[selection_mask].sum())
            assert low-1e-9 <= p_event <= high+1e-9
            local, resources = local_compile(qc, target, physical, edges, config["transpiler_seed"],
                native=method.startswith("native_full") or method == "uniform")
            path = out/"circuits"/f"{cid}_{method}.qpy"
            qpy_write(path, local)
            rows.append(dict(method=method, eligible=method != "uniform", fidelity=fidelity,
                ideal_selection=p_event, uniform_selection=float(selection_mask.mean()),
                selection_bound_low=low, selection_bound_high=high,
                **metrics(ins, prob), **resources, circuit=str(path.relative_to(ROOT)), circuit_sha256=file_hash(path)))
        write_json(dest, dict(instance_id=cid, split=split, customers=c, facilities=f, seed=seed, n_qubits=n,
            service=ins["service"].tolist(), opening=ins["opening"].tolist(), optimum=ins["optimum"],
            incumbent=incumbent, incumbent_opened=opened, incumbent_gap=incumbent/ins["optimum"]-1,
            physical=physical, edges=edges, patch=patch, training=training, candidates=rows,
            preparation_sec=time.perf_counter()-started, completed_utc=now()))
        print("Prepared", cid, "seconds", round(time.perf_counter()-started, 2), flush=True)


def simulate(config_path, split):
    config, out = freeze(config_path)
    if split == "test":
        assert (out/"test_selections.json").exists(), "Save all test predictions before noisy test evaluation"
    with (ROOT/config["target_snapshot"]).open("rb") as stream:
        target = pickle.load(stream)
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    for c, f, seed in cases(config, split):
        cid = case_id(c, f, seed)
        record = json.loads((out/"cases"/(cid+".json")).read_text())
        ins = instance(c, f, seed, config)
        selection_mask = ins["feasible"] & (ins["costs"] <= (1+config["incumbent_tolerance"])*record["incumbent"]+1e-10)
        for profile_index, (profile, multiplier) in enumerate(config["noise_profiles"].items()):
            for method_index, row in enumerate(record["candidates"]):
                destination = out/"noise"/f"{cid}_{row['method']}_{profile}.json"
                if destination.exists():
                    continue
                start = time.perf_counter()
                qc = qpy_read(ROOT/row["circuit"])
                assert file_hash(ROOT/row["circuit"]) == row["circuit_sha256"]
                noise = NoiseModel()
                seen = set()
                for item in qc.data:
                    name = item.operation.name
                    wires = tuple(qc.find_bit(q).index for q in item.qubits)
                    if name not in ["cz", "sx", "x"] or (name, wires) in seen:
                        continue
                    seen.add((name, wires))
                    error = circuit_error(target, record["physical"], name, wires)
                    d = 2**len(wires)
                    strength = multiplier*error*d/(d-1)
                    assert strength <= 1, "Noise profile outside intended convex mixture range"
                    noise.add_quantum_error(depolarizing_error(strength, len(wires)), name, list(wires))
                qc.save_probabilities(pershot=True)
                sim_seed = int(np.random.SeedSequence([config["noise_seed"], seed, profile_index, method_index]).generate_state(1)[0])
                simulator = AerSimulator(method="statevector", noise_model=noise,
                    max_parallel_threads=2, max_parallel_shots=2, max_parallel_experiments=1)
                result = simulator.run(qc, shots=config["noise_trajectories"], seed_simulator=sim_seed).result()
                assert result.success
                probabilities = np.asarray(result.data(0)["probabilities"])
                assert probabilities.shape == (config["noise_trajectories"], 2**ins["n_qubits"])
                assert np.max(abs(probabilities.sum(axis=1)-1)) < 1e-8
                basis = np.arange(2**ins["n_qubits"], dtype=np.uint32)
                mapped = np.zeros_like(basis)
                for q, wire in enumerate(row["final_layout"]):
                    mapped |= ((basis >> q) & 1) << wire
                events = {}
                for event, mask in [("selection", selection_mask), ("good", ins["good"]), ("feasible", ins["feasible"])]:
                    wire_mask = np.zeros(2**ins["n_qubits"], dtype=bool)
                    wire_mask[mapped] = mask
                    errors = [min(.5, multiplier*error) for error in row["readout_errors"]]
                    weights = readout_event_weights(wire_mask, errors)
                    values = probabilities @ weights
                    events[event] = dict(mean=float(values.mean()), mc_se=float(values.std(ddof=1)/np.sqrt(len(values))),
                                         trajectory_values=values.tolist())
                write_json(destination, dict(instance_id=cid, method=row["method"], profile=profile, split=split,
                    seed=sim_seed, trajectories=config["noise_trajectories"], events=events,
                    simulation_sec=time.perf_counter()-start, completed_utc=now()))
                print(split, cid, profile, row["method"], "seconds", round(time.perf_counter()-start, 2), flush=True)


def choose(rows, policy, multiplier=1., kappa=1.):
    rows = [r for r in rows if r["eligible"]]
    def score(row):
        visibility = math.exp(-kappa*multiplier*row["calibration_error_sum"])
        if policy == "event_calibrated":
            return row["uniform_selection"]+visibility*(row["ideal_selection"]-row["uniform_selection"])
        if policy == "fidelity_calibrated":
            return visibility*row["fidelity"]
        if policy == "ideal_event":
            return row["ideal_selection"]
        if policy == "ideal_fidelity":
            return row["fidelity"]
        if policy == "min_cz":
            return -row["cz_count"]
        raise ValueError(policy)
    return min(rows, key=lambda row: (-score(row), row["method"]))["method"]


def noise_record(out, cid, method, profile):
    return json.loads((out/"noise"/f"{cid}_{method}_{profile}.json").read_text())


def fit_rule(config_path):
    config, out = freeze(config_path)
    destination = out/"selection_rule.json"
    if destination.exists():
        return
    assert not any(r["split"] == "test" for p in (out/"noise").glob("*.json") for r in [json.loads(p.read_text())])
    records = [json.loads((out/"cases"/(case_id(*case)+".json")).read_text()) for case in cases(config, "development")]
    def quality(record, method, profile):
        value = noise_record(out, record["instance_id"], method, profile)["events"]["selection"]["mean"]
        return math.log(max(value, 1e-15)/record["candidates"][0]["uniform_selection"])
    fits, policies = {}, {}
    for policy in ["event_calibrated", "fidelity_calibrated"]:
        trials = []
        for kappa in config["kappa_grid"]:
            values = [quality(r, choose(r["candidates"], policy, multiplier, kappa), profile)
                for r in records for profile, multiplier in config["noise_profiles"].items()]
            trials.append(dict(kappa=kappa, mean_log_selection_lift=float(np.mean(values))))
        best = min(trials, key=lambda row: (-row["mean_log_selection_lift"], row["kappa"]))
        policies[policy] = best["kappa"]
        fits[policy] = trials
    fixed_trials = []
    for row in records[0]["candidates"]:
        if row["eligible"]:
            values = [quality(r, row["method"], profile) for r in records for profile in config["noise_profiles"]]
            fixed_trials.append(dict(method=row["method"], mean_log_selection_lift=float(np.mean(values))))
    fixed = min(fixed_trials, key=lambda row: (-row["mean_log_selection_lift"], row["method"]))["method"]
    write_json(destination, dict(frozen_utc=now(), protocol_sha256=file_hash(out/"protocol.json"),
        development_ids=[r["instance_id"] for r in records], policies=policies, fixed_method=fixed,
        development_tuning=fits, fixed_trials=fixed_trials, test_outcomes_seen=False,
        development_noise_sha256={p.name: file_hash(p) for p in sorted((out/"noise").glob("*.json"))}))


def predict(config_path):
    config, out = freeze(config_path)
    destination = out/"test_selections.json"
    if destination.exists():
        return
    rule = json.loads((out/"selection_rule.json").read_text())
    predictions = []
    for case in cases(config, "test"):
        cid = case_id(*case)
        record = json.loads((out/"cases"/(cid+".json")).read_text())
        for profile, multiplier in config["noise_profiles"].items():
            choices = {policy: choose(record["candidates"], policy, multiplier, kappa)
                       for policy, kappa in rule["policies"].items()}
            choices.update({p: choose(record["candidates"], p) for p in ["ideal_event", "ideal_fidelity", "min_cz"]})
            choices["development_fixed"] = rule["fixed_method"]
            predictions.append(dict(instance_id=cid, n_qubits=record["n_qubits"], profile=profile, choices=choices))
    assert all(not (out/"noise"/f"{r['instance_id']}_{m}_{r['profile']}.json").exists()
               for r in predictions for m in r["choices"].values())
    write_json(destination, dict(frozen_utc=now(), rule_sha256=file_hash(out/"selection_rule.json"),
        test_outcomes_seen=False, predictions=predictions,
        test_case_sha256={case_id(*case): file_hash(out/"cases"/(case_id(*case)+".json")) for case in cases(config, "test")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "simulate", "fit_rule", "predict"])
    parser.add_argument("--split", choices=["development", "test"], default="development")
    parser.add_argument("--config", default="configs/endpoint_selection.json")
    args = parser.parse_args()
    if args.action in ["prepare", "simulate"]:
        globals()[args.action](args.config, args.split)
    else:
        globals()[args.action](args.config)


if __name__ == "__main__":
    main()
