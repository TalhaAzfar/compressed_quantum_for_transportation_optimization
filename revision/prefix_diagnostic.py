"""Post-acquisition FLP20 diagnostic; reads fixed circuits, never submits jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/prefix_diagnostic"
PARENT = ROOT / "outputs/aqc_steps_10_12_bounded"
EXTRA = ""


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def feasible_assignments(model):
    import itertools
    import numpy as np
    assert model["problem"] == "flp"
    c, f = model["customers"], model["facilities"]
    values, costs = [], []
    for mask in range(1, 1 << f):
        opened = [j for j in range(f) if mask >> j & 1]
        for assignment in itertools.product(opened, repeat=c):
            values.append(mask + sum(1 << (f+i*f+j) for i,j in enumerate(assignment)))
            costs.append(sum(model["opening"][j] for j in opened) + sum(model["service"][i][j] for i,j in enumerate(assignment)))
    return np.array(values, dtype=np.int64), np.array(costs, dtype=float)


def energy_diagonal(model):
    import numpy as np
    from .tensor_prefix_models import decode
    result = np.empty(1 << model["n_qubits"])
    for start in range(0, len(result), 65536):
        stop = min(start+65536, len(result))
        result[start:stop] = decode(model, np.arange(start, stop, dtype=np.uint64))["energy"]
    return result


def continue_state(state, diagonal, angles):
    """Full penalized energy changes only global phase relative to Ising H_C."""
    import numpy as np
    gamma, beta = angles
    result = state * np.exp(-1j*gamma*diagonal)
    n = (len(result)-1).bit_length()
    for q in range(n):
        block = result.reshape(-1, 2, 1 << q)
        zero, one = block[:,0,:].copy(), block[:,1,:].copy()
        block[:,0,:] = np.cos(beta)*zero - 1j*np.sin(beta)*one
        block[:,1,:] = np.cos(beta)*one - 1j*np.sin(beta)*zero
    return result


def probabilities(state, model, feasible, costs, diagonal):
    import numpy as np
    norm = float(np.vdot(state, state).real)
    assert abs(norm-1) < 1e-8, norm
    p = np.abs(state[feasible])**2 / norm
    good = costs <= 1.05*model["optimum"]+1e-9
    optimal = np.isclose(costs, model["optimum"], atol=1e-9, rtol=0)
    return dict(p_feasible=float(p.sum()), p_good=float(p[good].sum()), p_optimal=float(p[optimal].sum()),
                mean_feasible_cost=float(p@costs/p.sum()), mean_penalized_energy=float((np.abs(state)**2)@diagonal/norm), norm=norm)


def incumbent(model):
    import numpy as np
    from .endpoint_selection import greedy_incumbent
    opening, service = np.array(model["opening"]), np.array(model["service"])
    # The already published selector starts with the best total-cost singleton.
    cost, opened = greedy_incumbent(service, opening)
    assignment = [min(opened, key=lambda j:(service[i,j],j)) for i in range(len(service))]
    value = sum(1 << j for j in opened) + sum(1 << (len(opening)+i*len(opening)+j) for i,j in enumerate(assignment))
    return dict(opened=opened, assignment=assignment, value=value, cost=cost)


def worker():
    import csv
    import importlib.metadata
    import pickle
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from .nighthawk_study import now, qpy_read
    from .corrected_study import file_hash, write_csv
    from .tensor_prefix_models import count_metrics
    from .validate_tensor_prefix import independent_decode
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads((PARENT/"hardware_plan.json").read_text())
    case = json.loads((PARENT/"hardware/flp20.json").read_text())
    model = case["model"]; n = model["n_qubits"]
    rows = {r["method"]:r for r in plan["rows"]}; assert {r["case_id"] for r in rows.values()} == {"flp20"}
    names = ["revision/prefix_diagnostic.py", "tests/test_prefix_diagnostic.py", "revision/tensor_prefix_models.py", "revision/validate_tensor_prefix.py", "revision/endpoint_selection.py"]
    inputs = [PARENT/"hardware_plan.json", PARENT/"hardware/flp20.json", PARENT/"targets/flp20_s12_half.pkl", PARENT/"hardware_final.csv"]
    inputs += [ROOT/r["logical_qpy"] for r in rows.values()]
    protocol = dict(created_utc=now(), post_acquisition=True, case="flp20_s12_half", hardware_outcomes_already_known=True,
        scope="Explanatory single-case analysis; no training, target selection, new QPU calls, or confirmatory mechanism test",
        crossed_prefixes=["aqc","product","uncompressed","standard"], angle_sources=["aqc","product","uncompressed","standard"],
        mps_samples=4096, sample_seed=93001, feasible_sample_seed=93002,
        source_sha256={name:file_hash(ROOT/name) for name in names},
        input_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in inputs},
        packages={p:importlib.metadata.version(p) for p in ["numpy","qiskit","qiskit-aer","quimb"]})
    existing = OUT/"protocol.json"
    if existing.exists():
        old=json.loads(existing.read_text())
        assert old["source_sha256"] == protocol["source_sha256"] and old["input_sha256"] == protocol["input_sha256"]
    else:save(existing, protocol)
    if (OUT/"complete.json").exists(): print("Diagnostic already complete", flush=True); return
    feasible, costs = feasible_assignments(model)
    assert len(set(feasible)) == len(feasible) and abs(costs.min()-model["optimum"]) < 1e-9
    for value, cost in zip(feasible, costs):
        check=independent_decode(model, format(int(value), f"0{n}b"))
        assert check["feasible"] and abs(check["cost"]-cost) < 1e-9
    diagonal = energy_diagonal(model)
    np.savez_compressed(OUT/"feasible_space.npz", values=feasible, costs=costs)
    simulator = AerSimulator(method="statevector", max_parallel_threads=1, max_parallel_experiments=1, fusion_enable=True)
    def exact(qc):
        circuit=qc.copy(); circuit.save_statevector()
        result=simulator.run(circuit, shots=1).result(); assert result.success
        return np.asarray(result.get_statevector(), dtype=complex)
    states, standalones, timings = {}, [], {}
    for prefix in protocol["crossed_prefixes"]:
        start=time.perf_counter()
        if prefix == "standard":
            qc=QuantumCircuit(n); qc.h(range(n))
        else:qc=qpy_read(ROOT/rows[prefix]["logical_qpy"])
        states[prefix]=exact(qc)
        timings[prefix+"_statevector_seconds"]=time.perf_counter()-start
        standalones.append(dict(prefix=prefix, **probabilities(states[prefix],model,feasible,costs,diagonal)))
        np.save(OUT/(prefix+"_state.npy"),states[prefix])
        print(prefix, standalones[-1], flush=True)
    write_csv(OUT/"standalone_ideal.csv", standalones)
    angles = {p:rows[p+"_qaoa"]["training"]["parameters"] for p in protocol["angle_sources"]}
    save(OUT/"angles.json", angles)
    crossed=[]; checks=[]
    for prefix in protocol["crossed_prefixes"]:
        for source, theta in angles.items():
            state=continue_state(states[prefix], diagonal, theta)
            row=dict(prefix=prefix, angle_source=source, matched_training=prefix==source,
                     **probabilities(state,model,feasible,costs,diagonal))
            if prefix==source:
                independent=exact(qpy_read(ROOT/rows[prefix+"_qaoa"]["logical_qpy"]))
                overlap=float(abs(np.vdot(state,independent))**2)
                assert overlap > 1-1e-9, overlap
                checks.append(dict(prefix=prefix, independent_qpy_fidelity=overlap))
            crossed.append(row)
        print("Completed angle cross", prefix, flush=True)
    write_csv(OUT/"crossed_angles.csv",crossed)
    target = pickle.loads((PARENT/"targets/flp20_s12_half.pkl").read_bytes())
    start=time.perf_counter()
    # Dense contraction is an exact check of this saved MPS on this 20-qubit case.
    dense=np.asarray(target.psi.to_dense()).reshape([2]*n).transpose(list(reversed(range(n)))).reshape(-1)
    dense/=np.linalg.norm(dense)
    mps_metrics=probabilities(dense,model,feasible,costs,diagonal)
    exact_target_overlap=float(abs(np.vdot(states["uncompressed"],dense))**2)
    compiled_overlap=float(abs(np.vdot(states["aqc"],dense))**2)
    fit=json.loads((PARENT/"tasks/flp20_s12_half/l2_r0.json").read_text())
    assert abs(compiled_overlap-fit["fidelity"]) < 1e-8
    assert exact_target_overlap >= .999
    timings["mps_dense_check_seconds"]=time.perf_counter()-start
    # Direct conditional MPS sampling; this loop never materializes a probability table.
    start=time.perf_counter(); counts={}
    for k,(bits, probability) in enumerate(target.psi.sample(protocol["mps_samples"],seed=protocol["sample_seed"]),1):
        key="".join(str(int(b)) for b in reversed(bits)); counts[key]=counts.get(key,0)+1
        if k%512==0:print("Direct MPS samples", k, flush=True)
    timings["direct_mps_sampling_seconds"]=time.perf_counter()-start
    save(OUT/"mps_counts.json",counts)
    sample_metrics=count_metrics(model,counts)
    # Independent scalar decoding of every classical sample count.
    for event in ["feasible","good","optimal"]:
        expected=sum(count*int(independent_decode(model,bits)[event]) for bits,count in counts.items())
        assert expected==sample_metrics[event+"_count"]
    samples=np.random.default_rng(protocol["feasible_sample_seed"]).choice(feasible,protocol["mps_samples"])
    feasible_counts={format(int(v),f"0{n}b"):int(c) for v,c in zip(*np.unique(samples,return_counts=True))}
    save(OUT/"feasible_generator_counts.json",feasible_counts)
    greedy=incumbent(model);inc=independent_decode(model,format(greedy["value"],f"0{n}b"))
    assert inc["feasible"] and abs(inc["cost"]-greedy["cost"]) < 1e-9
    good=costs<=1.05*model["optimum"]+1e-9
    baselines=dict(optimum=model["optimum"], feasible_states=len(feasible),good_states=int(good.sum()),
        uniform_strings=dict(p_feasible=len(feasible)/2**n,p_good=float(good.sum()/2**n)),
        uniform_feasible=dict(p_feasible=1.,p_good=float(good.mean()),mean_cost=float(costs.mean()),
            sampled_metrics=count_metrics(model,feasible_counts),implementation="Enumerates 680 feasible assignments; a small-case reference, not a scalable uniform-feasible sampler"),
        heuristic={**greedy,"p_good":float(inc["good"]),"p_feasible":1.},
        saved_target_mps=dict(ideal_metrics=mps_metrics,sampled_metrics=sample_metrics,
            fidelity_to_exact_target=exact_target_overlap,aqc_fidelity_to_saved_target=compiled_overlap,
            sampling_includes_target_construction=False),timings=timings)
    save(OUT/"classical_baselines.json",baselines)
    save(OUT/"validation.json",dict(passed=True,independent_qpy_checks=checks,feasible_states_scalar_checked=len(feasible),
        mps_shots_scalar_checked=sum(counts.values()),mps_target_exact_overlap=exact_target_overlap,aqc_fidelity_to_saved_target=compiled_overlap))
    save(OUT/"complete.json",dict(completed_utc=now(),standalone_states=4,crossed_continuations=16,new_hardware_shots=0))


def run():
    from .memory_guard import guarded_run
    OUT.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"-m","revision.local_runner","--extra-site",EXTRA,"revision.prefix_diagnostic","worker"]
    result=guarded_run(command,OUT/"analysis.log",address_gib=4,rss_gib=3,reserve_gib=12,seconds=3600)
    save(OUT/"process.json",result)
    if result["status"]!="completed":raise RuntimeError(result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("action",choices=["run","worker"])
    {"run":run,"worker":worker}[parser.parse_args().action]()
