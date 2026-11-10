"""Recompile the pre-execution pilot with gatewise-verified SABRE, preserving v1."""
import argparse
import importlib.metadata
import json
import pickle
import shutil
import time
from pathlib import Path

from .corrected_study import ROOT,file_hash,write_json
from .nighthawk_study import now,qpy_read,qpy_write
from .tensor_prefix_sabre import compile_isa_sabre

CONFIG=ROOT/"configs/aqc_tensor_qaoa_v2.json"
PARENT=ROOT/"outputs/aqc_tensor_qaoa_v1"


def settings():
    config=json.loads(CONFIG.read_text())
    return config,ROOT/config["output_dir"]


def freeze():
    config,out=settings()
    original=json.loads((PARENT/"protocol.json").read_text())
    names=list(original["source_sha256"])+[
        str(CONFIG.relative_to(ROOT)),"revision/tensor_prefix_sabre.py",
        "revision/tensor_prefix_sabre_pilot.py","revision/tensor_prefix_hardware.py",
        "revision/run_tensor_prefix_pilot.py","revision/analyze_tensor_prefix.py",
        "revision/analyze_nighthawk.py","tests/test_tensor_prefix_sabre.py"]
    hashes={name:file_hash(ROOT/name) for name in names}
    path=out/"protocol.json"
    if path.exists():
        record=json.loads(path.read_text())
        assert record["source_sha256"]==hashes,"Frozen verified-SABRE pilot source changed"
        assert record["target_sha256"]==file_hash(out/"preflight/target_snapshot.pkl")
        assert record["parent_protocol_sha256"]==file_hash(PARENT/"protocol.json")
        assert record["cancellation_sha256"]==file_hash(PARENT/"preexecution_cancellation.json")
        return
    cancellation=json.loads((PARENT/"preexecution_cancellation.json").read_text())
    assert cancellation["status_after"] in ["CANCELLED","JobStatus.CANCELLED"]
    assert cancellation["metrics"]["timestamps"]["running"] is None
    assert cancellation["metrics"]["usage"]["quantum_seconds"]==0
    assert not list((PARENT/"raw_counts").glob("*.json"))
    (out/"source_snapshot").mkdir(exist_ok=True)
    for name in names:shutil.copy2(ROOT/name,out/"source_snapshot"/Path(name).name)
    write_json(path,dict(frozen_utc=now(),config=config,source_sha256=hashes,
        target_sha256=file_hash(out/"preflight/target_snapshot.pkl"),hardware_outcomes_seen=False,
        parent_protocol_sha256=file_hash(PARENT/"protocol.json"),
        cancellation_sha256=file_hash(PARENT/"preexecution_cancellation.json"),
        preparation_amendments=["preflight/ROUTING_AMENDMENT.md"],
        packages={p:importlib.metadata.version(p) for p in ["qiskit","qiskit-aer","qiskit-addon-aqc-tensor","quimb","jax","scipy"]}))


def prepare():
    config,out=settings()
    for folder in ["preflight","cases","circuits","targets"]:(out/folder).mkdir(parents=True,exist_ok=True)
    if not (out/"protocol.json").exists():
        for name in ["target_snapshot.pkl","backend_snapshot.json","calibration.json"]:
            shutil.copy2(PARENT/"preflight"/name,out/"preflight"/name)
        shutil.copy2(PARENT/"fallback_protocol.json",out/"fallback_protocol.json")
        for path in (PARENT/"targets").glob("*.pkl"):shutil.copy2(path,out/"targets"/path.name)
        (out/"preflight/ROUTING_AMENDMENT.md").write_text(
            "# Pre-execution routing amendment\n\n"
            "The v1 screen was cancelled while queued, with no execution start and zero reported QPU usage. "
            "Its complete records remain in `outputs/aqc_tensor_qaoa_v1/`. An offline compiler sensitivity "
            "check found avoidable routing overhead. This version uses seeded SABRE swaps while retaining "
            "unitary verification of each native gate decomposition and independent replay of every logical wire's gate order. "
            "Small circuits additionally receive exact-state checks. All original fit parameters, targets, QAOA initial angles, "
            "physical patches, selection rules, SPSA settings and shot budgets are reused unchanged. "
            "Every prepared circuit is recompiled before the new screen. No hardware outcomes were available at amendment. "
            "This is a protocol revision before acquisition, not an independent repeat of v1.\n")
    freeze()
    target=pickle.loads((out/"preflight/target_snapshot.pkl").read_bytes())
    for spec in config["cases"]:
        cid=spec["id"];path=out/"cases"/(cid+".json")
        if path.exists():continue
        original=PARENT/"cases"/(cid+".json");case=json.loads(original.read_text());start=time.perf_counter()
        for name,method in case["methods"].items():
            isa,resources=compile_isa_sabre(qpy_read(ROOT/method["logical_qpy"]),target,case["patch"]["physical"],case["patch"]["edges"])
            new_path=out/"circuits"/f"{cid}_{name}_isa.qpy";qpy_write(new_path,isa)
            method["previous_isa_sha256"]=method["isa_sha256"]
            method.update(isa_qpy=str(new_path.relative_to(ROOT)),isa_sha256=file_hash(new_path),resources=resources)
            print(cid,name,"CZ depth",resources["cz_depth"],flush=True)
        case["routing_amendment"]=dict(completed_utc=now(),seconds=time.perf_counter()-start,
            parent_case=str(original.relative_to(ROOT)),parent_case_sha256=file_hash(original),
            definition="Same logical circuits and patch, now with gatewise-verified SABRE routing. Original preparation timings exclude this recompilation.")
        write_json(path,case)
    print("All verified-SABRE cases prepared",flush=True)


def bind_modules():
    # Reuse the frozen experiment and analysis implementation without rewriting v1.
    # These explicit substitutions are part of the hashed v2 source protocol.
    from . import tensor_prefix_hardware as hardware
    from . import run_tensor_prefix_pilot as driver
    hardware.settings=settings;hardware.freeze=freeze;hardware.compile_isa=compile_isa_sabre
    driver.settings=settings
    return hardware,driver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","plan","run","analyze","validate","plot"])
    action=p.parse_args().action
    if action=="prepare":prepare();return
    freeze();hardware,driver=bind_modules()
    if action=="plan":hardware.screen_plan()
    elif action=="run":
        from . import analyze_tensor_prefix
        analyze_tensor_prefix.settings=settings
        driver.run()
    elif action=="analyze":
        from . import analyze_tensor_prefix
        analyze_tensor_prefix.settings=settings;analyze_tensor_prefix.analyze()
    elif action=="validate":
        from . import validate_tensor_prefix
        validate_tensor_prefix.settings=settings;validate_tensor_prefix.validate()
    elif action=="plot":
        from . import plot_tensor_prefix
        plot_tensor_prefix.settings=settings;plot_tensor_prefix.plot()


if __name__=="__main__":main()
