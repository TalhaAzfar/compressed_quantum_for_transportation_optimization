"""Frozen complete-circuit compression and noise-management hardware pilot."""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

from .full_circuit_compression import ROOT, EXTRA, settings, context, save


def select_fits(records,layers,sources):
    selected=[]
    for depth in layers:
        pair=[]
        for source in sources:
            candidates=[r for r in records if r.get("verified") and r["layers"]==depth and r["source"]==source]
            if not candidates: break
            pair.append(max(candidates,key=lambda r:(r["exact_fidelity"],-r["index"])))
        if len(pair)==len(sources):
            assert len({r["resources"]["cz_count"] for r in pair})==1
            assert len({r["resources"]["cz_depth"] for r in pair})==1
            selected.extend(pair)
    return selected


def prepare():
    import importlib.metadata
    import numpy as np
    from qiskit import QuantumCircuit, ClassicalRegister
    from .nighthawk_study import now,qpy_read,qpy_write
    from .corrected_study import file_hash
    from .readout_indicator_mitigation import bit_mapping
    c,out,parent,case,original=context()
    assert (out/"preparation_complete.json").exists()
    if (out/"deadline_recovery_protocol.json").exists():assert (out/"deadline_recovery_complete.json").exists()
    records=[json.loads(p.read_text()) for p in (out/"fits").glob("*/*.json")]
    if (out/"explicit_gradient_amendment.json").exists():
        assert (out/"explicit_complete.json").exists()
        records += [json.loads(p.read_text()) for p in (out/"explicit_fits").glob("*/*.json")]
    chosen=select_fits(records,c["hardware_layers"],c["targets"])
    assert chosen,"No verified complete-compression pair; do not submit unrelated hardware"
    rows=[]
    for r in chosen:
        rows.append(dict(method=f"full_{r['source']}_l{r['layers']}",kind="experiment",source=r["source"],layers=r["layers"],
            exact_fidelity=r["exact_fidelity"],ideal=r["ideal"],resources=r["resources"],
            isa_qpy=r["isa_qpy"],isa_sha256=r["isa_sha256"],logical_qpy=r["logical_qpy"],logical_sha256=r["logical_sha256"],
            shots=c["hardware_shots"]))
    for source in c["targets"]:
        r=original[source]
        rows.append(dict(method="original_"+source,kind="experiment",source=source,layers=None,
            exact_fidelity=1.,ideal=json.loads((out/"targets"/(source+".json")).read_text())["ideal"],
            resources=r["resources"],isa_qpy=r["isa_qpy"],isa_sha256=r["isa_sha256"],
            logical_qpy=r["logical_qpy"],logical_sha256=r["logical_sha256"],shots=c["hardware_shots"]))
    target=pickle.loads((parent/"preflight/target_snapshot.pkl").read_bytes())
    physical=case["patch"]["physical"]; n=len(physical)
    calibration=[]
    # All-zero and all-one fit the local model. Mixed patterns are held out.
    rng=np.random.default_rng(94004)
    patterns=[0,(1<<n)-1,sum((q%2)<<q for q in range(n)),sum(((q+1)%2)<<q for q in range(n))]
    patterns += [int(rng.integers(1,1<<n)) for _ in range(2)]
    for index,value in enumerate(patterns):
        qc=QuantumCircuit(target.num_qubits); qc.add_register(ClassicalRegister(n,"meas"))
        for q,phys in enumerate(physical):
            if value>>q&1: qc.x(phys)
        qc.barrier(*physical)
        for q,phys in enumerate(physical):qc.measure(phys,q)
        path=out/"circuits"/f"readout_{index}.qpy"; qpy_write(path,qc)
        calibration.append(dict(method=f"readout_{index}",kind="calibration" if index<2 else "calibration_check",
            prepared=value,isa_qpy=str(path.relative_to(ROOT)),isa_sha256=file_hash(path),shots=c["calibration_shots"]))
    for row in rows+calibration:
        assert file_hash(ROOT/row["isa_qpy"])==row["isa_sha256"]
        qc=qpy_read(ROOT/row["isa_qpy"])
        row["measurement_mapping"]=bit_mapping(qc,physical)
        for inst in qc.data:
            if inst.operation.name=="barrier":continue
            qargs=tuple(qc.find_bit(q).index for q in inst.qubits)
            assert target.instruction_supported(operation_name=inst.operation.name,qargs=qargs)
    expected=c["hardware_blocks"]*(2*sum(r["shots"] for r in rows)+sum(r["shots"] for r in calibration))
    assert expected<=c["hardware_max_shots"]
    plan=dict(created_utc=now(),rows=rows,calibration=calibration,physical=physical,model=case["model"],
        expected_shots=expected,hardware_blocks=c["hardware_blocks"],arms=["raw","suppressed"],
        preparation_protocol_sha256=file_hash(out/"protocol.json"))
    path=out/"hardware_plan.json"
    if path.exists():
        prior=json.loads(path.read_text()); assert all(prior[k]==plan[k] for k in plan if k!="created_utc")
    else:save(path,plan)
    sources=["revision/full_circuit_hardware.py","revision/readout_indicator_mitigation.py",
             "revision/analyze_full_circuit_mitigation.py",
             "revision/full_circuit_explicit.py",
             "revision/recover_full_circuit_deadline.py",
             "revision/audit_tensor_prefix_statistics.py",
             "tests/test_full_circuit_mitigation.py"]
    hashes={name:file_hash(ROOT/name) for name in sources}
    p=out/"hardware_protocol.json"
    if p.exists():assert json.loads(p.read_text())["source_sha256"]==hashes
    else:
        for name in sources:shutil.copy2(ROOT/name,out/"source_snapshot"/Path(name).name)
        save(p,dict(frozen_utc=now(),source_sha256=hashes,plan_sha256=file_hash(path),config=c,
                    package_versions={name:importlib.metadata.version(name) for name in
                        ["qiskit","qiskit-aer","qiskit-addon-aqc-tensor","qiskit-ibm-runtime","quimb","jax","numpy","scipy"]},
                    secondary_block_check="Direct conditional two-sided convolution stratified by acquisition block, Holm-adjusted over the same eight comparisons. Secondary robustness check; pooled Fisher remains primary.",
                    literature=["https://doi.org/10.1103/PhysRevA.103.042605",
                                "https://quantum.cloud.ibm.com/docs/en/guides/sampler-noise-management"]))
    testlog=(out/"tests.log").read_text()
    assert "passed" in testlog and "FAILED" not in testlog and "ERROR" not in testlog
    save(out/"hardware_validation.json",dict(passed=True,plan_sha256=file_hash(path),
        circuits_checked=len(rows)+len(calibration),matched_gate_budgets=True,
        tests_sha256=file_hash(out/"tests.log")))
    print("Prepared",len(rows),"methods;",expected,"total shots",flush=True)


def existing_submission(out,block,arm):
    path=out/"jobs.json"; jobs=json.loads(path.read_text()) if path.exists() else []
    if any(j["block"]==block and j["arm"]==arm for j in jobs):return jobs,True
    if (out/f"intent_{block}_{arm}.json").exists():
        raise RuntimeError("Ambiguous submission: reconcile receipt or remote tags before retrying")
    return jobs,False


def submit(block,arm):
    from .nighthawk_study import connect,now,qpy_read
    from .corrected_study import file_hash
    from qiskit_ibm_runtime import SamplerV2
    import numpy as np
    c,out=settings(); jobs,exists=existing_submission(out,block,arm)
    if exists:print("Already submitted",block,arm,flush=True);return
    assert block in range(c["hardware_blocks"]) and arm in ["raw","suppressed"]
    plan_path=out/"hardware_plan.json"; plan=json.loads(plan_path.read_text())
    validation=json.loads((out/"hardware_validation.json").read_text())
    assert validation["passed"] and validation["plan_sha256"]==file_hash(plan_path)
    protocol=json.loads((out/"hardware_protocol.json").read_text())
    assert all(file_hash(ROOT/p)==h for p,h in protocol["source_sha256"].items())
    rows=plan["rows"]+(plan["calibration"] if arm=="raw" else [])
    order=np.random.default_rng(94010+2*block+(arm=="suppressed")).permutation(len(rows))
    rows=[rows[i] for i in order]
    shots=sum(r["shots"] for r in rows)
    assert len(jobs)<c["hardware_max_jobs"] and sum(j["total_shots"] for j in jobs)+shots<=c["hardware_max_shots"]
    service,backend=connect(c); assert backend.status().operational
    pubs=[]
    for row in rows:
        assert file_hash(ROOT/row["isa_qpy"])==row["isa_sha256"]
        qc=qpy_read(ROOT/row["isa_qpy"])
        for inst in qc.data:
            if inst.operation.name=="barrier":continue
            qargs=tuple(qc.find_bit(q).index for q in inst.qubits)
            assert backend.target.instruction_supported(operation_name=inst.operation.name,qargs=qargs)
        pubs.append((qc,None,row["shots"]))
    properties=backend.properties(refresh=True)
    save(out/f"device_calibration_{block}_{arm}.json",json.loads(json.dumps(properties.to_dict(),default=str)))
    sampler=SamplerV2(mode=backend)
    sampler.options.max_execution_time=c["hardware_max_seconds_per_job"]
    sampler.options.dynamical_decoupling.enable=arm=="suppressed"
    if arm=="suppressed":sampler.options.dynamical_decoupling.sequence_type="XX"
    sampler.options.twirling.enable_gates=arm=="suppressed"
    sampler.options.twirling.enable_measure=False
    if arm=="suppressed":
        sampler.options.twirling.num_randomizations=32
        sampler.options.twirling.shots_per_randomization=256
    sampler.options.environment.job_tags=[c["schema_version"],f"b{block}_{arm}",file_hash(plan_path)[:12]]
    record=dict(block=block,arm=arm,rows=rows,total_shots=shots,backend=backend.name,
        backend_version=backend.backend_version,processor=backend.configuration().processor_type,
        plan_sha256=file_hash(plan_path),options=json.loads(json.dumps(sampler.options.__dict__,default=str)))
    save(out/f"intent_{block}_{arm}.json",dict(prepared_utc=now(),**record))
    job=sampler.run(pubs)
    save(out/f"receipt_{block}_{arm}.json",dict(job_id=job.job_id(),received_utc=now()))
    jobs.append(dict(job_id=job.job_id(),submitted_utc=now(),**record)); save(out/"jobs.json",jobs)
    print("Submitted",block,arm,job.job_id(),shots,"shots",flush=True)


def collect():
    from .nighthawk_study import connect,now
    c,out=settings(); jobs=json.loads((out/"jobs.json").read_text())
    pending=[j for j in jobs if not (out/"raw_counts"/f"b{j['block']}_{j['arm']}.json").exists()]
    if not pending:return True
    service,_=connect(c)
    for item in pending:
        job=service.job(item["job_id"]); status=str(job.status()); print(item["job_id"],status,flush=True)
        if status in ["ERROR","CANCELLED","JobStatus.ERROR","JobStatus.CANCELLED"]:
            save(out/f"failure_{item['block']}_{item['arm']}.json",dict(job_id=item["job_id"],status=status,error=job.error_message()))
            raise RuntimeError("Hardware job failed; inspect recorded error")
        if status not in ["DONE","JobStatus.DONE"]:continue
        result=job.result(); assert len(result)==len(item["rows"])
        rows=[]
        for row,pub in zip(item["rows"],result):
            counts=pub.data.meas.get_counts(); assert sum(counts.values())==row["shots"]
            rows.append({**row,"counts":counts,"pub_metadata":json.loads(json.dumps(pub.metadata,default=str))})
        save(out/"raw_counts"/f"b{item['block']}_{item['arm']}.json",dict(collected_utc=now(),job_id=item["job_id"],
            block=item["block"],arm=item["arm"],rows=rows,metrics=job.metrics()))
    return all((out/"raw_counts"/f"b{j['block']}_{j['arm']}.json").exists() for j in jobs)


def guarded(action,block=None,arm=None):
    from .memory_guard import guarded_run
    c,out=settings()
    args=[action]+([str(block),arm] if block is not None else [])
    cmd=[sys.executable,"-m","revision.local_runner","--extra-site",EXTRA,"revision.full_circuit_hardware",*args]
    result=guarded_run(cmd,out/"logs"/("hardware_"+"_".join(args)+".log"),address_gib=4,rss_gib=3,reserve_gib=12,seconds=600)
    save(out/"logs"/("hardware_"+"_".join(args)+"_process.json"),result)
    if result["status"]!="completed":raise RuntimeError(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","submit","collect","guarded-prepare","guarded-submit","guarded-collect"])
    p.add_argument("block",nargs="?",type=int);p.add_argument("arm",nargs="?")
    a=p.parse_args()
    if a.action.startswith("guarded-"):guarded(a.action.removeprefix("guarded-"),a.block,a.arm)
    elif a.action=="submit":submit(a.block,a.arm)
    elif a.action=="prepare":prepare()
    else:collect()
