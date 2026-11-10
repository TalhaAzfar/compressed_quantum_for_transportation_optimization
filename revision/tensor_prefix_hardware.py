"""Bounded, resumable hardware screening, SPSA refinement and held-out evaluation."""
from __future__ import annotations
import argparse
import json
import pickle
import shutil
from pathlib import Path
import numpy as np

from .corrected_study import ROOT, file_hash, write_json
from .nighthawk_study import connect, now, qpy_read, qpy_write
from .tensor_prefix_prepare import settings, freeze
from .tensor_prefix_core import compile_isa, qaoa
from .tensor_prefix_models import count_metrics


def cases():
    config,out=settings()
    return {s["id"]:json.loads((out/"cases"/(s["id"]+".json")).read_text()) for s in config["cases"]}


def freeze_hardware():
    freeze()
    config,out=settings()
    for folder in ["plans","raw_counts","hardware_circuits"]:
        (out/folder).mkdir(exist_ok=True)
    hashes={name:file_hash(ROOT/name) for name in ["revision/tensor_prefix_hardware.py","revision/tensor_prefix_models.py","revision/tensor_prefix_core.py","revision/analyze_tensor_prefix.py"]}
    path=out/"hardware_protocol.json"
    if path.exists():
        record=json.loads(path.read_text())
        assert record["source_sha256"]==hashes,"Hardware source changed"
        for cid,digest in record["case_sha256"].items():
            assert file_hash(out/"cases"/(cid+".json"))==digest
    else:
        write_json(path,dict(frozen_utc=now(),source_sha256=hashes,config=config,
            preparation_protocol_sha256=file_hash(out/"protocol.json"),
            fallback_protocol_sha256=file_hash(out/"fallback_protocol.json") if (out/"fallback_protocol.json").exists() else None,
            case_sha256={cid:file_hash(out/"cases"/(cid+".json")) for cid in cases()}))
        for name in hashes:
            shutil.copy2(ROOT/name,out/"source_snapshot"/Path(name).name)


def write_plan(stage,rows,shots,metadata=None):
    config,out=settings()
    path=out/"plans"/(stage+".json")
    payload=dict(stage=stage,prepared_utc=now(),shots=shots,rows=rows,metadata=metadata or {},
                 hardware_protocol_sha256=file_hash(out/"hardware_protocol.json"))
    if path.exists():
        old=json.loads(path.read_text())
        assert old["rows"]==rows and old["shots"]==shots,"Existing plan differs"
        return
    write_json(path,payload)
    print("Prepared",stage,len(rows),"circuits",shots,"shots each",flush=True)


def screen_plan():
    freeze_hardware()
    config,out=settings()
    rows=[]
    for cid,case in cases().items():
        for name,method in case["methods"].items():
            rows.append(dict(case_id=cid,method=name,isa_qpy=method["isa_qpy"],isa_sha256=method["isa_sha256"]))
    assert 24 <= len(rows) <= 40
    write_plan("screen",rows,config["hardware_screen_shots"])


def count_rows(stage):
    _,out=settings()
    record=json.loads((out/"raw_counts"/(stage+".json")).read_text())
    return {(r["case_id"],r["method"]):r for r in record["rows"]}


def selection():
    config,out=settings()
    path=out/"screen_selection.json"
    if path.exists():
        return json.loads(path.read_text())
    rows=count_rows("screen")
    chosen={}
    for cid,case in cases().items():
        tags=[name.removeprefix("aqc_") for name,item in case["methods"].items() if name.startswith("aqc_") and "training" in item]
        results={tag:count_metrics(case["model"],rows[cid,"aqc_"+tag]["counts"]) for tag in tags}
        assert results,"No converged endpoint available"
        tag=min(results,key=lambda key:(results[key]["cvar"],key))
        chosen[cid]=dict(endpoint=tag,screen_metrics=results,methods={})
        for key,method in [("aqc","aqc_"+tag),("standard","standard_p1")]:
            chosen[cid]["methods"][key]=dict(original_method=method,
                parameters=case["methods"][method]["training"]["parameters"],
                prefix_qpy=case["methods"][method]["prefix_qpy"])
    result=dict(selected_utc=now(),screen_data_sha256=file_hash(out/"raw_counts/screen.json"),cases=chosen)
    write_json(path,result)
    return result


def theta_at(step):
    """Step zero is screening choice; each later state uses only earlier tuning shots."""
    config,out=settings()
    selected=selection()
    current={(cid,key):np.array(method["parameters"]) for cid,item in selected["cases"].items() for key,method in item["methods"].items()}
    history=[]
    for k in range(step):
        stage=f"refine_{k}"
        plan=json.loads((out/"plans"/(stage+".json")).read_text())
        raw=count_rows(stage)
        for item in plan["metadata"]["updates"]:
            cid,key=item["case_id"],item["policy"]
            assert np.allclose(current[cid,key],item["theta"])
            model=cases()[cid]["model"]
            plus=count_metrics(model,raw[cid,key+"_plus"]["counts"])["cvar"]
            minus=count_metrics(model,raw[cid,key+"_minus"]["counts"])["cvar"]
            delta=np.array(item["delta"])
            # Clip an update, not its recorded observations, to keep three noisy
            # steps bounded and the angle search identical for both policies.
            gradient=(plus-minus)/(2*item["c"])*delta
            displacement=np.clip(item["a"]*gradient,-.15,.15)
            theta=np.clip(current[cid,key]-displacement,[-.6,-.8],[.6,.8])
            current[cid,key]=theta
            history.append(dict(step=k,case_id=cid,policy=key,plus_cvar=plus,minus_cvar=minus,
                                gradient=gradient.tolist(),next_parameters=theta.tolist()))
    return current,history


def new_circuit(stage,cid,name,prefix_file,theta,case):
    config,out=settings()
    target=pickle.loads((out/"preflight/target_snapshot.pkl").read_bytes())
    qc=qaoa(qpy_read(ROOT/prefix_file),case["model"],theta)
    isa,resources=compile_isa(qc,target,case["patch"]["physical"],[tuple(e) for e in case["patch"]["edges"]])
    path=out/"hardware_circuits"/f"{stage}_{cid}_{name}.qpy"
    qpy_write(path,isa)
    return dict(case_id=cid,method=name,parameters=np.asarray(theta).tolist(),resources=resources,
                isa_qpy=str(path.relative_to(ROOT)),isa_sha256=file_hash(path))


def refinement_plan(step):
    freeze_hardware()
    config,out=settings()
    assert 0 <= step < config["spsa_steps"]
    if (out/"plans"/f"refine_{step}.json").exists():
        print("Refinement plan already exists",step);return
    selected=selection()
    current,history=theta_at(step)
    rows,updates=[],[]
    rng=np.random.default_rng(config["spsa_seed"]+step)
    for cid,case in cases().items():
        for key in ["aqc","standard"]:
            theta=current[cid,key]
            delta=rng.choice([-1.,1.],size=2)
            c=config["spsa_c"]/(step+1)**.101
            a=config["spsa_a"]/(step+1)**.602
            # Symmetric perturbations stay within the bounded search domain.
            c=min(c,max(.001,min(.6-abs(theta[0]),.8-abs(theta[1]))))
            # Center away from the boundary so the finite difference remains symmetric.
            theta=np.clip(theta,[-.6+c,-.8+c],[.6-c,.8-c])
            prefix=selected["cases"][cid]["methods"][key]["prefix_qpy"]
            updates.append(dict(case_id=cid,policy=key,theta=current[cid,key].tolist(),
                                perturbation_center=theta.tolist(),delta=delta.tolist(),a=a,c=c))
            for sign,label in [(1,"plus"),(-1,"minus")]:
                rows.append(new_circuit(f"refine_{step}",cid,key+"_"+label,prefix,theta+sign*c*delta,case))
    write_plan(f"refine_{step}",rows,config["spsa_shots"],dict(updates=updates,previous_history=history))


def final_plans():
    freeze_hardware()
    config,out=settings()
    if all((out/"plans"/f"final_{b}.json").exists() for b in range(config["final_blocks"])):
        print("Final plans already frozen");return
    selected=selection(); current,history=theta_at(config["spsa_steps"])
    rows=[]
    for cid,case in cases().items():
        tag=selected["cases"][cid]["endpoint"]
        for name,original in [("standard_p1_initial","standard_p1"),("standard_p2","standard_p2"),
            ("shallow_prefix_qaoa","shallow_"+tag),("uncompressed_prefix_qaoa","refined_"+tag),
            ("aqc_prefix_qaoa_initial","aqc_"+tag)]:
            method=case["methods"][original]
            rows.append(dict(case_id=cid,method=name,isa_qpy=method["isa_qpy"],isa_sha256=method["isa_sha256"]))
        for key,name in [("aqc","aqc_prefix_qaoa_refined"),("standard","standard_p1_refined")]:
            method=selected["cases"][cid]["methods"][key]
            rows.append(new_circuit("final",cid,name,method["prefix_qpy"],current[cid,key],case))
    write_json(out/"refinement_history.json",dict(final_parameters_frozen_utc=now(),history=history))
    for block in range(config["final_blocks"]):
        write_plan(f"final_{block}",rows,config["final_shots"],dict(refinement_history_sha256=file_hash(out/"refinement_history.json")))


def submit(stage):
    from qiskit_ibm_runtime import SamplerV2
    freeze_hardware()
    config,out=settings()
    jobs_path=out/"jobs.json"
    jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    if any(j["stage"]==stage for j in jobs):
        print("Already submitted",stage);return
    plan_path=out/"plans"/(stage+".json")
    plan=json.loads(plan_path.read_text())
    assert len(jobs)<config["max_jobs"]
    shots=sum(j["shots"]*len(j["rows"]) for j in jobs)+plan["shots"]*len(plan["rows"])
    assert shots<=config["max_total_shots"],shots
    _,backend=connect(config)
    assert backend.status().operational
    live_edges={tuple(sorted(e)) for e in backend.target["cz"]}
    stage_seed=sum((i+1)*ord(ch) for i,ch in enumerate(stage))
    order=np.random.default_rng(85101+stage_seed).permutation(len(plan["rows"]))
    rows=[plan["rows"][i] for i in order]
    circuits=[]
    for row in rows:
        path=ROOT/row["isa_qpy"]
        assert file_hash(path)==row["isa_sha256"]
        qc=qpy_read(path)
        for inst in qc.data:
            if inst.operation.num_qubits==2:
                assert tuple(sorted(qc.find_bit(q).index for q in inst.qubits)) in live_edges
        circuits.append(qc)
    calibration=backend.properties(refresh=True)
    (out/f"calibration_{stage}.json").write_text(json.dumps(calibration.to_dict(),default=str,indent=2)+'\n')
    sampler=SamplerV2(mode=backend)
    sampler.options.max_execution_time=config["max_execution_seconds_per_job"]
    sampler.options.dynamical_decoupling.enable=False
    sampler.options.twirling.enable_gates=False
    sampler.options.twirling.enable_measure=False
    sampler.options.environment.job_tags=[config["schema_version"],stage]
    job=sampler.run(circuits,shots=plan["shots"])
    record=dict(stage=stage,job_id=job.job_id(),submitted_utc=now(),shots=plan["shots"],rows=rows,
                backend=backend.name,backend_version=backend.backend_version,
                processor_type=backend.configuration().processor_type,
                calibration_timestamp=str(calibration.last_update_date),plan_sha256=file_hash(plan_path))
    jobs.append(record);write_json(jobs_path,jobs)
    print("Submitted",stage,job.job_id(),len(rows),"circuits",flush=True)


def collect():
    config,out=settings()
    service,_=connect(config)
    for record in json.loads((out/"jobs.json").read_text()):
        path=out/"raw_counts"/(record["stage"]+".json")
        if path.exists():
            continue
        job=service.job(record["job_id"])
        status=str(job.status())
        print(record["stage"],record["job_id"],status,flush=True)
        if status not in ["DONE","JobStatus.DONE"]:
            if status in ["ERROR","CANCELLED","JobStatus.ERROR","JobStatus.CANCELLED"]:
                write_json(out/(record["stage"]+"_failure.json"),dict(status=status,message=job.error_message()))
            continue
        result=job.result()
        assert len(result)==len(record["rows"])
        rows=[]
        for row,pub in zip(record["rows"],result):
            counts=pub.data.meas.get_counts()
            assert sum(counts.values())==record["shots"]
            rows.append({**row,"counts":counts,"metadata":pub.metadata})
        payload=dict(stage=record["stage"],job_id=record["job_id"],collected_utc=now(),rows=rows,
                     job_metrics=job.metrics(),result_metadata=result.metadata)
        path.write_text(json.dumps(payload,indent=2,default=str)+'\n')


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["screen-plan","refine-plan","final-plans","submit","collect"])
    p.add_argument("value",nargs="?")
    args=p.parse_args()
    if args.action=="screen-plan":screen_plan()
    elif args.action=="refine-plan":refinement_plan(int(args.value))
    elif args.action=="final-plans":final_plans()
    elif args.action=="submit":submit(args.value)
    else:collect()
