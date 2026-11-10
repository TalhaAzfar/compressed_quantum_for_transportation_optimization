"""Independent artifact, physical-coupling, measurement, and count checks."""
import json
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
from .corrected_study import ROOT,file_hash,write_json
from .nighthawk_study import qpy_read,now
from .tensor_prefix_prepare import settings


def independent_decode(model,bitstring):
    """Scalar decoder, independent of the vectorized experiment decoder."""
    x=[int(c) for c in bitstring.replace(" ","")[::-1]]
    if model["problem"]=="flp":
        c,f=model["customers"],model["facilities"]
        cost=sum(model["opening"][j]*x[j] for j in range(f))
        violations=0
        for i in range(c):
            assigned=[j for j in range(f) if x[f+i*f+j]]
            violations+=(len(assigned)-1)**2+sum(1-x[j] for j in assigned)
            cost+=sum(model["service"][i][j] for j in assigned)
    else:
        m=model["nodes"]-1
        positions=[[i+1 for i in range(m) if x[t*m+i]] for t in range(m)]
        violations=sum((len(s)-1)**2 for s in positions)
        violations+=sum((sum(city in s for s in positions)-1)**2 for city in range(1,m+1))
        distance=model["distance"]
        cost=sum(distance[0][city] for city in positions[0])+sum(distance[city][0] for city in positions[-1])
        cost+=sum(distance[a][b] for left,right in zip(positions,positions[1:]) for a in left for b in right)
    feasible=violations==0
    return dict(feasible=bool(feasible),good=bool(feasible and cost<=model["optimum"]*1.05+1e-9),
                optimal=bool(feasible and abs(cost-model["optimum"])<=1e-9),
                cost=cost,energy=(cost+model["penalty"]*violations)/model["scale"])


def check_circuit(row,case,target,allowed):
    assert file_hash(ROOT/row["isa_qpy"])==row["isa_sha256"]
    qc=qpy_read(ROOT/row["isa_qpy"])
    resources=row["resources"]
    physical=case["patch"]["physical"]
    for inst in qc.data:
        qargs=tuple(qc.find_bit(q).index for q in inst.qubits)
        if inst.operation.num_qubits==2:
            assert inst.operation.name=="cz" and tuple(sorted(qargs)) in allowed
        assert all(q in physical for q in qargs)
        assert target.instruction_supported(operation_name=inst.operation.name,qargs=qargs)
    measured={qc.find_bit(inst.clbits[0]).index:qc.find_bit(inst.qubits[0]).index for inst in qc.data if inst.operation.name=="measure"}
    assert [measured[i] for i in range(case["model"]["n_qubits"])]==[physical[i] for i in resources["final_logical_mapping"]]
    assert qc.count_ops().get("cz",0)==resources["cz_count"]
    assert qc.depth(lambda inst:inst.operation.name=="cz")==resources["cz_depth"]
    assert resources["conservative_accumulated_matrix_error_bound"]<1e-8
    return resources["maximum_gate_matrix_error"]


def validate():
    config,out=settings()
    protocol=json.loads((out/"protocol.json").read_text())
    for name,digest in protocol["source_sha256"].items():
        assert file_hash(ROOT/name)==digest
        assert file_hash(out/"source_snapshot"/Path(name).name)==digest
    assert file_hash(out/"preflight/target_snapshot.pkl")==protocol["target_sha256"]
    target=pickle.loads((out/"preflight/target_snapshot.pkl").read_bytes())
    allowed={tuple(sorted(e)) for e in target["cz"]}
    circuits_checked=0;max_matrix_error=0.
    cases={};checked_hashes=set()
    for spec in config["cases"]:
        case=json.loads((out/"cases"/(spec["id"]+".json")).read_text());cases[spec["id"]]=case
        if "routing_amendment" in case:
            parent_path=ROOT/case["routing_amendment"]["parent_case"]
            assert file_hash(parent_path)==case["routing_amendment"]["parent_case_sha256"]
            parent=json.loads(parent_path.read_text())
            for field in ["model","patch","targets","candidates"]:assert case[field]==parent[field]
            for name,method in case["methods"].items():
                for field in ["logical_sha256","logical_qpy","training","prefix_qpy"]:
                    if field in method:assert method[field]==parent["methods"][name][field]
        for method in case["methods"].values():
            assert file_hash(ROOT/method["logical_qpy"])==method["logical_sha256"]
            if "prefix_sha256" in method.get("training",{}):
                assert file_hash(ROOT/method["prefix_qpy"])==method["training"]["prefix_sha256"]
            elif "training" in method:
                # Fallback preparations predate the cached-training prefix hash.
                # Reconstruct the complete initial circuit from its prefix and
                # recorded angles and compare it with the frozen logical circuit.
                from .tensor_prefix_core import qaoa
                rebuilt=qaoa(qpy_read(ROOT/method["prefix_qpy"]),case["model"],method["training"]["parameters"])
                assert rebuilt==qpy_read(ROOT/method["logical_qpy"])
            max_matrix_error=max(max_matrix_error,check_circuit(method,case,target,allowed))
            checked_hashes.add(method["isa_sha256"])
            circuits_checked+=1
    independent_shots=0;count_sets=0;hardware_circuits_checked=0;final_complete=False
    independent_cvar={};updates_checked=0
    if (out/"hardware_protocol.json").exists():
        hw=json.loads((out/"hardware_protocol.json").read_text())
        assert file_hash(out/"protocol.json")==hw["preparation_protocol_sha256"]
        if hw.get("fallback_protocol_sha256"):
            assert file_hash(out/"fallback_protocol.json")==hw["fallback_protocol_sha256"]
            fallback=json.loads((out/"fallback_protocol.json").read_text())
            for name,digest in fallback["source_sha256"].items():assert file_hash(ROOT/name)==digest
        for name,digest in hw["source_sha256"].items():assert file_hash(ROOT/name)==digest
        for cid,digest in hw["case_sha256"].items():assert file_hash(out/"cases"/(cid+".json"))==digest
    if (out/"jobs.json").exists():
        from .tensor_prefix_models import count_metrics
        jobs=json.loads((out/"jobs.json").read_text())
        assert len(jobs)<=config["max_jobs"]
        assert sum(j["shots"]*len(j["rows"]) for j in jobs)<=config["max_total_shots"]
        assert len({j["stage"] for j in jobs})==len(jobs)
        for job in jobs:
            plan_path=out/"plans"/(job["stage"]+".json")
            assert file_hash(plan_path)==job["plan_sha256"]
            plan=json.loads(plan_path.read_text())
            assert plan["hardware_protocol_sha256"]==file_hash(out/"hardware_protocol.json")
            assert plan["shots"]==job["shots"]
            assert datetime.fromisoformat(plan["prepared_utc"])<=datetime.fromisoformat(job["submitted_utc"])
            planned={(r["case_id"],r["method"]):r for r in plan["rows"]}
            assert len(planned)==len(job["rows"])
            for row in job["rows"]:
                assert row==planned[row["case_id"],row["method"]]
                assert file_hash(ROOT/row["isa_qpy"])==row["isa_sha256"]
                if row["isa_sha256"] not in checked_hashes:
                    max_matrix_error=max(max_matrix_error,check_circuit(row,cases[row["case_id"]],target,allowed))
                    checked_hashes.add(row["isa_sha256"]);hardware_circuits_checked+=1
            raw_path=out/"raw_counts"/(job["stage"]+".json")
            if not raw_path.exists():continue
            raw=json.loads(raw_path.read_text())
            assert raw["job_id"]==job["job_id"] and raw["stage"]==job["stage"]
            assert len(raw["rows"])==len(job["rows"])
            for row,submitted in zip(raw["rows"],job["rows"]):
                assert all(row[key]==value for key,value in submitted.items())
                model=cases[row["case_id"]]["model"]
                events={event:0 for event in ["feasible","good","optimal"]};energies=[];costs=[]
                for bits,count in row["counts"].items():
                    assert count>0 and len(bits.replace(" ",""))==model["n_qubits"]
                    decoded=independent_decode(model,bits)
                    for event in events:events[event]+=decoded[event]*count
                    energies.extend([decoded["energy"]]*count)
                    if decoded["feasible"]:costs.append(decoded["cost"])
                    independent_shots+=count
                calculated=count_metrics(model,row["counts"])
                for event,value in events.items():assert value==calculated[event+"_count"]
                tail_size=.2*len(energies);whole=int(tail_size);fraction=tail_size-whole
                ordered=sorted(energies)
                cvar=(sum(ordered[:whole])+fraction*ordered[whole])/tail_size
                assert abs(cvar-calculated["cvar"])<1e-10
                independent_cvar[job["stage"],row["case_id"],row["method"]]=cvar
                assert abs(sum(energies)/len(energies)-calculated["mean_energy"])<1e-10
                if costs:assert abs(min(costs)-calculated["best_feasible_cost"])<1e-9
                else:assert calculated["best_feasible_cost"] is None
                assert calculated["shots"]==job["shots"]
                count_sets+=1
        if (out/"screen_selection.json").exists():
            selected=json.loads((out/"screen_selection.json").read_text())
            assert selected["screen_data_sha256"]==file_hash(out/"raw_counts/screen.json")
            current={}
            for cid,item in selected["cases"].items():
                tags=[name.removeprefix("aqc_") for name,m in cases[cid]["methods"].items() if name.startswith("aqc_") and "training" in m]
                expected=min(tags,key=lambda tag:(independent_cvar["screen",cid,"aqc_"+tag],tag))
                assert item["endpoint"]==expected
                for policy,original in [("aqc","aqc_"+expected),("standard","standard_p1")]:
                    current[cid,policy]=np.array(cases[cid]["methods"][original]["training"]["parameters"])
                    assert np.allclose(current[cid,policy],item["methods"][policy]["parameters"],atol=1e-12,rtol=0)
            for step in range(config["spsa_steps"]):
                stage=f"refine_{step}";path=out/"plans"/(stage+".json")
                if not path.exists():break
                plan=json.loads(path.read_text());planned={(r["case_id"],r["method"]):r for r in plan["rows"]}
                for update in plan["metadata"]["updates"]:
                    cid,policy=update["case_id"],update["policy"]
                    assert np.allclose(current[cid,policy],update["theta"],atol=1e-8,rtol=0)
                    center=np.array(update["perturbation_center"]);delta=np.array(update["delta"])
                    for sign,label in [(1,"plus"),(-1,"minus")]:
                        assert np.allclose(planned[cid,policy+"_"+label]["parameters"],center+sign*update["c"]*delta,atol=1e-12,rtol=0)
                    key=stage,cid,policy+"_plus"
                    if key in independent_cvar:
                        difference=independent_cvar[key]-independent_cvar[stage,cid,policy+"_minus"]
                        displacement=np.clip(update["a"]*difference/(2*update["c"])*delta,-.15,.15)
                        current[cid,policy]=np.clip(current[cid,policy]-displacement,[-.6,-.8],[.6,.8])
                        updates_checked+=1
                if not (out/"raw_counts"/(stage+".json")).exists():break
            if (out/"plans/final_0.json").exists():
                plan=json.loads((out/"plans/final_0.json").read_text())
                for row in plan["rows"]:
                    policy={"aqc_prefix_qaoa_refined":"aqc","standard_p1_refined":"standard"}.get(row["method"])
                    if policy:assert np.allclose(row["parameters"],current[row["case_id"],policy],atol=1e-8,rtol=0)
        final_jobs=[j for j in jobs if j["stage"].startswith("final_")]
        if final_jobs:
            history_path=out/"refinement_history.json"
            frozen=datetime.fromisoformat(json.loads(history_path.read_text())["final_parameters_frozen_utc"])
            for stage in ["screen"]+[f"refine_{k}" for k in range(config["spsa_steps"])]:
                collected=json.loads((out/"raw_counts"/(stage+".json")).read_text())["collected_utc"]
                assert datetime.fromisoformat(collected)<frozen
            reference_rows=None
            for job in final_jobs:
                plan=json.loads((out/"plans"/(job["stage"]+".json")).read_text())
                assert plan["metadata"]["refinement_history_sha256"]==file_hash(history_path)
                assert frozen<=datetime.fromisoformat(plan["prepared_utc"])<min(datetime.fromisoformat(j["submitted_utc"]) for j in final_jobs)
                rows={(r["case_id"],r["method"]):r["isa_sha256"] for r in plan["rows"]}
                if reference_rows is None:reference_rows=rows
                else:assert rows==reference_rows
            final_complete=all((out/"raw_counts"/f"final_{b}.json").exists() for b in range(config["final_blocks"]))
    result=dict(validated_utc=now(),prepared_cases=len(cases),prepared_circuits_checked=circuits_checked,
                independent_count_sets=count_sets,independently_decoded_shots=independent_shots,
                additional_hardware_circuits_checked=hardware_circuits_checked,final_evaluation_complete=final_complete,
                independent_spsa_updates_checked=updates_checked,
                maximum_elementary_matrix_error=max_matrix_error,passed=True)
    write_json(out/"independent_validation.json",result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":validate()
