"""Summarize the new AQC-Tensor prefix pilot without mixing tuning and final data."""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
from .corrected_study import ROOT,file_hash,write_csv,write_json
from .nighthawk_study import now
from .tensor_prefix_prepare import settings
from .tensor_prefix_models import count_metrics
from .analyze_nighthawk import holm,wilson,stratified_exact


def analyze():
    config,out=settings()
    case_records={s["id"]:json.loads((out/"cases"/(s["id"]+".json")).read_text()) for s in config["cases"] if (out/"cases"/(s["id"]+".json")).exists()}
    fits=[]
    for cid,case in case_records.items():
        for row in case["candidates"]:
            fits.append(dict(case_id=cid,qubits=case["model"]["n_qubits"],candidate=row["name"],fidelity=row["fidelity"],
                cz_depth=row["resources"]["cz_depth"],cz_count=row["resources"]["cz_count"],seconds=row["seconds"],
                parameters=row["parameter_count"],iterations=row["iterations"]))
    write_csv(out/"prefix_fit_summary.csv",fits)
    jobs=json.loads((out/"jobs.json").read_text()) if (out/"jobs.json").exists() else []
    rows=[];pooled=defaultdict(lambda:defaultdict(int));block_rows={}
    qpu_seconds=0.
    for job in jobs:
        raw_path=out/"raw_counts"/(job["stage"]+".json")
        if not raw_path.exists():continue
        raw=json.loads(raw_path.read_text())
        qpu_seconds+=float(raw["job_metrics"].get("usage",{}).get("quantum_seconds",0))
        assert raw["job_id"]==job["job_id"] and len(raw["rows"])==len(job["rows"])
        assert file_hash(out/"plans"/(job["stage"]+".json"))==job["plan_sha256"]
        for record in raw["rows"]:
            assert file_hash(ROOT/record["isa_qpy"])==record["isa_sha256"]
            cid,method=record["case_id"],record["method"]
            counts=record["counts"]
            assert sum(counts.values())==job["shots"]
            assert all(len(s.replace(" ",""))==case_records[cid]["model"]["n_qubits"] for s in counts)
            metrics=count_metrics(case_records[cid]["model"],counts)
            row=dict(stage=job["stage"],case_id=cid,method=method,**metrics)
            rows.append(row)
            if job["stage"].startswith("final_"):
                block_rows[job["stage"],cid,method]=row
                for bits,count in counts.items():pooled[cid,method][bits]+=count
    write_csv(out/"hardware_by_stage.csv",rows)
    finals=[]
    for (cid,method),counts in pooled.items():
        row=dict(case_id=cid,method=method,**count_metrics(case_records[cid]["model"],counts))
        for event in ["feasible","good"]:
            lo,hi=wilson(row[event+"_count"],row["shots"])
            row[event+"_ci_low"],row[event+"_ci_high"]=lo,hi
        finals.append(row)
    write_csv(out/"hardware_final.csv",finals)
    comparisons=[]
    complete=all((out/"raw_counts"/f"final_{b}.json").exists() for b in range(config["final_blocks"]))
    if complete:
        indexed={(r["case_id"],r["method"]):r for r in finals}
        for cid in case_records:
            for treatment,reference in [("aqc_prefix_qaoa_initial","uncompressed_prefix_qaoa"),
                ("aqc_prefix_qaoa_initial","standard_p1_initial"),
                ("aqc_prefix_qaoa_refined","aqc_prefix_qaoa_initial"),
                ("aqc_prefix_qaoa_refined","standard_p1_refined")]:
                a,b=indexed[cid,treatment],indexed[cid,reference]
                for event in ["good","feasible"]:
                    k=event+"_count"
                    p=fisher_exact([[a[k],a["shots"]-a[k]],[b[k],b["shots"]-b[k]]]).pvalue
                    tables=[]
                    for block in range(config["final_blocks"]):
                        x,y=block_rows[f"final_{block}",cid,treatment],block_rows[f"final_{block}",cid,reference]
                        tables.append((x[k],x["shots"],y[k],y["shots"]))
                    comparisons.append(dict(case_id=cid,treatment=treatment,reference=reference,event=event,
                        probability_difference=a["p_"+event]-b["p_"+event],p_value=p,
                        stratified_p_value=stratified_exact(tables)))
        for row,p,ps in zip(comparisons,holm([r["p_value"] for r in comparisons]),holm([r["stratified_p_value"] for r in comparisons])):
            row["holm_p_value"],row["stratified_holm_p_value"]=float(p),float(ps)
    write_csv(out/"comparisons.csv",comparisons)
    summary=dict(generated_utc=now(),prepared_cases=len(case_records),fitted_prefixes=len(fits),
        submitted_jobs=len(jobs),completed_jobs=len({r["stage"] for r in rows}),
        hardware_shots=sum(r["shots"] for r in rows),reported_qpu_seconds=qpu_seconds,final_evaluation_complete=complete,
        independent_instances=len(case_records),source="ibm_rensselaer Nighthawk; new acquisition, not prior data",
        scope=config["scope"],comparisons=comparisons)
    write_json(out/"summary.json",summary)
    text=["# Larger-instance AQC-Tensor prefixes for QAOA", "",
        f"Status: {'completed hardware refinement and held-out evaluation' if complete else 'in progress; final hardware evaluation incomplete'}.","",
        "This exploratory pilot uses real AQC-Tensor MPS fidelity optimization and shallow ansatzes confined to physical CZ edges. "
        "The complete QAOA cost layer retains all problem interactions and can require routing. Exact-state correctness checks use small cases; "
        "larger targets are checked by increasing MPS bond limits. Agreement between finite bond limits is a convergence diagnostic, not a proof of exactness.","",
        "The planned instances are FLP at 20 and 30 qubits (four and five facilities) and TSP at 16 and 25 qubits (five and six cities). "
        "There is one instance per model/size. Exact classical references use facility subsets or tours, without enumerating all measured strings. "
        "These instances remain easy for those classical methods.","",
        "## Native prefix candidates","",
        "[Native-prefix fidelity/depth figure](prefix_tradeoffs.pdf).", "",
        "| Case | Candidate | MPS fidelity | Native CZ depth | Fit seconds |",
        "|---|---|---:|---:|---:|"]
    for r in fits:text.append(f"| {r['case_id']} | {r['candidate']} | {r['fidelity']:.6f} | {r['cz_depth']} | {r['seconds']:.2f} |")
    text += ["","## Target convergence and unmet fidelity thresholds",""]
    for cid,case in case_records.items():
        if "excluded_original_endpoints" in case:
            text.append(f"- {cid}: original endpoints {case['excluded_original_endpoints']} were excluded after convergence failures; "
                        "a separately recorded pre-hardware fallback retreats to the quarter endpoint with the same threshold and budgets.")
        for tag,target in case["targets"].items():
            if target.get("status")=="excluded_unconverged":
                text.append(f"- {cid}/{tag}: excluded before hardware; successive-bond fidelities {target['convergence']}.")
            else:
                method=case["methods"]["aqc_"+tag+"_prefix"]
                text.append(f"- {cid}/{tag}: target bond {target['selected_bond']}, achieved bond {target['achieved_bond']}; "
                    f"selected {method['selected_fit']}; .98 fit threshold {'met' if method['fidelity_threshold_met'] else 'NOT met'}.")
    text += ["","Local QAOA angles are initialized using finite-bond mean-energy estimates. These are approximate training inputs, not exact quality certificates. "
        "Both full prefixed-QAOA circuits and the ordinary-QAOA baselines are evaluated on hardware. "
        "Native/shallow/uncompressed prefixes have separately trained QAOA angles, so their hardware comparison is an end-to-end circuit comparison, "
        "not a controlled test holding the continuation unitary fixed.","",
        "| Case | Current preparation pass (s) | Peak RSS (MiB) | Largest 64/128 energy difference |",
        "|---|---:|---:|---:|"]
    for cid,case in case_records.items():
        difference=max(m["training"]["energy_bond_difference"] for m in case["methods"].values() if "training" in m)
        text.append(f"| {cid} | {case['preparation_sec']:.1f} | {case['peak_process_rss_mib']:.1f} | {difference:.6g} |")
    text += ["","Preparation-pass timings exclude earlier failed engineering attempts and can reuse archived fitted angles. "
        "Fit timings and the original attempts are retained separately; these timings are not a complete end-to-end optimization cost comparison."]
    text += ["","## Hardware evaluation","",
        f"Submitted {len(jobs)} jobs; collected {summary['hardware_shots']:,} shots and {qpu_seconds:.1f} reported QPU seconds. "
        "Screening selects the prefix endpoint; three SPSA updates refine two QAOA angles for both prefixed and ordinary p=1 QAOA. "
        "The prefix parameters remain fixed after offline AQC-Tensor fitting. Two later randomized blocks evaluate the frozen final circuits. "
        "No final counts are used to choose the endpoint or tune angles.","",
        "| Case | Final method | Feasible / shots | Within 5% / shots | CVaR (lower is better) |",
        "|---|---|---:|---:|---:|"]
    for r in finals:text.append(f"| {r['case_id']} | {r['method']} | {r['feasible_count']} / {r['shots']} | {r['good_count']} / {r['shots']} | {r['cvar']:.5f} |")
    text += ["","Final event comparisons are corrected together across 32 tests, with a block-stratified sensitivity analysis. "
        "Pointwise Wilson intervals and all counts are released. CVaR training improvements are not evidence of improved feasible/useful sampling. "
        "One instance at each width does not support population or scaling-advantage claims.","",
        "[Held-out before/after refinement figure](hardware_refinement.pdf) is generated when final data are complete; lines are descriptive point estimates.","",
        "## Reproduction and provenance","",
        "See `TENSOR_PREFIX_STUDY.md` at the repository root, the frozen config/protocols, MPS convergence records, all fitted candidates, "
        "hardware plans, job IDs, calibration snapshots, and per-stage raw counts. Earlier paper data are untouched. "
        "Preflight records retain the coupling-map adapter failure and the pre-hardware amendment excluding unconverged targets. "
        "Previously successful fits keep their original protocol hashes and are reused only under unchanged numerical settings.","",
        "Uses the [official AQC-Tensor fidelity objective](https://qiskit.github.io/qiskit-addon-aqc-tensor/apidocs/objective.html) "
        "with its [Quimb backend](https://qiskit.github.io/qiskit-addon-aqc-tensor/stubs/qiskit_addon_aqc_tensor.simulation.quimb.QuimbSimulator.html).",""]
    (out/"report.md").write_text("\n".join(text))
    print(json.dumps({k:v for k,v in summary.items() if k!='comparisons'},indent=2))


if __name__=="__main__":analyze()
