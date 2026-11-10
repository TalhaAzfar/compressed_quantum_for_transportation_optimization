"""Raw event counts, local readout correction and uncertainty for the pilot."""
import argparse
import json
import sys
from collections import Counter

from .full_circuit_compression import ROOT,EXTRA,settings,save


def analyze():
    import numpy as np
    from scipy.stats import fisher_exact
    from .tensor_prefix_models import count_metrics
    from .validate_tensor_prefix import independent_decode
    from .prefix_diagnostic import feasible_assignments
    from .statistics import holm_correction,proportion_ci
    from .audit_tensor_prefix_statistics import conditional_two_sided
    from .corrected_study import file_hash,write_csv
    from .nighthawk_study import now
    from .readout_indicator_mitigation import assignment_matrices,event_weights,count_arrays,resample_counts
    c,out=settings(); plan=json.loads((out/"hardware_plan.json").read_text())
    methods={r["method"]:r for r in plan["rows"]}; model=plan["model"];n=model["n_qubits"]
    paths=sorted((out/"raw_counts").glob("b*.json"))
    assert len(paths)==2*c["hardware_blocks"]
    blocks=[json.loads(p.read_text()) for p in paths]
    assert {(b["block"],b["arm"]) for b in blocks}=={(b,a) for b in range(c["hardware_blocks"]) for a in plan["arms"]}
    source_names=["revision/analyze_full_circuit_mitigation.py","revision/readout_indicator_mitigation.py"]
    provenance=dict(source_sha256={p:file_hash(ROOT/p) for p in source_names},
        input_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in paths+[out/"hardware_plan.json"]})
    protocol_path=out/"analysis_protocol.json"
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text());assert all(old[k]==v for k,v in provenance.items())
    else:save(protocol_path,dict(frozen_utc=now(),replicates=c["bootstrap_replicates"],**provenance))
    raw=[]; pooled={}; byblock={}; calibrations={}; total_shots=0;qpu_seconds=0.
    for block in blocks:
        qpu_seconds+=float(block["metrics"].get("usage",{}).get("quantum_seconds",0))
        cal={}
        for row in block["rows"]:
            total_shots+=sum(row["counts"].values())
            assert sum(row["counts"].values())==row["shots"]
            if row["kind"]!="experiment":cal[row["method"]]=row;continue
            metric=count_metrics(model,row["counts"])
            independent={"feasible_count":0,"good_count":0}
            for key,count in row["counts"].items():
                check=independent_decode(model,key)
                independent["feasible_count"]+=count*bool(check["feasible"])
                independent["good_count"]+=count*bool(check["good"])
            assert all(metric[k]==v for k,v in independent.items())
            raw.append(dict(block=block["block"],arm=block["arm"],method=row["method"],**metric))
            key=(block["arm"],row["method"])
            pooled.setdefault(key,Counter()).update(row["counts"])
            byblock[(block["block"],*key)]=row
        if cal:calibrations[block["block"]]=cal
    assert total_shots==plan["expected_shots"]
    write_csv(out/"hardware_by_block.csv",raw)
    pooled_rows=[]
    for (arm,method),counts in pooled.items():
        metric=count_metrics(model,dict(counts));row=dict(arm=arm,method=method,**metric)
        for event in ["feasible","good"]:
            interval=proportion_ci(metric[event+"_count"],sum(counts.values()))
            row.update(interval.as_dict(event+"_ci"))
        pooled_rows.append(row)
    write_csv(out/"hardware_pooled.csv",pooled_rows)
    contrasts=[]
    for layers in c["hardware_layers"]:
        a=f"full_aqc_qaoa_l{layers}";b=f"full_product_qaoa_l{layers}"
        for arm in plan["arms"]:
            for event in ["feasible","good"]:
                if a not in methods or b not in methods:
                    contrasts.append(dict(layers=layers,arm=arm,event=event,p=1.,stratified_p=1.,status="unavailable_pair"));continue
                ra=next(r for r in pooled_rows if r["arm"]==arm and r["method"]==a)
                rb=next(r for r in pooled_rows if r["arm"]==arm and r["method"]==b)
                na=sum(pooled[(arm,a)].values());nb=sum(pooled[(arm,b)].values())
                ka,kb=ra[event+"_count"],rb[event+"_count"]
                p=float(fisher_exact([[ka,na-ka],[kb,nb-kb]]).pvalue)
                tables=[]
                for block in range(c["hardware_blocks"]):
                    left=next(r for r in raw if r["block"]==block and r["arm"]==arm and r["method"]==a)
                    right=next(r for r in raw if r["block"]==block and r["arm"]==arm and r["method"]==b)
                    tables.append((left[event+"_count"],left["shots"],right[event+"_count"],right["shots"]))
                contrasts.append(dict(layers=layers,arm=arm,event=event,p=p,status="evaluated",
                    stratified_p=conditional_two_sided(tables),
                    left=a,right=b,left_count=ka,right_count=kb,left_shots=na,right_shots=nb,difference=ka/na-kb/nb))
    for row,p in zip(contrasts,holm_correction(r["p"] for r in contrasts)):row["holm_p"]=p
    for row,p in zip(contrasts,holm_correction(r["stratified_p"] for r in contrasts)):row["stratified_holm_p"]=p
    write_csv(out/"primary_contrasts.csv",contrasts)
    feasible,costs=feasible_assignments(model)
    indicators=np.zeros((1<<n,2));indicators[feasible,0]=1;indicators[feasible[costs<=1.05*model["optimum"]+1e-9],1]=1
    keys=sorted(pooled); point=np.zeros((len(keys),2)); rawpoint=np.zeros_like(point)
    boots=np.zeros((c["bootstrap_replicates"],len(keys),2)); rawboots=np.zeros_like(boots)
    rng=np.random.default_rng(c["bootstrap_seed"]); readout_records=[]; checks=[]
    for block in range(c["hardware_blocks"]):
        cal=calibrations[block];zero=cal["readout_0"]["counts"];one=cal["readout_1"]["counts"]
        matrices=assignment_matrices(zero,one,n)
        readout_records.append(dict(block=block,assignment_matrices=matrices.tolist(),
            mean_p1_given0=float(matrices[:,1,0].mean()),mean_p0_given1=float(matrices[:,0,1].mean()),
            max_condition=float(np.max(np.linalg.cond(matrices)))))
        # Held-out per-bit tests compare with the same prepared bit calibration.
        for name,r in cal.items():
            if r["kind"]!="calibration_check":continue
            values,counts=count_arrays(r["counts"],n);shots=int(counts.sum())
            for q in range(n):
                prepared=r["prepared"]>>q&1;observed=int(counts@((values>>q)&1))
                vv,cc=count_arrays(one if prepared else zero,n);expected=int(cc@((vv>>q)&1));nn=int(cc.sum())
                p=float(fisher_exact([[observed,shots-observed],[expected,nn-expected]]).pvalue)
                checks.append(dict(block=block,pattern=name,qubit=q,physical=plan["physical"][q],prepared=prepared,
                    observed_p1=observed/shots,calibrated_p1=expected/nn,p=p))
        data={}
        groups={}
        for index,key in enumerate(keys):
            r=byblock[(block,*key)]; mapping=tuple(r["measurement_mapping"])
            values,counts=count_arrays(r["counts"],n)
            data[index]=(values,counts,int(counts.sum()),r["counts"])
            groups.setdefault(mapping,[]).append(index)
            rawpoint[index]+=counts@indicators[values]/counts.sum()/c["hardware_blocks"]
        def evaluate(mats,bootstrap_index=None):
            for mapping,indices in groups.items():
                weights=event_weights(indicators,mats[list(mapping)])
                for index in indices:
                    values,counts,shots,_=data[index]
                    if bootstrap_index is not None:counts=rng.multinomial(shots,counts/shots)
                    estimate=counts@weights[values]/shots/c["hardware_blocks"]
                    if bootstrap_index is None:point[index]+=estimate
                    else:
                        boots[bootstrap_index,index]+=estimate
                        rawboots[bootstrap_index,index]+=counts@indicators[values]/shots/c["hardware_blocks"]
        evaluate(matrices)
        for b in range(c["bootstrap_replicates"]):
            sampled=assignment_matrices(resample_counts(zero,rng),resample_counts(one,rng),n)
            evaluate(sampled,b)
            if (b+1)%25==0:print("Readout bootstrap",block,b+1,flush=True)
    for row,p in zip(checks,holm_correction(r["p"] for r in checks)):row["holm_p"]=p
    write_csv(out/"readout_heldout_checks.csv",checks)
    save(out/"readout_calibrations.json",readout_records)
    mitigation=[]
    for index,(arm,method) in enumerate(keys):
        for event_id,event in enumerate(["feasible","good"]):
            low,high=np.quantile(boots[:,index,event_id],[.025,.975])
            delta=boots[:,index,event_id]-rawboots[:,index,event_id]
            dl,dh=np.quantile(delta,[.025,.975])
            ideal=methods[method]["ideal"]["p_"+event]
            mitigation.append(dict(arm=arm,method=method,event=event,raw_probability=float(rawpoint[index,event_id]),
                corrected_estimate=float(point[index,event_id]),ci_low=float(low),ci_high=float(high),
                correction=float(point[index,event_id]-rawpoint[index,event_id]),correction_ci_low=float(dl),correction_ci_high=float(dh),
                ideal_probability=ideal,raw_absolute_error=float(abs(rawpoint[index,event_id]-ideal)),
                corrected_absolute_error=float(abs(point[index,event_id]-ideal))))
    write_csv(out/"mitigated_events.csv",mitigation)
    np.savez_compressed(out/"bootstrap_estimates.npz",corrected=boots,raw=rawboots,keys=np.array(keys))
    secondary=[]
    comparisons=[]
    for method in methods:comparisons.append(("suppression",("suppressed",method),("raw",method)))
    for layers in c["hardware_layers"]:
        a=f"full_aqc_qaoa_l{layers}";b=f"full_product_qaoa_l{layers}"
        if a in methods and b in methods:
            for arm in plan["arms"]:comparisons.append(("matched_compression",(arm,a),(arm,b)))
    for kind,left,right in comparisons:
        ia,ib=keys.index(left),keys.index(right)
        for event_id,event in enumerate(["feasible","good"]):
            differences=boots[:,ia,event_id]-boots[:,ib,event_id]; low,high=np.quantile(differences,[.025,.975])
            secondary.append(dict(contrast=kind,left=list(left),right=list(right),event=event,
                corrected_difference=float(point[ia,event_id]-point[ib,event_id]),ci_low=float(low),ci_high=float(high)))
    save(out/"mitigated_contrasts.json",secondary)
    save(out/"analysis_summary.json",dict(completed_utc=now(),shots=total_shots,reported_qpu_seconds=qpu_seconds,
        methods=len(methods),independently_checked_count_sets=len(raw),primary_tests=len(contrasts),
        readout_heldout_tests=len(checks),readout_heldout_holm_rejections=sum(r["holm_p"]<.05 for r in checks),
        bootstrap_replicates=c["bootstrap_replicates"],
        caveats=["Signed readout estimates are not observed useful samples and are never clipped.",
                 "Readout correction assumes independent assignment errors; held-out marginal checks do not establish absence of correlated errors.",
                 "Bootstrap intervals reflect sampling and shared calibration uncertainty, not drift, model bias or independent-day variation.",
                 "Rare-event bootstrap coverage is limited; raw zero counts retain exact binomial upper bounds.",
                 "Gate twirling and DD are a bundled suppression arm; this does not identify their individual effects.",
                 "Complete-state compression uses fixed saved QAOA angles and is not a parameterized replacement cost unitary."]))
    print("Analysis complete",total_shots,"shots",qpu_seconds,"reported QPU seconds",flush=True)


def run():
    from .memory_guard import guarded_run
    c,out=settings();command=[sys.executable,"-m","revision.local_runner","--extra-site",EXTRA,
                             "revision.analyze_full_circuit_mitigation","worker"]
    result=guarded_run(command,out/"logs/analysis.log",address_gib=4,rss_gib=3,reserve_gib=12,seconds=3600)
    save(out/"analysis_process.json",result)
    if result["status"]!="completed":raise RuntimeError(result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["run","worker"])
    {"run":run,"worker":analyze}[p.parse_args().action]()
