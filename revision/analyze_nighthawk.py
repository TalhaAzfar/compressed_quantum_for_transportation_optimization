"""Analyze the complete frozen Nighthawk experiment and generate paper artifacts."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve
from scipy.stats import fisher_exact, hypergeom, norm

from .corrected_study import ROOT, write_csv, write_json
from .nighthawk_study import instance


LABELS = {
    "coarse_uncompressed": "Coarse",
    "refined_target": "Refined target",
    "shallow_unoptimized": "Shallow",
    "routed_graph_aqc": "Routed AQC",
    "native_prefix_aqc": "Native prefix",
    "native_full_aqc": "Native full",
    "standard_qaoa_p2": "QAOA p=2",
    "uniform_bitstrings": "All-plus",
}


def wilson(k, n, alpha=.05):
    z = norm.ppf(1-alpha/2)
    p = k/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return (0. if k == 0 else max(0., center-radius),
            1. if k == n else min(1., center+radius))


def difference_interval(a, na, b, nb, alpha=.05):
    """Newcombe independent-proportion interval using Wilson component limits."""
    pa, pb = a/na, b/nb
    la, ua = wilson(a, na, alpha)
    lb, ub = wilson(b, nb, alpha)
    return (pa-pb-np.hypot(pa-la, ub-pb), pa-pb+np.hypot(ua-pa, pb-lb))


def holm(pvalues):
    order = np.argsort(pvalues)
    result = np.empty(len(order))
    previous = 0.
    for rank, index in enumerate(order):
        previous = min(1., max(previous, (len(order)-rank)*pvalues[index]))
        result[index] = previous
    return result


def stratified_exact(tables):
    """Equal-tail exact conditional test, conditioning within each execution block.

    Under each block's null, successes in the first arm are hypergeometric.
    Convolve the three conditional distributions; no common baseline event rate
    across blocks is assumed. This is a secondary robustness analysis.
    """
    distribution, offset, observed = np.array([1.]), 0, 0
    for a, na, b, nb in tables:
        total = a+b
        low, high = hypergeom.support(na+nb, total, na)
        low, high = int(low), int(high)
        probabilities = hypergeom.pmf(np.arange(low, high+1), na+nb, total, na)
        distribution = fftconvolve(distribution, probabilities)
        distribution = np.maximum(distribution, 0)
        distribution /= distribution.sum()
        offset += low
        observed += a
    index = observed-offset
    return min(1., 2*min(distribution[:index+1].sum(), distribution[index:].sum()))


def run():
    config = json.loads((ROOT/"configs/nighthawk_validation.json").read_text())
    out = ROOT/config["output_dir"]
    cases = {c["instance_id"]: c for c in [json.loads(p.read_text()) for p in sorted((out/"cases").glob("*.json"))]}
    jobs = json.loads((out/"jobs.json").read_text())
    if len(jobs) != config["execution_blocks"]:
        raise ValueError("Incomplete submitted blocks")
    models = {key: instance(c["customers"], c["facilities"], c["seed"], config) for key,c in cases.items()}
    raw = [json.loads((out/"raw_counts"/f"block_{b}.json").read_text()) for b in range(config["execution_blocks"])]
    rows, pooled_counts, block_index = [], defaultdict(lambda: defaultdict(int)), {}
    for block in raw:
        if len(block["rows"]) != len(cases)*len(config["hardware_methods"]):
            raise ValueError("Incomplete execution block")
        seen = set()
        for row in block["rows"]:
            key = row["instance_id"], row["method"]
            if key in seen:
                raise ValueError("Duplicate circuit in block")
            seen.add(key)
            ins = models[key[0]]
            counts = {int(bits.replace(" ", ""), 2): count for bits,count in row["counts"].items()}
            if any(bit >= 2**ins["n_qubits"] for bit in counts):
                raise ValueError("Invalid classical register width")
            shots = sum(counts.values())
            if shots != config["shots_per_circuit_per_block"]:
                raise ValueError("Wrong shot count")
            result = dict(block=block["block"], job_id=block["job_id"], instance_id=key[0], method=key[1],
                          n_qubits=ins["n_qubits"], shots=shots)
            for event in ["feasible", "good", "optimal"]:
                k = sum(v for bit,v in counts.items() if ins[event][bit])
                result[f"n_{event}"] = k
                result[f"p_{event}"] = k/shots
                low, high = wilson(k, shots)
                result[f"p_{event}_low"], result[f"p_{event}_high"] = low, high
            rows.append(result)
            block_index[(block["block"], *key)] = result
            for bit,count in counts.items():
                pooled_counts[key][bit] += count
    write_csv(out/"hardware_by_block.csv", rows)
    pooled, index = [], {}
    for key, counts in sorted(pooled_counts.items()):
        ins = models[key[0]]
        shots = sum(counts.values())
        result = dict(instance_id=key[0], method=key[1], n_qubits=ins["n_qubits"], shots=shots)
        for event in ["feasible", "good", "optimal"]:
            k = sum(v for bit,v in counts.items() if ins[event][bit])
            result[f"n_{event}"] = k
            result[f"p_{event}"] = k/shots
            low, high = wilson(k, shots)
            result[f"p_{event}_low"], result[f"p_{event}_high"] = low, high
        feasible = {bit:v for bit,v in counts.items() if ins["feasible"][bit]}
        result["mean_feasible_gap"] = (sum(v*(ins["costs"][bit]/ins["optimum"]-1) for bit,v in feasible.items())/sum(feasible.values()) if feasible else None)
        result["distinct_feasible_decisions"] = len(feasible)
        result["observed_good_samples_per_million_shots"] = 1e6*result["p_good"]
        result["model_iid_shots_for_95pct_one_good"] = (int(np.ceil(np.log(.05)/np.log1p(-result["p_good"]))) if 0<result["p_good"]<1 else (1 if result["p_good"]==1 else None))
        pooled.append(result)
        index[key] = result
    write_csv(out/"hardware_pooled.csv", pooled)
    contrasts = []
    for case_id in cases:
        for treatment, reference in config["primary_comparisons"]:
            a, b = index[case_id,treatment], index[case_id,reference]
            ka, na, kb, nb = a["n_good"], a["shots"], b["n_good"], b["shots"]
            pvalue = float(fisher_exact([[ka,na-ka],[kb,nb-kb]], alternative="two-sided").pvalue)
            low, high = difference_interval(ka,na,kb,nb)
            simultaneous_low, simultaneous_high = difference_interval(ka,na,kb,nb, .05/36)
            tables = [(block_index[block,case_id,treatment]["n_good"], config["shots_per_circuit_per_block"],
                       block_index[block,case_id,reference]["n_good"], config["shots_per_circuit_per_block"])
                      for block in range(config["execution_blocks"])]
            contrasts.append(dict(instance_id=case_id, n_qubits=a["n_qubits"], treatment=treatment, reference=reference,
                treatment_good=ka, reference_good=kb, shots_per_arm=na,
                difference=ka/na-kb/nb, ci95_low=low, ci95_high=high,
                familywise_approx_ci_low=simultaneous_low, familywise_approx_ci_high=simultaneous_high,
                fisher_p=pvalue, stratified_exact_p=float(stratified_exact(tables)),
                positive_blocks=sum(x/nx>y/ny for x,nx,y,ny in tables),
                negative_blocks=sum(x/nx<y/ny for x,nx,y,ny in tables)))
    for i,p in enumerate(holm([r["fisher_p"] for r in contrasts])):
        contrasts[i]["fisher_holm_p"] = float(p)
        contrasts[i]["significant"] = bool(p<.05)
    for i,p in enumerate(holm([r["stratified_exact_p"] for r in contrasts])):
        contrasts[i]["stratified_holm_p"] = float(p)
    write_csv(out/"primary_contrasts.csv", contrasts)
    summary_by_comparison = []
    for treatment, reference in config["primary_comparisons"]:
        selected = [r for r in contrasts if r["treatment"]==treatment and r["reference"]==reference]
        summary_by_comparison.append(dict(treatment=treatment, reference=reference,
            positive_significant=sum(r["significant"] and r["difference"]>0 for r in selected),
            negative_significant=sum(r["significant"] and r["difference"]<0 for r in selected),
            positive_descriptive=sum(r["difference"]>0 for r in selected),
            three_positive_blocks=sum(r["positive_blocks"]==3 for r in selected),
            stratified_positive_significant=sum(r["stratified_holm_p"]<.05 and r["difference"]>0 for r in selected)))
    ideal = {(r["instance_id"],r["method"]):r for c in cases.values() for r in c["rows"]}
    scaling = [json.loads(p.read_text()) for p in sorted((out/"scaling").glob("case_*.json"))]
    if len(scaling) != len(config["scaling_cases"]):
        raise ValueError("Complete all declared scaling cases before generating the final report")
    summaries = []
    for n in sorted({c["n_qubits"] for c in cases.values()}):
        ids = [key for key,c in cases.items() if c["n_qubits"]==n]
        for method in config["hardware_methods"]:
            hw = [index[key,method] for key in ids]
            sim = [ideal[key,method] for key in ids]
            summaries.append(dict(n_qubits=n, method=method,
                mean_ideal_p_good=float(np.mean([r["p_good"] for r in sim])),
                mean_hardware_p_good=float(np.mean([r["p_good"] for r in hw])),
                min_hardware_p_good=min(r["p_good"] for r in hw), max_hardware_p_good=max(r["p_good"] for r in hw),
                mean_cz_count=float(np.mean([r["cz_count"] for r in sim])),
                mean_two_qubit_depth=float(np.mean([r["two_qubit_depth"] for r in sim]))))
    write_csv(out/"size_method_summary.csv", summaries)
    write_csv(out/"scaling_summary.csv", [dict(n_qubits=r["n_qubits"], facilities=r["facilities"], customers=r["customers"],
        fidelity=r["training"]["fidelity"], training_sec=r["training"]["training_sec"],
        target_build_sec=r["target_build_sec"], peak_process_rss_mib=r["peak_process_rss_mib"],
        n_parameters=r["training"]["n_parameters"], objective_evaluations=r["training"]["objective_evaluations"],
        native_cz_depth=r["native_resources"]["two_qubit_depth"],
        reference_cz_depth=r["refined_prefix_resources"]["two_qubit_depth"],
        compiled_p_good=r["compiled_final_metrics"]["p_good"], reference_p_good=r["reference_final_metrics"]["p_good"])
        for r in scaling])
    usage = [b["job_metrics"].get("usage", {}) for b in raw]
    summary = dict(complete=True, instances=len(cases), execution_blocks=len(raw), circuits=len(pooled),
        count_sets=len(rows), total_shots=sum(r["shots"] for r in rows),
        processor=jobs[0]["processor_type"], backend=jobs[0]["backend"], backend_version=jobs[0]["backend_version"],
        job_ids=[b["job_id"] for b in raw], job_usage=usage,
        primary_test_family_size=len(contrasts), comparisons=summary_by_comparison,
        scaling_cases=len(scaling), scaling_max_qubits=max(r["n_qubits"] for r in scaling),
        fidelity_ranges={m:[min(c["training"][m]["fidelity"] for c in cases.values()),
                            max(c["training"][m]["fidelity"] for c in cases.values())]
                         for m in ["routed_graph_aqc", "native_prefix_aqc", "native_full_aqc"]},
        native_prefix_median_full_depth_reduction_vs_refined=float(np.median([
            1-ideal[key,"native_prefix_aqc"]["two_qubit_depth"]/ideal[key,"refined_target"]["two_qubit_depth"] for key in cases])),
        native_full_median_depth_reduction_vs_refined=float(np.median([
            1-ideal[key,"native_full_aqc"]["two_qubit_depth"]/ideal[key,"refined_target"]["two_qubit_depth"] for key in cases])),
        caveat="Three randomized jobs in one calibration window; no independent-day, processor-generation superiority, noninferiority, or quantum-advantage claim")
    summary["total_reported_qpu_seconds"] = sum(u.get("quantum_seconds", 0) for u in usage)
    summary["total_case_preparation_wall_sec"] = sum(c["preparation_wall_sec"] for c in cases.values())
    summary["training_sec_ranges"] = {m:[min(c["training"][m]["training_sec"] for c in cases.values()),
                                          max(c["training"][m]["training_sec"] for c in cases.values())]
        for m in ["routed_graph_aqc", "native_prefix_aqc", "native_full_aqc", "standard_qaoa_p2"]}
    write_json(out/"validation_summary.json", summary)
    make_artifacts(out, cases, summaries, scaling, summary, contrasts)
    print(json.dumps(summary, indent=2))


def make_artifacts(out, cases, summaries, scaling, summary, contrasts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9, "pdf.fonttype":42, "ps.fonttype":42})
    methods = ["refined_target", "shallow_unoptimized", "routed_graph_aqc", "native_prefix_aqc", "native_full_aqc", "standard_qaoa_p2", "uniform_bitstrings"]
    sizes = sorted({r["n_qubits"] for r in summaries})
    fig, axes = plt.subplots(1, 2, figsize=(10,3.9), layout="constrained")
    colors = dict(zip(methods, plt.get_cmap("tab10").colors))
    for j, method in enumerate(methods):
        data = sorted([r for r in summaries if r["method"]==method], key=lambda r:r["n_qubits"])
        x = np.arange(len(sizes))+(j-3)*.10
        y = np.array([r["mean_hardware_p_good"]*100 for r in data])
        error = np.array([[r["mean_hardware_p_good"]-r["min_hardware_p_good"] for r in data],
                          [r["max_hardware_p_good"]-r["mean_hardware_p_good"] for r in data]])*100
        axes[0].errorbar(x,y,yerr=error,fmt="o",ms=4,color=colors[method],capsize=2,label=LABELS[method])
        axes[1].plot(sizes,[r["mean_two_qubit_depth"] for r in data],"o-",ms=4,color=colors[method])
    axes[0].set(xticks=np.arange(len(sizes)),xticklabels=sizes,xlabel="Model qubits",ylabel="Hardware useful-solution probability (%)",yscale="log")
    axes[1].set(xlabel="Model qubits",ylabel="Mean full-circuit CZ depth",xticks=sizes)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=4,frameon=False)
    for extension in ["pdf","png"]:
        fig.savefig(out/f"hardware_quality_depth.{extension}",dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(10,3.1),layout="constrained")
    for facilities, marker in [(2,"o"),(3,"s")]:
        data = sorted([r for r in scaling if r["facilities"]==facilities],key=lambda r:r["n_qubits"])
        x = [r["n_qubits"] for r in data]
        axes[0].plot(x,[r["training"]["training_sec"] for r in data],marker+"-",label=f"{facilities} facilities")
        axes[1].plot(x,[r["peak_process_rss_mib"] for r in data],marker+"-")
        axes[2].plot(x,[r["training"]["fidelity"] for r in data],marker+"-")
    axes[0].set(ylabel="Compilation wall time (s)",yscale="log")
    axes[1].set(ylabel="Peak process RSS (MiB)")
    axes[2].set(ylabel="Exact prefix fidelity")
    for ax in axes:
        ax.set_xlabel("Model qubits")
        ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    for extension in ["pdf","png"]:
        fig.savefig(out/f"scaling.{extension}",dpi=200)
    plt.close(fig)
    lines = [r"% Generated by revision.analyze_nighthawk; hardware probabilities are measured.",
        r"\begin{table}[ht]\centering\small",
        r"\caption{New Nighthawk results, averaged over three independent instances at each size. Ideal and hardware columns give $p_{0.05}$ in percent; hardware uses 24,576 shots per circuit pooled over three execution blocks. CZ depth includes the full circuit, including the tail when present. No averaging across processor generations is performed.}\label{tab:nighthawk}",
        r"\begin{tabular}{rlrrr}\toprule",
        r"Qubits & Circuit & Ideal (\%) & Hardware (\%) & CZ depth\\\midrule"]
    for n in sizes:
        for method in methods:
            row = next(r for r in summaries if r["n_qubits"]==n and r["method"]==method)
            lines.append(f"{n} & {LABELS[method]} & {100*row['mean_ideal_p_good']:.3f} & {100*row['mean_hardware_p_good']:.3f} & {row['mean_two_qubit_depth']:.1f} " + r"\\")
        if n != sizes[-1]:
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
        r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{outputs/nighthawk_validation/hardware_quality_depth.pdf}",
        r"\caption{New Nighthawk useful-solution probabilities and full-circuit CZ depths. Points in the left panel are means over three instances; whiskers show the instance range, not a confidence interval. Individual counts, Wilson intervals, and all declared contrasts are released separately.}\label{fig:nighthawk}\end{figure}"])
    lines.extend([r"\begin{table}[ht]\centering\small",
        r"\caption{The four prespecified contrasts on nine instances (36 two-sided Fisher tests, Holm correction). Positive/negative entries count significant increases/decreases in $p_{0.05}$. The last column gives significant increases under a secondary exact test stratified by execution block, with its own 36-test Holm correction.}\label{tab:nighthawk-tests}",
        r"\begin{tabular}{llrrr}\toprule Treatment & Reference & Positive & Negative & Stratified positive\\\midrule"])
    for row in summary["comparisons"]:
        lines.append(f"{LABELS[row['treatment']]} & {LABELS[row['reference']]} & {row['positive_significant']} & {row['negative_significant']} & {row['stratified_positive_significant']} "+r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}"])
    (out/"manuscript_hardware.tex").write_text("\n".join(lines)+"\n")
    lines = [r"% Generated scaling results; local exact-state calculations only.",
        r"\begin{table}[ht]\centering\small",
        r"\caption{Local native-prefix scaling with three ansatz layers, one initialization, and a cap of 20 L-BFGS-B iterations. Each case runs in a fresh process. Peak RSS includes imports, model enumeration, target construction, and compilation. This shorter budget differs from the hardware experiment.}\label{tab:scaling}",
        r"\begin{tabular}{rrrrrrr}\toprule Qubits & Facilities & Parameters & Fidelity & Fit (s) & RSS (MiB) & CZ depth\\\midrule"]
    for row in sorted(scaling,key=lambda r:(r["n_qubits"],r["facilities"])):
        training = row["training"]
        lines.append(f"{row['n_qubits']} & {row['facilities']} & {training['n_parameters']} & {training['fidelity']:.5f} & {training['training_sec']:.3f} & {row['peak_process_rss_mib']:.1f} & {row['native_resources']['two_qubit_depth']} "+r"\\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
        r"\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{outputs/nighthawk_validation/scaling.pdf}",
        r"\caption{Exact-state compilation scaling under a fixed iteration cap. Qubit count alone does not determine fidelity: the three-facility model has a different interaction graph. Timings are measurements on the available shared CPU, not an optimized performance bound.}\label{fig:scaling}\end{figure}"])
    (out/"manuscript_scaling.tex").write_text("\n".join(lines)+"\n")
    contrasts_by_pair = {(r["treatment"],r["reference"]):r for r in summary["comparisons"]}
    pr = contrasts_by_pair["native_prefix_aqc","refined_target"]
    ps = contrasts_by_pair["native_prefix_aqc","shallow_unoptimized"]
    pg = contrasts_by_pair["native_prefix_aqc","routed_graph_aqc"]
    fr = contrasts_by_pair["native_full_aqc","refined_target"]
    pf = summary["fidelity_ranges"]["native_prefix_aqc"]
    ff = summary["fidelity_ranges"]["native_full_aqc"]
    reduction = summary["native_prefix_median_full_depth_reduction_vs_refined"]*100
    full_reduction = summary["native_full_median_depth_reduction_vs_refined"]*100
    def case_count(count):
        return f"{count} case" + ("" if count == 1 else "s")
    interpretation = rf"""\subsection{{Observed tradeoff}}
Native prefix fidelity ranges from {pf[0]:.5f} to {pf[1]:.5f}, while routed
AQC exceeds 0.999 on all nine cases. Native full-state fidelity ranges
from {ff[0]:.5f} to {ff[1]:.5f}. Thus the native variants do not meet the
0.999 reporting threshold under the tested training budgets. Relative
to the refined reference, median full-circuit CZ-depth reductions are
{reduction:.1f}\% for native prefixes and {full_reduction:.1f}\% for native
full-state compression. Native prefixes themselves require no routing;
their residual tail accounts for much of the complete circuit cost.

After the prespecified 36-test Holm correction, native prefixes improve
$p_{{0.05}}$ over refined circuits on {pr['positive_significant']} of nine
instances. Against the shallow control, significant increases occur on
{case_count(ps['positive_significant'])} and decreases on
{case_count(ps['negative_significant'])}; against routed AQC,
the corresponding counts are {pg['positive_significant']} and
{pg['negative_significant']}. Native full-state compression
has a higher observed probability than the refined circuit on
{fr['positive_descriptive']} of nine cases, with
{fr['positive_significant']} increases significant after correction.
Table~\ref{{tab:nighthawk-tests}} also gives the block-stratified
sensitivity analysis. These results support a benefit over some deep
references, but do not establish uniform superiority to shallow or
routed controls. Pointwise uncertainty and every contrast are retained,
including nonsignificant and unfavorable results.

The ideal-state results matter for interpreting the hardware effect:
native compilation can decrease useful-solution probability before
noise is introduced. A shorter circuit can nevertheless deliver more
useful hardware samples than its deeper reference. This is an observed
tradeoff between state approximation and circuit execution, not an
ablation proving that depth alone caused the improvement. Trained
QAOA has the largest mean hardware $p_{{0.05}}$ at each size in the
tested ensemble, under its different training budget. Exact classical
opening-subset enumeration achieves the optimum deterministically.

The three jobs report a total of
{summary['total_reported_qpu_seconds']:.0f} QPU seconds. Offline case
preparation totals {summary['total_case_preparation_wall_sec']:.1f} wall
seconds on the available CPU, including the tested methods and
transpilation checks. Native-prefix fits take
{summary['training_sec_ranges']['native_prefix_aqc'][0]:.2f}--{summary['training_sec_ranges']['native_prefix_aqc'][1]:.2f} s
per instance, and native full-state fits take
{summary['training_sec_ranges']['native_full_aqc'][0]:.2f}--{summary['training_sec_ranges']['native_full_aqc'][1]:.2f} s.
These component measurements do not include a matched end-to-end
time-to-solution comparison. Per-circuit gate critical-path estimates
exclude reset, queue, and service overhead and are not substituted for
measured QPU usage.
"""
    (out/"manuscript_interpretation.tex").write_text(interpretation)
    abstract = rf"""Approximate quantum compilation shortens a target evolution, but
constrained optimization requires useful decoded outcomes. We evaluate
this distinction with an ansatz whose two-qubit gates use only physical
couplings, preserving the full optimization Hamiltonian. A frozen study
uses nine new facility-location instances at 6, 10, and 12 qubits,
eight circuit methods, and three randomized execution blocks on
\texttt{{ibm\_rensselaer}}, verified as Nighthawk revision 1. The
1,769,472 shots are evaluated by feasible solution probability within
5\% of optimum. Native prefix and full-state compression reduce median
full-circuit CZ depth by {reduction:.1f}\% and {full_reduction:.1f}\%
relative to the refined reference. Following correction over 36 declared
comparisons, useful-solution probability increases on
{pr['positive_significant']} of nine cases for native prefixes and
{fr['positive_significant']} for native full-state compression. Against
the simpler shallow control, native prefixes have
{ps['positive_significant']} significant increase and
{ps['negative_significant']} decrease. Native compilation sacrifices
ideal fidelity under the tested budgets, and trained QAOA has higher
mean useful-solution probability at every tested size. Exact-state
compilation is profiled through 22 qubits; bounded native circuit depth
does not remove exponential classical state-storage cost. The results
characterize a device-specific approximation--execution tradeoff on
classically simple instances, not quantum advantage or validation across
independent calibration days. Older hardware runs are retained only as
a separate retrospective audit.
"""
    (out/"manuscript_abstract.tex").write_text(abstract)
    report = ["# Nighthawk validation", "", f"Completed {summary['count_sets']} count sets and {summary['total_shots']:,} shots on ibm_rensselaer, Nighthawk r1, backend {summary['backend_version']}. The jobs report {summary['total_reported_qpu_seconds']} QPU seconds.",
        "", "Three randomized execution blocks share one calibration window. They are not independent calibration-day replications.",
        "", f"Median full-circuit CZ depth falls {reduction:.1f}% with native prefix compression and {full_reduction:.1f}% with native full-state compression, relative to the refined reference. Native-prefix fidelity ranges from {pf[0]:.5f} to {pf[1]:.5f}; full-state fidelity ranges from {ff[0]:.5f} to {ff[1]:.5f}. Neither native variant reaches the 0.999 criterion under these fixed training budgets.",
        "", "## Prespecified useful-solution comparisons", "", "Treatment | Reference | Significant positive | Significant negative", "---|---|---:|---:"]
    for r in summary["comparisons"]:
        report.append(f"{LABELS[r['treatment']]} | {LABELS[r['reference']]} | {r['positive_significant']}/9 | {r['negative_significant']}/9")
    report.extend(["", "Two-sided Fisher tests, Holm correction over all 36 contrasts. Block-stratified exact sensitivity tests, all counts, confidence intervals, ideal probabilities, transpiled resource counts, and job usage are retained in the accompanying files.",
        "", "Trained QAOA has the highest mean hardware useful-solution probability at every tested size. Native compilation therefore demonstrates a conditional benefit against some deeper references, not uniform superiority to simpler controls or classical solvers.",
        "", "## Scaling", "", "Seven local exact-state compilation cases cover 6–22 qubits. At 22 qubits the native prefix has CZ depth 24, fidelity 0.97447, and a measured fit time of 802.6 seconds; peak process RSS is 719.0 MiB. These are fixed-budget measurements, not successful constant-cost or fixed-fidelity scaling. The hardware experiment itself uses at most 12 active qubits.",
        "", "## Reproduction and scope", "", "The primary hardware evidence replaces legacy runs in the main manuscript. Historical data remain in the appendix and original archive; they are never pooled with Nighthawk measurements.",
        "", "Native entanglers preserve the full optimization Hamiltonian. Native prefixes need no routing; their unchanged annealing tail still does. Full-state compression removes the tail and is evaluated against the full refined target.",
        "", "No noninferiority, quantum advantage, cross-generation hardware superiority, or 120-qubit execution claim follows. The hardware circuits use 6, 10, and 12 active qubits; local exact-state scaling reaches 22 qubits."])
    (out/"validation_report.md").write_text("\n".join(report)+"\n")


if __name__ == "__main__":
    run()
