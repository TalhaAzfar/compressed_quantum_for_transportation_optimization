"""Audit the frozen pilot analysis with stable direct conditional convolution.

The frozen secondary test uses FFT convolution, whose absolute roundoff floor
can dominate very small tail probabilities. Preserve that analysis and publish
this numerical audit separately; hypotheses, tails and multiplicity are unchanged.
"""
import csv
import json
import numpy as np
from scipy.stats import fisher_exact, hypergeom
from .corrected_study import write_csv, write_json
from .nighthawk_study import now
from .tensor_prefix_sabre_pilot import settings


def conditional_two_sided(tables):
    distribution = np.array([1.], dtype=np.longdouble)
    offset = observed = 0
    for a, na, b, nb in tables:
        if min(a, b, na-a, nb-b) < 0:
            raise ValueError("Invalid event count")
        low, high = map(int, hypergeom.support(na+nb, a+b, na))
        probabilities = hypergeom.pmf(np.arange(low, high+1), na+nb, a+b, na)
        distribution = np.convolve(distribution, probabilities.astype(np.longdouble))
        distribution /= distribution.sum()
        offset += low
        observed += a
    index = observed-offset
    return float(min(1., 2*min(distribution[:index+1].sum(), distribution[index:].sum())))


def holm_adjust(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    previous = 0.
    for rank, index in enumerate(order):
        previous = max(previous, min(1., (len(values)-rank)*values[index]))
        result[index] = previous
    return result


def audit():
    _, out = settings()
    summary = json.loads((out/"summary.json").read_text())
    assert summary["final_evaluation_complete"]
    rows = list(csv.DictReader((out/"hardware_by_stage.csv").open()))
    indexed = {(r["stage"], r["case_id"], r["method"]): r for r in rows}
    results = []
    for original in summary["comparisons"]:
        row = dict(original)
        tables = []
        for stage in ["final_0", "final_1"]:
            a = indexed[stage, row["case_id"], row["treatment"]]
            b = indexed[stage, row["case_id"], row["reference"]]
            key = row["event"]+"_count"
            tables.append((int(a[key]), int(a["shots"]), int(b[key]), int(b["shots"])))
        a, na, b, nb = map(sum, zip(*tables))
        primary = float(fisher_exact([[a, na-a], [b, nb-b]]).pvalue)
        assert np.isclose(primary, row["p_value"], rtol=1e-12, atol=0.)
        row["direct_stratified_p_value"] = conditional_two_sided(tables)
        results.append(row)
    for row, adjusted in zip(results, holm_adjust([r["direct_stratified_p_value"] for r in results])):
        row["direct_stratified_holm_p_value"] = adjusted
    changed = [r for r in results if (r["stratified_holm_p_value"] < .05) != (r["direct_stratified_holm_p_value"] < .05)]
    write_csv(out/"comparisons_audited.csv", results)
    write_json(out/"statistical_audit.json", dict(generated_utc=now(), comparisons_checked=len(results),
        primary_fisher_values_reproduced=True, secondary_significance_changes=len(changed),
        method="Direct long-double convolution of hypergeometric PMFs; same equal-tail test and 32-test Holm family as frozen analysis.",
        reason="FFT absolute roundoff dominates one tiny secondary tail. Frozen analysis is retained; direct values are the numerical audit used in the interpretation.",
        comparisons=results))
    print(f"Statistical audit: {len(results)} comparisons; {len(changed)} significance decisions changed", flush=True)


if __name__ == "__main__":
    audit()
