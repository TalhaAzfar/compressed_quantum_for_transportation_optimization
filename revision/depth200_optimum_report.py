"""Generate auditable tables and a study report, including incomplete outcomes."""
import csv
import json
import math
from .depth200_optimum_screen import OUT,case_data,save
from .flp42_fidelity_parallel import now


def build():
    sources=[]
    for cid in ['tsp25','flp30']:
        if cid=='tsp25':events=json.loads((OUT/'screen/tsp25_standard_p2_exact.json').read_text())['events']
        else:events=next(r['events'] for r in json.loads((OUT/'exact_flp_source_events.json').read_text())['rows'] if r['case_id']==cid and r['method']=='standard_p2')
        case=case_data(cid);p=events['p_optimal']
        sources.append(dict(case_id=cid,qubits=case['model']['n_qubits'],optimum=case['model']['optimum'],
            cz_depth=case['methods']['standard_p2']['resources']['cz_depth'],
            cz_count=case['methods']['standard_p2']['resources']['cz_count'],
            ideal_feasible_percent=100*events['p_feasible'],ideal_optimal_percent=100*p,
            expected_optima_32768=32768*p,ideal_shots_95pct_hit=math.ceil(math.log(.05)/math.log1p(-p)),
            optimal_given_feasible_percent=100*p/events['p_feasible'],
            uniform_feasible_optimal_percent=100*events['optimal_strings']/events['feasible_strings']))
    fits=[json.loads(p.read_text()) for p in (OUT/'fits').glob('*.json')]
    rows=[]
    for r in sorted(fits,key=lambda r:(r['case_id'],r['layers'],r['start'])):
        rows.append(dict(case_id=r['case_id'],layers=r['layers'],start=r['start'],verified=r['verified'],
            fidelity=r['fidelity'],cz_depth=r['resources']['cz_depth'],cz_count=r['resources']['cz_count'],
            ideal_feasible_percent=100*r['events']['p_feasible'],ideal_optimal_percent=100*r['events']['p_optimal'],
            optimizer_objective_consistent=r['optimizer_objective_consistent'],evaluations=r['evaluations'],
            fitting_seconds=r['fitting_seconds']))
    for name,table in [('ideal_source_audit',sources),('fitting_summary',rows)]:
        if not table:continue
        with (OUT/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    text=['# Larger-problem optimum and feasibility study','',
          'This study follows the frozen placement revision. It asks whether full-output AQC-Tensor compression can recover feasible and optimal hardware samples from sources that produced no optima in earlier device runs. The two instances are reused development cases; fresh acquisitions are not independent-instance replication.','',
          '## Exact ideal-source audit','',
          '| Problem | Qubits | Source CZ depth | Feasible | Optimal | Ideal shots for 95% chance of an optimum |',
          '|---|---:|---:|---:|---:|---:|']
    for r in sources:
        description='TSP: 6 cities, fixed depot' if r['case_id']=='tsp25' else 'FLP: 5 facilities, 5 customers'
        text.append(f"| {description} | {r['qubits']} | {r['cz_depth']} | {r['ideal_feasible_percent']:.6g}% | {r['ideal_optimal_percent']:.6g}% | {r['ideal_shots_95pct_hit']:,} |")
    text+=['','The TSP reference is checked by exact 25-qubit statevector simulation. The FLP probabilities are independently computed by exact facility-history contraction for standard p=2 QAOA; this specialized calculation does not certify arbitrary compiled circuits. Neither fitting objective nor initialization uses the known optimum bitstring.','',
           'Conditional on a feasible sample, the ideal source optimum probabilities are 1.678% for TSP and 0.002071% for FLP, compared with 1.667% and 0.009091% under their enumerated uniform-feasible references. The source-only diagnostic therefore finds little or negative optimum enrichment before compression. This is separate from the hardware comparison.','',
           '## Complete-circuit depth ladder','',
           'Product control at CZ depth 0; native entangling candidates at depths 16, 48, 96 and 192. All use the saved logical model and physical patch, with no SWAPs inside the native ansatz. Source references retain their original complete circuit. Higher depth can improve ideal approximation while increasing execution error; selection does not assume the deepest candidate is best.','',
           '## Validation and status','',f"All {len(fits)} declared fits completed; {sum(r['verified'] for r in fits)} pass the executed independent checks. The table retains both optimizer-loss consistency and independent output validation. Reference cancellations and protocol amendments remain in the archive. Hardware conclusions use the actual measured outcomes.",'']
    hardware=OUT/'hardware_analysis.json'
    if hardware.exists():
        d=json.loads(hardware.read_text())
        optimal=sum(r['optimal_count'] for r in d['rows'] if r['method'].startswith('compressed_'))
        primary_gains=[r for r in d['primary_tests'] if r['event']=='feasible' and r['difference']>0 and r['p_holm']<.05]
        optimum_gains=[r for r in d['primary_tests']+d['secondary_tests'] if r['event']=='optimal' and r['difference']>0 and r['p_holm']<.05]
        optimum_statement=('At least one declared optimum comparison shows a corrected positive gain.' if optimum_gains else
                           'No optimum-probability gain survives the declared corrections.')
        text+=['## Fresh Nighthawk results','',
            f"Completed {d['total_shots']:,} shots in four jobs on ibm_rensselaer, Nighthawk revision 1, using {d['reported_qpu_seconds']:g} reported QPU seconds. Independent scalar decoding agrees for all {d['independent_count_sets']} experimental count sets.",
            '',f"The four compressed depths together produce {optimal} optimum observations across both instances and execution arms. These are observations across distinct circuits, not a pooled treatment comparison. {len(primary_gains)} of four primary feasibility gains survive the eight-test correction. {optimum_statement}",
            '',
            '| Problem | Arm | Circuit | CZ depth | Optimal / shots | Feasible | Best feasible gap |',
            '|---|---|---|---:|---:|---:|---:|']
        order=['original','product','compressed_l2','compressed_l6','compressed_l12','compressed_l24']
        for r in sorted(d['rows'],key=lambda r:(['tsp25','flp30'].index(r['case_id']),r['arm'],order.index(r['method']))):
            gap='none' if r['best_feasible_gap'] is None else f"{100*r['best_feasible_gap']:.3g}%"
            text.append(f"| {r['case_id']} | {r['arm']} | {r['method']} | {r['cz_depth']} | {r['optimal_count']} / {r['shots']} | {100*r['p_feasible']:.4g}% | {gap} |")
        text+=['','Counts are actual measured bitstrings, without repair or postselection. The suppressed arm combines XX dynamical decoupling and gate twirling. Pointwise exact confidence intervals, the eight-test primary family and the separate 64-test secondary family are in the CSV/JSON analysis. A single optimum hit is distinguished from evidence of an increased probability.','',
            '## Best observed native depth and its controls','',
            'Selection here uses the measured feasible percentage among the four native circuits, so it is descriptive and exploratory. It is not the preselected primary method.','',
            '| Case | Arm | Depth | Native feasible | Original feasible | Product feasible | Native/source adjusted p | Native/product adjusted p |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
        for cid in ['tsp25','flp30']:
            for arm in ['raw','suppressed']:
                candidates=[r for r in d['rows'] if r['case_id']==cid and r['arm']==arm and r['method'].startswith('compressed_')]
                best=max(candidates,key=lambda r:(r['p_feasible'],-r['cz_depth']))
                controls={m:next(r for r in d['rows'] if (r['case_id'],r['arm'],r['method'])==(cid,arm,m)) for m in ['original','product']}
                tests={m:next(r for r in d['secondary_tests'] if (r['case_id'],r['arm'],r['method'],r['control'],r['event'])==(cid,arm,best['method'],m,'feasible')) for m in controls}
                text.append(f"| {cid} | {arm} | {best['cz_depth']} | {100*best['p_feasible']:.4g}% | {100*controls['original']['p_feasible']:.4g}% | {100*controls['product']['p_feasible']:.4g}% | {tests['original']['p_holm']:.4g} | {tests['product']['p_holm']:.4g} |")
        text+=['','The ideal compiled TSP event rates predict about 0.45 optimum hits per circuit/arm at this budget. The FLP expectations are below 0.00016. The acquisition can record hits but is not a powered confirmation of reliable optimum recovery. Absence of significance is not equivalence.','',
            'All 18 fits and all four hardware depths are retained. Depth 192 uses 1,344 CZ gates for TSP and 1,824 for FLP; the latter exceeds its source count of 1,050. The primary search has best exact fidelity 0.5778 for TSP and reference-state fidelity 0.5513 for FLP. Successful high-fidelity scaling is not established by this bounded search.','']
        if (OUT/'observed_optimum_witness.json').exists():
            text+=['## The observed optimum','',
                   'The zero-CZ product control is itself a fitted approximation of the source, using only independent single-qubit rotations. Its one observed TSP optimum is the tour **0–1–3–4–5–2–0**, with cost **18.399165242883228**, in the suppressed arm (1/24,576 pooled shots). The raw bitstring and independent reconstruction are in `observed_optimum_witness.json`. This is an observed success of the simplest fitted approximation, not evidence that an entangling ansatz improves optimum probability.','']
    audit=OUT/'additional_exact30/summary.json'
    if audit.exists():
        a=json.loads(audit.read_text())
        text+=['## Additional exact 30-qubit validation','',
               f"The separately declared event and fidelity checks use the already frozen depth-16 and depth-192 FLP circuits. Completed checks: {len(a['event_methods_completed'])+len(a['fidelity_methods_completed'])}/4. Maximum relative event error against the frozen MPS estimates is {a['max_relative_event_error']:.6g}; maximum absolute fidelity difference is {a['max_fidelity_absolute_difference']:.6g}. The echo calculation evaluates the all-zero amplitude of the inverse source after the compiled circuit. These are ideal classical checks, not hardware fidelity measurements.",
               '',f"The additional validation uses {a['worker_seconds']:.1f} worker wall seconds in total. Exact values and all process records are under `additional_exact30/`; primary fitting, selection and acquisition remain unchanged.",'']
        if len(a['exact_fidelities'])==2:
            text += [f"Exact FLP fidelity is **{a['exact_fidelities']['compressed_l2']:.7f}** at depth 16 and **{a['exact_fidelities']['compressed_l24']:.7f}** at depth 192.",'']
    (OUT/'REPORT.md').write_text('\n'.join(text))
    save(OUT/'report_status.json',dict(updated_utc=now(),fit_records=len(fits),hardware_complete=hardware.exists()))
    print('Updated',OUT/'REPORT.md',flush=True)


if __name__=='__main__':build()
