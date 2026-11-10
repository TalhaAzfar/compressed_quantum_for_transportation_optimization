"""Independent confirmation analysis for the native depth refinement study.

Screening ranks choose circuits. Only separately acquired confirmation shots
estimate the reported treatment effects and enter the 162-test Holm family.
"""
import argparse
import csv
import math
from statistics import NormalDist
from .native_depth_refinement import ROOT, OUT, PARENT, settings
from .contribution_hardware_core import read, save, digest, now
from .hardware_width_figure import block_mover_interval, block_difference_interval

EVENTS = ('feasible', 'optimal')
REFERENCES = ('source', 'refined_product', 'old_aqc_l2')
COLORS = {'FLP':'#0072B2', 'VRP':'#009E73', 'TSP':'#D55E00'}
MARKERS = {'FLP':'o', 'VRP':'s', 'TSP':'^'}


def write_csv(path, rows):
    assert rows
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def wilson(k, n, alpha=.05):
    assert 0 <= k <= n and n > 0
    z = NormalDist().inv_cdf(1-alpha/2); p = k/n; den = 1+z*z/n
    center = (p+z*z/(2*n))/den
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half), min(1.,center+half)


def rate_interval(blocks, event):
    """Pointwise additive-variance Wilson interval for a fixed block mean."""
    rates = [r[event+'_count']/r['shots'] for r in blocks]
    intervals = [wilson(r[event+'_count'],r['shots']) for r in blocks]
    p = sum(rates)/len(rates)
    low = p-math.sqrt(sum((p0-ci[0])**2 for p0,ci in zip(rates,intervals)))/len(rates)
    high = p+math.sqrt(sum((ci[1]-p0)**2 for p0,ci in zip(rates,intervals)))/len(rates)
    return p,max(0.,low),min(1.,high)


def aggregate(phase):
    from .width_sweep_models import metrics
    assert phase in ('screen','confirm')
    plan = read(OUT/(phase+'_plan.json'))
    expected = {(r['case_id'],r['method']):r for r in plan['rows']}
    assert len(expected)==len(plan['rows'])
    jobs = [j for j in read(OUT/'jobs.json') if j['phase']==phase]
    assert len(jobs)==6 and len({(j['family'],j['block']) for j in jobs})==6
    indexed = {}; inputs = {}; checks = []
    for job in jobs:
        assert job['backend']=='ibm_rensselaer' and job['processor']['family']=='Nighthawk'
        assert job['plan_sha256']==digest(OUT/(phase+'_plan.json'))
        path = OUT/'raw_counts'/f"{phase}_{job['family']}_b{job['block']}.json"
        raw = read(path); inputs[str(path.relative_to(ROOT))]=digest(path)
        assert raw['job_id']==job['job_id'] and raw['block']==job['block']
        assert len(raw['rows'])==len(job['rows'])
        for row in raw['rows']:
            cid,method = row['case_id'],row['method']; key = cid,method,raw['block']
            assert key not in indexed
            assert expected[cid,method]['candidate_id']==row['candidate_id']
            model = read(PARENT/'cases'/cid/'model.json')
            counts = row['counts']
            assert all(len(b)==model['n_qubits'] and set(b)<=set('01') and isinstance(k,int) and k>0 for b,k in counts.items())
            values = metrics(model,counts)
            assert values==row['metrics'] and values['shots']==plan['shots_per_circuit']
            record = dict(phase=phase,case_id=cid,family=expected[cid,method]['family'],
                n_qubits=model['n_qubits'],method=method,candidate_id=row['candidate_id'],
                block=raw['block'],job_id=raw['job_id'],shots=values['shots'],
                feasible_count=values['feasible_count'],optimal_count=values['optimal_count'])
            checks.append(record); indexed[key]=record
    assert set(indexed)=={(cid,method,b) for cid,method in expected for b in range(plan['blocks'])}
    assert sum(r['shots'] for r in checks)==plan['total_shots']
    pooled=[]
    for row in plan['rows']:
        blocks = [indexed[row['case_id'],row['method'],b] for b in range(plan['blocks'])]
        result = {k:row[k] for k in ['case_id','family','n_qubits','method','candidate_id','exact_fidelity']}
        result.update(cz_depth=row['resources']['cz_depth'],cz_count=row['resources']['cz_count'],
                      shots=sum(r['shots'] for r in blocks))
        for event in EVENTS:
            p,lo,hi = rate_interval(blocks,event)
            result.update({event+'_count':sum(r[event+'_count'] for r in blocks),
                'p_'+event:p,'p_'+event+'_low':lo,'p_'+event+'_high':hi,
                'ideal_p_'+event:row['ideal_events']['p_'+event]})
        pooled.append(result)
    write_csv(OUT/(phase+'_by_block.csv'),checks)
    write_csv(OUT/(phase+'_pooled.csv'),pooled)
    return plan,indexed,pooled,inputs


def rank_native(rows, pooled):
    """Fixed screening rule. This function never receives confirmation data."""
    indexed = {(r['case_id'],r['method']):r for r in pooled}
    candidates = []
    for row in rows:
        if row['layers'] is None or row['layers']==0: continue
        result = indexed[row['case_id'],row['method']]
        lower,_ = wilson(result['feasible_count'],result['shots'])
        candidates.append(dict(row,screen_feasible_count=result['feasible_count'],
            screen_shots=result['shots'],screen_p_feasible=result['p_feasible'],
            screen_feasibility_wilson_lower=lower))
    assert candidates
    return sorted(candidates,key=lambda r:(-r['screen_feasibility_wilson_lower'],
        r['resources']['cz_depth'],r['resources']['cz_count'],r['candidate_id']))


def select():
    assert not (OUT/'confirmation_protocol.json').exists(), 'Selection already frozen'
    assert not any(j['phase']=='confirm' for j in read(OUT/'jobs.json'))
    protocol=read(OUT/'hardware_protocol.json')
    assert all(digest(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    assert digest(OUT/'screen_plan.json')==protocol['screen_plan_sha256']
    plan,_,pooled,inputs = aggregate('screen')
    rankings={};rows=[];selected=[]
    for spec in settings()['cases']:
        cid=spec['id'];case_rows=[r for r in plan['rows'] if r['case_id']==cid]
        ranked=rank_native(case_rows,pooled);rankings[cid]=ranked
        best=ranked[0]
        selected.append(dict(case_id=cid,candidate_id=best['candidate_id'],
            cz_depth=best['resources']['cz_depth'],screen_method=best['method'],
            screen_feasibility_wilson_lower=best['screen_feasibility_wilson_lower']))
        winner=next(r for r in case_rows if r['candidate_id']==best['candidate_id'])
        rows.append(dict(winner,method='selected_native',screen_method=winner['method']))
        for method in ('old_aqc_l2','refined_product','source'):
            ref=next(r for r in case_rows if r['method']==method)
            rows.append(dict(ref,screen_method=method))
    c=settings();shots=len(rows)*c['hardware_confirmation_blocks']*c['hardware_confirmation_shots_per_circuit']
    assert len(rows)==108 and shots+plan['total_shots']<=c['hardware_max_total_shots']
    save(OUT/'screen_selection.json',dict(selected_utc=now(),rule=c['selection_rule'],
        selected=selected,rankings=rankings,screen_raw_sha256=inputs,
        inference_scope='Exploratory selection only; no screening winner confidence claim.'))
    save(OUT/'confirm_plan.json',dict(created_utc=now(),phase='confirm',rows=rows,
        blocks=c['hardware_confirmation_blocks'],shots_per_circuit=c['hardware_confirmation_shots_per_circuit'],
        total_shots=shots,selection_sha256=digest(OUT/'screen_selection.json')))
    save(OUT/'confirmation_protocol.json',dict(frozen_utc=now(),
        selection_sha256=digest(OUT/'screen_selection.json'),plan_sha256=digest(OUT/'confirm_plan.json'),
        screen_raw_sha256=inputs,confirmation_outcomes_seen=False,
        tests=c['confirmation_tests'],source_sha256=protocol['source_sha256']))
    print('Frozen independent confirmation:',len(rows),'circuits;',shots,'shots',flush=True)


def contrast(case_rows, indexed, reference, event):
    from .audit_tensor_prefix_statistics import conditional_two_sided
    selected=next(r for r in case_rows if r['method']=='selected_native')
    ref=next(r for r in case_rows if r['method']==reference)
    cid=selected['case_id'];tables=[]
    for b in [0,1]:
        a=indexed[cid,'selected_native',b];r=indexed[cid,reference,b]
        tables.append((a[event+'_count'],a['shots'],r[event+'_count'],r['shots']))
    estimate,lo,hi=block_mover_interval(tables)
    exact_estimate,clo,chi=block_difference_interval(tables)
    assert abs(estimate-exact_estimate)<1e-12
    k,n,l,m=map(sum,zip(*tables))
    same=selected['isa_sha256']==ref['isa_sha256']
    return dict(case_id=cid,family=selected['family'],n_qubits=selected['n_qubits'],
        selected_candidate=selected['candidate_id'],reference=reference,event=event,
        selected_cz_depth=selected['resources']['cz_depth'],reference_cz_depth=ref['resources']['cz_depth'],
        selected_count=k,selected_shots=n,reference_count=l,reference_shots=m,
        selected_probability=k/n,reference_probability=l/m,
        improvement_pp=100*estimate,ci_low_pp=100*lo,ci_high_pp=100*hi,
        conservative_ci_low_pp=100*clo,conservative_ci_high_pp=100*chi,
        identical_circuit_self_control=same,
        p_value=1. if same else conditional_two_sided(tables))


def plots(effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    labels={'source':'full 12-step annealing','refined_product':'refined product fit','old_aqc_l2':'previous shallow AQC'}
    handles=[Line2D([],[],marker=MARKERS[f],color=COLORS[f],ls='',label=f,ms=7) for f in COLORS]
    for event in EVENTS:
        for ref in REFERENCES:
            rows=[r for r in effects if r['event']==event and r['reference']==ref]
            for conservative in [False,True]:
                fig,axes=plt.subplots(1,2,figsize=(10.3,4.4),gridspec_kw={'width_ratios':[1.25,1]})
                fig.subplots_adjust(left=.115,right=.985,bottom=.23,top=.80,wspace=.28)
                for ax,large in zip(axes,[False,True]):
                    for r in rows:
                        if large and r['n_qubits']<14:continue
                        offset=(int(r['case_id'].rsplit('r',1)[1])-2)*.15
                        if r['n_qubits']==20:offset+=-.27 if r['family']=='FLP' else .27
                        lo=r[('conservative_' if conservative else '')+'ci_low_pp']
                        hi=r[('conservative_' if conservative else '')+'ci_high_pp'];p=r['improvement_pp']
                        ax.errorbar(r['n_qubits']+offset,p,yerr=[[p-lo],[hi-p]],
                            fmt=MARKERS[r['family']],color=COLORS[r['family']],capsize=2,
                            ms=5,mfc='white' if r['identical_circuit_self_control'] else COLORS[r['family']],elinewidth=1)
                    ax.axhline(0,color='#666666',ls='--',lw=.8)
                    ax.grid(axis='y',color='#dddddd',lw=.5);ax.set_axisbelow(True)
                    ax.set_xlabel('Number of qubits');ax.set_title('14–25 qubits' if large else 'All 27 fixed instances',fontsize=10)
                    ax.set_xticks([14,16,20,25] if large else [6,9,10,12,14,16,20,25])
                    ax.set_xlim((13,26) if large else (5,26))
                axes[0].set_ylabel(('Feasible' if event=='feasible' else 'Optimum')+' probability improvement (pp)')
                fig.suptitle('Selected native AQC versus '+labels[ref],x=.08,ha='left',y=.98,fontsize=13,weight='bold')
                fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.98,.945),ncol=3,frameon=False)
                kind='conservative pointwise 95% intervals' if conservative else 'nominal pointwise 95% block-stratified score intervals'
                fig.text(.08,.06,'ibm_rensselaer · Nighthawk · independent confirmation · '+kind,fontsize=8)
                fig.text(.08,.015,'One point per instance; horizontal offsets separate replicates. Hollow symbols: identical-circuit self-control.',fontsize=8)
                stem=event+'_vs_'+ref+('_conservative' if conservative else '')
                for ext in ['pdf','png','svg']:fig.savefig(folder/(stem+'.'+ext),bbox_inches='tight',pad_inches=.06)
                plt.close(fig)


def classical_report():
    from .native_depth_hardware import parent_rows,candidate_row
    flat=[];best=[]
    for spec in settings()['cases']:
        cid=spec['id'];path=OUT/'cases'/cid/'complete.json'
        if not path.exists(): continue
        completed=read(path);case,old=parent_rows(cid)
        rows=old+[candidate_row(case,r) for r in completed['rows']]
        for row in rows:
            fit=next((r for r in completed['rows'] if r['id']==row['candidate_id']),None)
            flat.append(dict(case_id=cid,family=row['family'],n_qubits=row['n_qubits'],
                candidate_id=row['candidate_id'],cz_depth=row['resources']['cz_depth'],
                cz_count=row['resources']['cz_count'],exact_fidelity=row['exact_fidelity'],
                ideal_p_feasible=row['ideal_events']['p_feasible'],ideal_p_optimal=row['ideal_events']['p_optimal'],
                evaluations=fit['evaluations'] if fit else None,converged=fit['converged'] if fit else None,
                fitting_seconds=fit['fitting_seconds'] if fit else None,
                adequate_evaluation_budget=fit['adequate_evaluation_budget'] if fit else None))
        for cap in [0,next(r for r in old if r['candidate_id']=='parent_aqc_l2')['resources']['cz_depth'],32,64]:
            pool=[r for r in rows if r['layers'] is not None and r['resources']['cz_depth']<=cap]
            for criterion in ['fidelity','feasibility']:
                chosen=max(pool,key=lambda r:(r['exact_fidelity'] if criterion=='fidelity' else r['ideal_events']['p_feasible'],
                    -r['resources']['cz_depth'],r['candidate_id']))
                source=next(r for r in old if r['candidate_id']=='parent_source')
                best.append(dict(case_id=cid,family=chosen['family'],n_qubits=chosen['n_qubits'],
                    criterion=criterion,cz_depth_cap=cap,candidate_id=chosen['candidate_id'],
                    cz_depth=chosen['resources']['cz_depth'],exact_fidelity=chosen['exact_fidelity'],
                    ideal_p_feasible=chosen['ideal_events']['p_feasible'],ideal_p_optimal=chosen['ideal_events']['p_optimal'],
                    source_p_feasible=source['ideal_events']['p_feasible'],source_p_optimal=source['ideal_events']['p_optimal']))
    if flat:write_csv(OUT/'classical_all_candidates.csv',flat);write_csv(OUT/'classical_best_by_budget.csv',best)
    print('Classical cases complete:',len({r['case_id'] for r in flat}),flush=True)
    return flat,best


def classical_plots(flat,best):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    settings_list=sorted({(r['family'],r['n_qubits']) for r in best})
    assert len(settings_list)==9
    for metric,label in [('exact_fidelity','Exact source fidelity'),
                         ('ideal_p_feasible','Ideal feasible probability (%)'),
                         ('ideal_p_optimal','Ideal optimum probability (%)')]:
        fig,axes=plt.subplots(3,3,figsize=(10.0,8.0));fig.subplots_adjust(left=.08,right=.98,bottom=.13,top=.92,hspace=.44,wspace=.31)
        for ax,(family,width) in zip(axes.flat,settings_list):
            cids=sorted({r['case_id'] for r in best if r['family']==family and r['n_qubits']==width})
            for cid,style in zip(cids,['-','--',':']):
                selected=sorted([r for r in best if r['case_id']==cid and r['criterion']=='fidelity'],key=lambda r:r['cz_depth_cap'])
                scale=1 if metric=='exact_fidelity' else 100
                ax.plot([r['cz_depth'] for r in selected],[scale*r[metric] for r in selected],
                    marker=MARKERS[family],ls=style,color=COLORS[family],ms=4,lw=1.1)
                old=[r for r in flat if r['case_id']==cid and r['candidate_id'] in ['parent_aqc_l2','parent_aqc_l6']]
                ax.scatter([r['cz_depth'] for r in old],[scale*r[metric] for r in old],marker='x',color='#666666',s=25)
                if metric!='exact_fidelity':
                    source=next(r for r in flat if r['case_id']==cid and r['candidate_id']=='parent_source')
                    ax.axhline(100*source[metric],color='#999999',ls=style,lw=.6)
            ax.set_title(f'{family}: {width} qubits',fontsize=10)
            ax.set_xticks([0,16,32,48,64]);ax.grid(alpha=.18);ax.set_xlim(-2,66)
            if metric=='exact_fidelity':ax.set_ylim(0,1.035)
            ax.set_xlabel('Executed CZ depth',fontsize=9)
        fig.supylabel(label,x=.015,fontsize=11)
        fig.suptitle('Best exact-fidelity fit within each native-depth budget',fontsize=13,weight='bold')
        note='Lines: three cost instances per setting. Gray crosses: earlier fits.'
        if metric!='exact_fidelity':note+=' Gray horizontal lines: full-anneal event probabilities.'
        fig.text(.08,.045,note,fontsize=8)
        fig.text(.08,.02,'Product preparation is the zero-depth point; shallower candidates remain eligible at every larger budget.',fontsize=8)
        for ext in ['pdf','png','svg']:fig.savefig(folder/('classical_'+metric+'.'+ext),bbox_inches='tight',pad_inches=.06)
        plt.close(fig)


def classical_references(pooled):
    """Exact descriptive baselines, without adding post hoc hypothesis tests."""
    from .width_sweep_models import feasible_costs
    rows=[]
    for spec in settings()['cases']:
        cid=spec['id'];model=read(PARENT/'cases'/cid/'model.json');costs=feasible_costs(model)
        feasible=len(costs);optimal=sum(abs(v-model['optimum'])<=1e-9 for v in costs.values())
        n=model['n_qubits'];assert feasible>0 and optimal>0
        for measured in [r for r in pooled if r['case_id']==cid]:
            row=dict(case_id=cid,family=measured['family'],n_qubits=n,method=measured['method'],
                feasible_strings=feasible,optimal_strings=optimal,
                uniform_bitstring_p_feasible=feasible/(1<<n),uniform_bitstring_p_optimal=optimal/(1<<n),
                enumerated_uniform_feasible_p_optimal=optimal/feasible)
            for event in EVENTS:
                baseline=row['uniform_bitstring_p_'+event]
                row.update({event+'_enrichment_over_uniform':measured['p_'+event]/baseline,
                    event+'_enrichment_low':measured['p_'+event+'_low']/baseline,
                    event+'_enrichment_high':measured['p_'+event+'_high']/baseline})
            rows.append(row)
    write_csv(OUT/'classical_reference_comparisons.csv',rows)
    return rows


def analyze():
    from .audit_tensor_prefix_statistics import holm_adjust
    protocol=read(OUT/'confirmation_protocol.json')
    assert all(digest(ROOT/p)==h for p,h in protocol['source_sha256'].items())
    assert digest(OUT/'confirm_plan.json')==protocol['plan_sha256']
    assert digest(OUT/'screen_selection.json')==protocol['selection_sha256']
    plan,indexed,pooled,inputs=aggregate('confirm');effects=[]
    for spec in settings()['cases']:
        rows=[r for r in plan['rows'] if r['case_id']==spec['id']]
        for ref in REFERENCES:
            for event in EVENTS:effects.append(contrast(rows,indexed,ref,event))
    assert len(effects)==162
    for row,p in zip(effects,holm_adjust([r['p_value'] for r in effects])):
        row['holm_p_value']=p
        row['corrected_direction']=('gain' if row['improvement_pp']>0 else 'loss') if p<.05 else 'unresolved'
    write_csv(OUT/'confirmation_comparisons.csv',effects)
    plots(effects);classical_references(pooled);classical=classical_report()
    if classical is not None:classical_plots(*classical)
    summary=[]
    for ref in REFERENCES:
        for event in EVENTS:
            rows=[r for r in effects if r['reference']==ref and r['event']==event]
            summary.append(dict(reference=ref,event=event,instances=len(rows),
                positive_point_estimates=sum(r['improvement_pp']>0 for r in rows),
                corrected_gains=sum(r['corrected_direction']=='gain' for r in rows),
                corrected_losses=sum(r['corrected_direction']=='loss' for r in rows),
                identical_circuit_self_controls=sum(r['identical_circuit_self_control'] for r in rows)))
    jobs=read(OUT/'jobs.json')
    catalog=[]
    for spec in settings()['cases']:
        cid=spec['id'];selected=next(r for r in plan['rows'] if r['case_id']==cid and r['method']=='selected_native')
        values=[r for r in pooled if r['case_id']==cid]
        native=next(r for r in values if r['method']=='selected_native')
        observed=max(values,key=lambda r:(r['p_feasible'],-r['cz_depth'],r['method']))
        catalog.append(dict(case_id=cid,model_spec=spec,selected_native=selected,
            confirmation=native,all_confirmation_methods=values,
            highest_observed_confirmation_feasibility_method=observed['method'],
            interpretation='Native circuit chosen before confirmation; observed confirmation ranking is descriptive. See all corrected comparisons for unresolved differences.'))
    save(OUT/'best_circuit_catalog.json',dict(created_utc=now(),cases=catalog,
        selection='Native feasibility winner of independent screening, among the declared searched pool; no global certificate.'))
    qpu_seconds=0.
    for job in jobs:
        raw=read(OUT/'raw_counts'/f"{job['phase']}_{job['family']}_b{job['block']}.json")
        qpu_seconds+=float(raw.get('job_metrics',{}).get('usage',{}).get('quantum_seconds',0))
    save(OUT/'analysis.json',dict(completed_utc=now(),all_confirmation_counts_independently_decoded=True,
        summary=summary,comparisons=effects,confirmation_raw_sha256=inputs,
        total_shots=sum(j['total_shots'] for j in jobs),hardware_jobs=len(jobs),reported_qpu_seconds=qpu_seconds,
        inference='Confirmation-only block-conditional exact two-sided tests; Holm across all 162 contrasts. Fixed instances and fixed block weights; intervals pointwise, not simultaneous.',
        limitations='Selected among the declared candidate pool using feasibility. No claim of a globally best ansatz, independent-day reproducibility, population scaling law, or end-to-end tensor-network scalability. Optimum was not the selection objective.'))
    print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['select','analyze','classical']);a=p.parse_args()
    {'select':select,'analyze':analyze,'classical':classical_report}[a.action]()
