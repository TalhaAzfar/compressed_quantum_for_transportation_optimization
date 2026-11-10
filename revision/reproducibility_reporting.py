"""Keep progress honest; export diagnostic and two-day figures when complete."""
import argparse
import csv
import fcntl
import time
from .reproducibility_study import ROOT,OUT,settings
from .contribution_hardware_core import read,save,digest,now


def diagnostic_report():
    folders=sorted((OUT/'diagnostics').glob('*_r*'))
    completed=[read(p/'complete.json') for p in folders if (p/'complete.json').exists()]
    rows=[r for case in completed for r in case['rows']]
    if len(rows)!=36:
        return dict(complete_cases=len(completed),planned_cases=6,complete=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from .reproducibility_diagnostics import summarize
    summarize()
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    colors={'TSP':'#D55E00','VRP':'#009E73'}
    for case in completed:
        for steps in [12,24]:
            selected=sorted((r for r in case['rows'] if r['steps']==steps),key=lambda r:r['duration'])
            for ax,event in zip(axes,['feasible','optimal']):
                ax.plot([r['duration'] for r in selected],[100*r['p_'+event] for r in selected],
                    color=colors[selected[0]['family']],ls='--' if steps==12 else '-',
                    marker='^' if selected[0]['family']=='TSP' else 's',ms=4,alpha=.75)
    for ax,event in zip(axes,['feasible','optimal']):
        ax.set(xlabel='Annealing duration T',ylabel=f'Ideal {event} probability (%)',xticks=[2,4,8],yscale='log')
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles=[Line2D([],[],color=colors['TSP'],marker='^',label='TSP: 25 qubits'),
             Line2D([],[],color=colors['VRP'],marker='s',label='VRP: 20 qubits'),
             Line2D([],[],color='black',ls='--',label='12 steps'),Line2D([],[],color='black',label='24 steps')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.91),ncol=4,frameon=False)
    fig.suptitle('Larger-instance source diagnosis: exact ideal simulation',y=.99)
    fig.text(.08,.01,'Three fixed costs per family; all duration/step settings retained. These are not hardware results.',fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.82])
    for ext in ['pdf','png','svg']:fig.savefig(OUT/'figures'/('larger_source_diagnosis.'+ext),bbox_inches='tight',dpi=200)
    plt.close(fig)
    by_family={}
    for family in ['TSP','VRP']:
        cases=[case for case in completed if case['rows'][0]['family']==family]
        by_family[family]=[]
        for case in cases:
            initial=next(r for r in case['rows'] if r['duration']==2 and r['steps']==12)
            final=next(r for r in case['rows'] if r['duration']==8 and r['steps']==24)
            by_family[family].append(dict(case_id=case['case_id'],initial=initial,extended=final,
                feasibility_ratio=final['p_feasible']/initial['p_feasible'],
                optimum_ratio=final['p_optimal']/initial['p_optimal']))
    result=dict(completed_utc=now(),complete=True,complete_cases=6,by_family=by_family,
        interpretation='Changing annealing duration changes the target state. This diagnoses source quality, not compression or hardware advantage. Twelve/twenty-four-step agreement is a two-resolution check, not an asymptotic convergence proof.')
    save(OUT/'diagnostics/summary.json',result)
    return result


def target_diversity():
    import numpy as np
    cases=settings()['cases'];rows=[]
    for family in ['flp','tsp']:
        subset=[s for s in cases if s['problem']==family]
        for i,a in enumerate(subset):
            x=np.load(OUT/'cases'/a['id']/'reference.npy',mmap_mode='r')
            for b in subset[i+1:]:
                y=np.load(OUT/'cases'/b['id']/'reference.npy',mmap_mode='r')
                rows.append(dict(family=family,case_a=a['id'],case_b=b['id'],exact_overlap=float(abs(np.vdot(x,y))**2)))
    save(OUT/'target_diversity.json',dict(checked_utc=now(),rows=rows,
        interpretation='Fresh costs on a fixed constraint graph do not imply broadly different state targets. Report the overlaps alongside instance-level replication.'))


def final_report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    analysis=read(OUT/'analysis.json');primary=analysis['primary_day_specific']
    assert len(primary)==72 and all((OUT/f'day{d}_complete.json').exists() for d in [0,1])
    for reference in ['source','refined_product','fixed_shallow']:
        fig,axes=plt.subplots(2,2,figsize=(9,7),squeeze=False)
        for day in [0,1]:
            for ei,event in enumerate(['feasible','optimal']):
                ax=axes[ei,day]
                for r in primary:
                    if (r['day'],r['event'],r['reference'])!=(day,event,reference):continue
                    offset=(int(r['case_id'][-1])-2)*.2
                    color='#0072B2' if r['family']=='FLP' else '#D55E00'
                    delta=r['improvement_pp']
                    ax.errorbar(r['n_qubits']+offset,delta,
                        yerr=[[delta-r['ci_low_pp']],[r['ci_high_pp']-delta]],
                        fmt='o' if r['family']=='FLP' else '^',color=color,capsize=3,
                        mfc='white' if r['identical_circuit_self_control'] else color)
                ax.axhline(0,color='grey',ls='--',lw=.8)
                ax.set(xticks=[16,20],xlim=(15,21),xlabel='Number of qubits',ylabel=event.capitalize()+' improvement (pp)',title=f'Confirmation day {day+1}')
                ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        fig.legend(handles=[Line2D([],[],color='#0072B2',marker='o',ls='',label='FLP'),
                            Line2D([],[],color='#D55E00',marker='^',ls='',label='TSP')],loc='upper right',ncol=2,frameon=False)
        label={'source':'full annealing','refined_product':'refined product','fixed_shallow':'fixed shallow AQC'}[reference]
        fig.suptitle('Same selected native circuits versus '+label,x=.08,ha='left',y=.99)
        fig.text(.08,.015,'ibm_rensselaer · Nighthawk · fresh costs · pointwise 95% intervals; 72 tests jointly corrected',fontsize=8)
        fig.tight_layout(rect=[0,.035,1,.93])
        for ext in ['pdf','png','svg']:fig.savefig(OUT/'figures'/('confirmation_two_days_vs_'+reference+'.'+ext),bbox_inches='tight',dpi=200)
        plt.close(fig)
    for name,rows in [('confirmation_day_comparisons',primary),('confirmation_pooled_secondary',analysis['secondary_pooled']),('replication',analysis['replication'])]:
        with (OUT/(name+'.csv')).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    target_diversity();diagnostic_report()
    summary=[]
    for ref in ['source','refined_product','fixed_shallow']:
        for event in ['feasible','optimal']:
            rs=[r for r in analysis['replication'] if r['reference']==ref and r['event']==event]
            summary.append(dict(reference=ref,event=event,replicated_gains=sum(r['replicated_gain'] for r in rs),
                                replicated_losses=sum(r['replicated_loss'] for r in rs),cases=len(rs)))
    protocol=read(OUT/'hardware_protocol.json')
    for field in ['source_sha256','input_sha256']:
        assert all(digest(ROOT/p)==h for p,h in protocol[field].items())
    jobs=read(OUT/'jobs.json');assert len(jobs)==12 and len({j['job_id'] for j in jobs})==12
    assert sum(j['total_shots'] for j in jobs)<=settings()['hardware_max_total_shots']
    save(OUT/'report_complete.json',dict(completed_utc=now(),summary=summary,total_shots=sum(j['total_shots'] for j in jobs),
        analysis_sha256=digest(OUT/'analysis.json'),frozen_sources_unchanged=True,
        interpretation='Replicated gains require corrected positive effects on both days. Visual inspection and manuscript integration remain subsequent editorial tasks.'))


def watch():
    lock=(OUT/'reporting.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.time()+9*86400
    while not (OUT/'analysis.json').exists():
        assert time.time()<deadline,'Reporting wait exceeded nine days'
        for name in ['preparation_stop.json','pipeline_stop.json']:
            if (OUT/name).exists():
                save(OUT/'reporting_stop.json',dict(stopped_utc=now(),upstream_stop=name));return
        if not (OUT/'diagnostics/summary.json').exists():diagnostic_report()
        if (OUT/'preparation_complete.json').exists() and not (OUT/'target_diversity.json').exists():target_diversity()
        time.sleep(60)
    final_report()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['watch','final_report','diagnostic_report']);a=p.parse_args()
    globals()[a.action]()
