"""Retrospective, fixed-depth hardware figures; never submits or fits circuits."""
import csv
import itertools
import json
import math
from statistics import NormalDist
from functools import lru_cache
from pathlib import Path
from .contribution_hardware_core import ROOT, digest, save, now

OUT = ROOT/'outputs/hardware_width_figures_2026'
STUDIES = ['full_circuit_mitigation', 'full_circuit_transfer', 'depth200_optimum_study']
COLORS = {'FLP':'#0072B2', 'VRP':'#009E73', 'TSP':'#D55E00'}
MARKERS = {'FLP':'o', 'VRP':'s', 'TSP':'^'}


@lru_cache(None)
def cp(k, n, alpha):
    from scipy.stats import beta
    assert 0 <= k <= n and n > 0 and 0 < alpha < 1
    return (0. if k == 0 else float(beta.ppf(alpha/2, k, n-k+1)),
            1. if k == n else float(beta.ppf(1-alpha/2, k+1, n-k)))


def block_difference_interval(tables, alpha=.05):
    """Conservative interval for an equal-block mean probability difference.

    Apply CP with error alpha/(2*B) to each arm/block parameter; a union
    bound gives simultaneous coverage >=1-alpha within this one contrast.
    Average the lower treatment minus upper control endpoints, and vice versa.
    Blocks may have different probabilities. This is not a population interval
    or an adjustment across plotted comparisons or retrospective method choice.
    """
    assert tables
    b=len(tables); estimate=low=high=0.
    for k,n,l,m in tables:
        a0,a1=cp(k,n,alpha/(2*b)); b0,b1=cp(l,m,alpha/(2*b))
        estimate+=(k/n-l/m)/b;low+=(a0-b1)/b;high+=(a1-b0)/b
    assert -1 <= low <= estimate <= high <= 1
    return estimate, low, high


def block_mover_interval(tables, alpha=.05, weights=None):
    """Tang (2021), section 2.2, additive-variance MOVER with Wilson scores.

    Fixed block weights allow different rates and effects across blocks.
    Nominal approximate pointwise coverage, not an exact coverage guarantee.
    The conservative CP construction above is retained as a sensitivity check.
    """
    z=NormalDist().inv_cdf(1-alpha/2)
    weights=[1/len(tables)]*len(tables) if weights is None else weights
    assert len(weights)==len(tables) and abs(sum(weights)-1)<1e-12
    def wilson(k,n):
        assert 0<=k<=n and n>0
        p=k/n;den=1+z*z/n
        center=(p+z*z/(2*n))/den
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        return max(0.,center-half),min(1.,center+half)
    estimate=vl=vu=0.
    for (k,n,l,m),w in zip(tables,weights):
        a0,a1=wilson(k,n);b0,b1=wilson(l,m);pa=k/n;pb=l/m
        estimate+=w*(pa-pb)
        vl+=w*w*((pa-a0)**2+(b1-pb)**2)
        vu+=w*w*((a1-pa)**2+(pb-b0)**2)
    return estimate,max(-1.,estimate-math.sqrt(vl)),min(1.,estimate+math.sqrt(vu))


def feasible_costs(m):
    """Construct actual feasible decisions, independently of experiment decoders."""
    costs={}
    if m['problem']=='flp':
        f=m['facilities'];c=m['customers']
        for mask in range(1,1<<f):
            opened=[j for j in range(f) if mask>>j&1]
            for assignment in itertools.product(opened,repeat=c):
                value=mask+sum(1<<(f+i*f+j) for i,j in enumerate(assignment))
                costs[value]=sum(m['opening'][j] for j in opened)+sum(m['service'][i][j] for i,j in enumerate(assignment))
    else:
        n=m['nodes'];distance=m['distance']
        def route_cost(route):
            return sum(distance[a][b] for a,b in zip([0]+list(route),list(route)+[0]))
        if m['problem']=='tsp':
            for route in itertools.permutations(range(1,n)):
                value=sum(1<<(t*(n-1)+city-1) for t,city in enumerate(route))
                costs[value]=route_cost(route)
        else:
            assert m['problem']=='routing' and n==4 and m['vehicles']==2
            arcs=[(i,j) for i in range(n) for j in range(n) if i!=j]
            for perm in itertools.permutations(range(1,n)):
                for cut in [1,2]:
                    routes=[perm[:cut],perm[cut:]]
                    chosen={arc for route in routes for arc in zip([0]+list(route),list(route)+[0])}
                    value=sum(1<<arcs.index(arc) for arc in chosen)
                    cost=sum(route_cost(route) for route in routes)
                    assert value not in costs or abs(costs[value]-cost)<1e-9
                    costs[value]=cost
            assert len(costs)==6
    assert abs(min(costs.values())-m['optimum'])<1e-8
    return costs


def write_csv(path, rows):
    with path.open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def prepare():
    assert not (OUT/'complete.json').exists(), 'Figure release already frozen'
    OUT.mkdir(exist_ok=True);inputs={}
    def read(path):
        inputs[str(path.relative_to(ROOT))]=digest(path)
        return json.loads(path.read_text())
    def csvread(path):
        inputs[str(path.relative_to(ROOT))]=digest(path)
        return list(csv.DictReader(path.open()))
    plans={s:read(ROOT/'outputs'/s/'hardware_plan.json') for s in STUDIES}
    cases=[];models={}
    def add(study,cid,model,compiled,source,product,original_product=None):
        rows=plans[study]['rows']
        def resource(method):
            return next(r['resources'] for r in rows if r['method']==method and r.get('case',r.get('case_id',cid))==cid)
        comp=resource(compiled);src=resource(source)
        assert comp['cz_depth']==16
        models[cid]=model
        cases.append(dict(study=study,case_id=cid,problem='VRP' if model['problem']=='routing' else model['problem'].upper(),
            n_qubits=model['n_qubits'],compiled=compiled,source=source,product_reference=product,
            original_product=original_product,compiled_cz_depth=16,source_cz_depth=src['cz_depth'],
            compiled_cz_count=comp['cz_count'],source_cz_count=src['cz_count'],
            source_family='standard p=2 QAOA' if study=='depth200_optimum_study' else 'AQC prefix + one QAOA layer',
            product_reference_kind='zero-CZ fit to same source' if study=='depth200_optimum_study' else 'depth-16 compilation of product-prefix QAOA',
            retrospective_depth_choice=study!='full_circuit_transfer'))
    s=STUDIES[0]
    add(s,'flp20',plans[s]['model'],'full_aqc_qaoa_l2','original_aqc_qaoa','full_product_qaoa_l2','original_product_qaoa')
    s=STUDIES[1]
    for old in plans[s]['cases']:
        spec=old['spec'];model=dict(spec,n_qubits=old['n_qubits'],distance=old['distance'],optimum=old['optimum'])
        add(s,spec['id'],model,'full_aqc','original_aqc','full_product','original_product')
    s=STUDIES[2]
    for cid in plans[s]['cases']:
        model=read(ROOT/'outputs/aqc_tensor_qaoa_v2/cases'/f'{cid}.json')['model']
        add(s,cid,model,'compressed_l2','original','product')
    assert len(cases)==9
    support={cid:feasible_costs(m) for cid,m in models.items()}
    blocks={};checks=[]
    for study in STUDIES:
        folder=ROOT/'outputs'/study
        jobs=read(folder/'jobs.json')
        assert all(j['backend']=='ibm_rensselaer' and j['processor']=={'family':'Nighthawk','revision':'1'} for j in jobs)
        expected=csvread(folder/'hardware_by_block.csv');pooled=csvread(folder/'hardware_pooled.csv')
        for path in sorted((folder/'raw_counts').glob('b*.json')):
            raw=read(path)
            for r in raw['rows']:
                if r.get('kind','experiment')!='experiment':continue
                cid=r.get('case',r.get('case_id','flp20'));model=models[cid]
                counts=r['counts'];shots=sum(counts.values());fs=support[cid];opt=model['optimum']
                events={'feasible':0,'optimal':0}
                for bits,k in counts.items():
                    assert len(bits)==model['n_qubits'] and set(bits)<=set('01') and k>0
                    value=int(bits,2)
                    if value in fs:
                        events['feasible']+=k
                        events['optimal']+=k*(abs(fs[value]-opt)<=1e-8)
                key=(study,cid,raw['arm'],r['method'],raw['block'])
                assert key not in blocks
                record=dict(shots=shots,**{e+'_count':int(k) for e,k in events.items()})
                csvrow=next(x for x in expected if x.get('case',x.get('case_id','flp20'))==cid and x['arm']==raw['arm'] and x['method']==r['method'] and int(x['block'])==raw['block'])
                assert all(int(csvrow[k])==v for k,v in record.items())
                blocks[key]=record
                checks.append(dict(study=study,case_id=cid,arm=raw['arm'],method=r['method'],block=raw['block'],job_id=raw['job_id'],**record))
        for r in pooled:
            cid=r.get('case',r.get('case_id','flp20'))
            bs=[blocks[study,cid,r['arm'],r['method'],b] for b in [0,1]]
            assert all(sum(x[k] for x in bs)==int(r[k]) for k in ['shots','feasible_count','optimal_count'])
    assert len(checks)==168
    effects=[]
    for case in sorted(cases,key=lambda c:(c['n_qubits'],c['case_id'])):
        for arm in ['raw','suppressed']:
            for comparison in ['source','product_reference','original_product']:
                if case[comparison] is None:continue
                a=[blocks[case['study'],case['case_id'],arm,case['compiled'],b] for b in [0,1]]
                ref=[blocks[case['study'],case['case_id'],arm,case[comparison],b] for b in [0,1]]
                for event in ['feasible','optimal']:
                    tables=[(x[event+'_count'],x['shots'],y[event+'_count'],y['shots']) for x,y in zip(a,ref)]
                    estimate,low,high=block_mover_interval(tables)
                    exact_estimate,conservative_low,conservative_high=block_difference_interval(tables)
                    assert abs(estimate-exact_estimate)<1e-12
                    ka,na,kb,nb=map(sum,zip(*tables))
                    assert abs(estimate-(ka/na-kb/nb))<1e-12  # Equal shots between blocks in each method.
                    effects.append(dict(**case,arm=arm,comparison=comparison,event=event,
                        compiled_count=ka,compiled_shots=na,reference_count=kb,reference_shots=nb,
                        compiled_probability=ka/na,reference_probability=kb/nb,
                        improvement_pp=100*estimate,ci_low_pp=100*low,ci_high_pp=100*high,
                        conservative_ci_low_pp=100*conservative_low,conservative_ci_high_pp=100*conservative_high))
    write_csv(OUT/'plot_data.csv',effects);write_csv(OUT/'independent_count_checks.csv',checks)
    save(OUT/'plot_specification.json',dict(created_utc=now(),analysis_type='Retrospective descriptive synthesis of existing hardware',
        method='Full-output AQC-Tensor, two physical-edge layers, complete CZ depth 16',
        main_arm='raw',main_reference='Own original source in the same acquisition',
        y='100*(p_compiled-p_reference), in percentage points',
        points='One fixed instance each, with slight horizontal offsets at duplicate widths; never averaged across instances',
        uncertainty='Main: nominal 95% block-stratified MOVER additive-variance intervals from Wilson scores, fixed weights 1/2, Tang (2021) section 2.2. Approximate coverage, including at sparse boundaries. Sensitivity: Bonferroni Clopper-Pearson projection with at least 95% pointwise coverage under block-binomial sampling. Neither accounts for retrospective selection or multiplicity across points.',
        statistical_reference='https://doi.org/10.1002/sim.9230; https://arxiv.org/abs/2110.12636',
        zero_counts='Retained without pseudocounts; zero/zero has zero point estimate and nonzero uncertainty',
        scope='Nine fixed instances, three separate campaigns, unequal source families and fitting budgets; not a scaling law or independent-day replication. Restricted VRP has no capacity/time-window constraints.',
        selection='Same depth 16 everywhere; all available cases and both execution arms retained. Depth choice uses known outcomes in the FLP20 and 25/30-qubit scans. No per-point best-method or best-arm selection.',
        exclusions='Prefix-only and selector studies use different treatments and are not part of this full-output method figure; no verified full-output hardware above 30 qubits or VRP width beyond 12.',
        baseline_caveat='Product-reference companion uses depth-16 compiled product-prefix QAOA at 12/16/20 qubits, but a zero-CZ fit to the same source at 25/30. Original product-prefix QAOA is additionally retained for the first seven cases.',
        cases=cases,input_sha256=inputs,independent_count_sets=len(checks),new_hardware_shots=0))
    return effects


def plot(effects, conservative=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    destination=OUT/'conservative_intervals' if conservative else OUT
    destination.mkdir(exist_ok=True)
    if conservative:
        effects=[dict(r,ci_low_pp=r['conservative_ci_low_pp'],ci_high_pp=r['conservative_ci_high_pp']) for r in effects]
    interval_label='95% conservative intervals' if conservative else '95% block-stratified score intervals'
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'axes.titleweight':'bold','axes.labelsize':11,'savefig.dpi':300})
    def draw(ax,event,arm,comparison,zoom=False):
        rows=[r for r in effects if r['event']==event and r['arm']==arm and r['comparison']==comparison]
        for r in rows:
            jitter=0.32*(int(r['case_id'][-1])-2) if '_s' in r['case_id'] else 0.
            if zoom and r['n_qubits']==20:continue
            point=r['improvement_pp'];color=COLORS[r['problem']]
            ax.errorbar(r['n_qubits']+jitter,point,
                yerr=[[point-r['ci_low_pp']],[r['ci_high_pp']-point]],fmt=MARKERS[r['problem']],
                color=color,ecolor=color,ms=6 if not zoom else 4.5,mew=1.1,
                capsize=3 if not zoom else 2,elinewidth=1.1,zorder=3)
        ax.axhline(0,color='#555555',lw=.8,ls=(0,(4,3)),zorder=1)
        ax.set_xlim(10.5,31.5);ax.set_xticks([12,16,20,25,30])
        ax.grid(axis='y',color='#dddddd',lw=.6,alpha=.8);ax.set_axisbelow(True)
        if not zoom:
            ax.set_xlabel('Number of qubits')
            ax.set_ylabel(('Feasible' if event=='feasible' else 'Optimum')+' probability change (pp)')
            lower=min(r['ci_low_pp'] for r in rows);upper=max(r['ci_high_pp'] for r in rows)
            pad=(upper-lower)*.15;ax.set_ylim(min(lower-pad,-pad*.6),upper+pad)
        return rows
    handles=[Line2D([],[],ls='',marker=MARKERS[p],color=COLORS[p],label=p,markersize=7) for p in ['FLP','VRP','TSP']]
    for arm in ['raw','suppressed']:
        for comparison in ['source','product_reference']:
            for event in ['feasible','optimal']:
                fig,ax=plt.subplots(figsize=(7.1,4.8));fig.subplots_adjust(left=.14,right=.97,bottom=.26,top=.85)
                rows=draw(ax,event,arm,comparison)
                title='Feasible sampling' if event=='feasible' else 'Optimum sampling'
                fig.suptitle(title+' after full-output AQC compression',x=.14,ha='left',y=.985,fontsize=13,weight='bold')
                ref='original source circuit' if comparison=='source' else 'product reference (study-specific)'
                fig.text(.14,.915,f'CZ depth 16  |  {arm} hardware  |  change versus {ref}',fontsize=9,color='#444444')
                ax.legend(handles=handles,loc='upper left',frameon=False,ncol=3,handletextpad=.4,columnspacing=1.1)
                if event=='feasible' and comparison=='source':
                    inset=ax.inset_axes([.60,.39,.37,.44]);draw(inset,event,arm,comparison,True)
                    zrows=[r for r in rows if r['n_qubits']!=20]
                    inset.set_ylim(min(-.05,min(r['ci_low_pp'] for r in zrows))-.08,max(r['ci_high_pp'] for r in zrows)+.15)
                    inset.set_xticks([12,16,25,30]);inset.tick_params(labelsize=7)
                    inset.set_title('Zoom: other eight cases',fontsize=8,pad=4)
                    inset.patch.set_alpha(1.);inset.spines['top'].set_visible(True);inset.spines['right'].set_visible(True)
                fig.text(.14,.065,f'Points: individual instances; {interval_label}.\nShot uncertainty only; repeated widths are offset. Separate campaigns; descriptive comparison.',fontsize=8,color='#555555')
                stem=f'{event}_improvement_{arm}_vs_{comparison}'
                for ext in ['pdf','svg','png']:fig.savefig(destination/f'{stem}.{ext}')
                plt.close(fig)
    for arm in ['raw','suppressed']:
        fig,axs=plt.subplots(1,2,figsize=(10.0,4.2));fig.subplots_adjust(left=.075,right=.985,bottom=.29,top=.82,wspace=.30)
        for ax,event,label in zip(axs,['feasible','optimal'],['(a) Feasible samples','(b) Optimum samples']):
            draw(ax,event,arm,'source');ax.set_title(label,loc='left',fontsize=11,pad=12)
            ax.tick_params(labelsize=9);ax.xaxis.label.set_size(10);ax.yaxis.label.set_size(10)
            if event=='feasible':
                inset=ax.inset_axes([.60,.41,.37,.42]);draw(inset,event,arm,'source',True)
                zrows=[r for r in effects if r['event']==event and r['arm']==arm and r['comparison']=='source' and r['n_qubits']!=20]
                inset.set_ylim(min(-.05,min(r['ci_low_pp'] for r in zrows))-.08,max(r['ci_high_pp'] for r in zrows)+.15)
                inset.set_xticks([12,16,25,30]);inset.tick_params(labelsize=6)
                inset.set_title('Other eight cases',fontsize=7,pad=3)
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.81,.98),frameon=False,ncol=3,fontsize=9)
        fig.text(.075,.95,f'Full-output AQC, CZ depth 16 | {arm} Nighthawk hardware',fontsize=11,weight='bold')
        fig.text(.075,.085,f'Change versus each original source circuit (before full-output compression); {interval_label} (shot uncertainty only).\nPoints are individual instances, offset at repeated widths. Different source families and campaigns; no fitted scaling trend.',fontsize=8,color='#444444')
        for ext in ['pdf','svg','png']:fig.savefig(destination/f'combined_{arm}.{ext}')
        plt.close(fig)


def report(effects):
    rows=[r for r in effects if r['arm']=='raw' and r['comparison']=='source']
    lines=['# Fixed-depth AQC hardware comparison by circuit width','',
        'Recommended method: **full-output AQC-Tensor on physical edges, two native layers, complete CZ depth 16**. The entire source state is compiled; no routed QAOA tail remains. This is the strongest tested common AQC configuration for a multi-problem feasibility figure, not a universally best circuit or an optimum-recovery method.','',
        'The main plots compare each compressed circuit against its own contemporaneously measured original source. Positive values mean an increase in event probability, in **percentage points**. A ratio is deliberately avoided because several baselines have zero observed optima or feasible samples.','',
        '| Problem/case | Qubits | Source → AQC CZ depth | Feasible AQC/source | Feasible change (pp) | Optimum AQC/source | Optimum change (pp) | Shots/method |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    tex=[r'\begin{table}[htbp]\centering\small',r'\caption{Raw hardware counts for the fixed-depth width comparison. F/O denote feasible/optimal counts for AQC versus its own source. Every AQC output has complete CZ depth 16. Counts are not pooled across instances.}\label{tab:width-counts}',
         r'\begin{tabular}{lrrrrr}\toprule Case & Qubits & Source CZ depth & F: AQC/source & O: AQC/source & Shots each\\\midrule']
    for f in [r for r in rows if r['event']=='feasible']:
        o=next(r for r in rows if r['case_id']==f['case_id'] and r['event']=='optimal')
        label=f['case_id'].replace('_s','-').upper()
        lines.append(f"| {label} | {f['n_qubits']} | {f['source_cz_depth']} → 16 | {f['compiled_count']}/{f['reference_count']} | {f['improvement_pp']:+.5f} | {o['compiled_count']}/{o['reference_count']} | {o['improvement_pp']:+.5f} | {f['compiled_shots']:,} |")
        tex.append(f"{label} & {f['n_qubits']} & {f['source_cz_depth']} & {f['compiled_count']}/{f['reference_count']} & {o['compiled_count']}/{o['reference_count']} & {f['compiled_shots']:,}"+r'\\')
    tex.extend([r'\bottomrule\end{tabular}\end{table}','']);(OUT/'counts_table.tex').write_text('\n'.join(tex))
    lines.extend(['','All nine raw feasibility differences are positive. This does not mean all nine pass their original multiple-comparison correction. The optimum differences are small, with rare/zero counts and uncertainty spanning zero. FLP20 full AQC has zero optima versus one from its source, despite its large feasible-sampling gain.','',
        '## Interpretation for the paper','',
        '- Use the feasible and optimum panels together. The supported statement is conditional feasibility recovery through full-output compression; reliable optimum recovery remains unestablished.',
        '- FLP: four facilities/four customers (20 qubits) and five/five (30). TSP: five cities (16 qubits) and six cities (25), with fixed-depot encodings. VRP: three customers plus depot and two nonempty vehicles (12 qubits), without capacities or time windows.',
        '- Show each of the three VRP12 and three TSP16 instances separately. FLP20, TSP25 and FLP30 each have one instance. Do not fit a scaling line or treat shots as independent instance replication.',
        '- The first seven sources are AQC-prefix plus one QAOA layer; the 25/30-qubit sources are standard p=2 QAOA. Targets, fitting budgets and acquisition dates differ. The fixed output depth is the common property.',
        '- Depth 16 is chosen for this retrospective synthesis after inspecting the existing depth comparisons. It was the sole full-compression depth in the six-instance transfer study. These plots are not an independently validated depth-selection rule.',
        '- Raw and suppressed figures are separate, with the same cases and method. No per-instance choice of the better execution arm is made.',
        '- Product-reference companion plots retain stronger controls. At 12/16/20 qubits these are matched depth-16 compilations of product-prefix QAOA; at 25/30 they are zero-CZ fits to the same source. These are not a single matched-budget control family. The original product-prefix QAOA counts are also retained in the CSV, including its higher feasibility on all three routing cases.',
        '', '## Error bars and verification','',
        'For each event and instance, the estimand is the average over the two acquisition blocks of p(AQC) − p(reference). Main figures use nominal 95% MOVER additive-variance intervals built from Wilson score limits, with fixed equal block weights. This accommodates block-dependent rates and effects. Coverage is approximate, not guaranteed for very sparse events. The method follows [Tang (2021), section 2.2](https://arxiv.org/html/2110.12636v1#S2.SS2), [journal DOI](https://doi.org/10.1002/sim.9230). A validation test reproduces the published four-stratum example.',
        '', 'The `conservative_intervals/` figures and CSV sensitivity columns retain a stricter construction: give each of the four arm/block rates a Clopper–Pearson interval with error allocation 0.05/4, then project to the mean difference. These have at least 95% pointwise coverage under block-binomial sampling, at the cost of wider intervals. Neither version adds pseudocounts or gives zero-width uncertainty to zero/zero counts. Both versions leave all optimum differences against the original source unresolved. Some product-control differences are negative; all directions are retained.',
        '', 'These are shot-uncertainty intervals for fixed cases, not instance-population uncertainty, independent-day reproducibility, a new significance family, or selection-adjusted confidence intervals. Original study-specific corrected tests remain unchanged. All 168 experimental count sets in the three input campaigns are rechecked against independently enumerated feasible decisions and optimum costs, including exact optimum counts.',
        '', 'No new quantum hardware jobs or circuit fits were run. PDF/SVG files are vector figures; PNG previews use 300 dpi. `plot_data.csv` includes both events, both execution arms and all available source/product contrasts. Input checksums and the plotting specification are retained.',''])
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    effects=prepare();plot(effects);plot(effects,conservative=True);report(effects)
    print('Rendered nine-instance hardware width plots, with 168 independent count-set checks.')
    for r in effects:
        if r['arm']=='raw' and r['comparison']=='source':
            print(r['case_id'],r['event'],f"{r['improvement_pp']:+.5f} [{r['ci_low_pp']:+.5f}, {r['ci_high_pp']:+.5f}] pp")
