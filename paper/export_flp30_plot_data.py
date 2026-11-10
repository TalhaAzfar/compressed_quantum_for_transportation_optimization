"""Verify saved FLP30 hardware counts and export compact figure data.

Plotting needs only the resulting data/flp30_hardware_plot.json. This audit also
needs the original experiment archive; pass --workspace-root when it is elsewhere.
It does not simulate states, fit circuits, submit jobs, or repeat significance tests.
"""
from pathlib import Path
from statistics import NormalDist
import argparse
import csv
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent


def wilson(k,n):
    z=NormalDist().inv_cdf(.975);p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0.,center-half),min(1.,center+half)


def rate_interval(blocks,event):
    rates=[b[event+'_count']/b['shots'] for b in blocks]
    cis=[wilson(b[event+'_count'],b['shots']) for b in blocks]
    p=sum(rates)/len(rates)
    lo=p-math.sqrt(sum((q-a)**2 for q,(a,b) in zip(rates,cis)))/len(rates)
    hi=p+math.sqrt(sum((b-q)**2 for q,(a,b) in zip(rates,cis)))/len(rates)
    return p,max(0.,lo),min(1.,hi)


def difference_interval(a,b,event):
    estimate=vl=vu=0.;w=1/len(a)
    for x,y in zip(a,b):
        k,n=x[event+'_count'],x['shots'];l,m=y[event+'_count'],y['shots']
        a0,a1=wilson(k,n);b0,b1=wilson(l,m);pa=k/n;pb=l/m
        estimate+=w*(pa-pb)
        vl+=w*w*((pa-a0)**2+(b1-pb)**2)
        vu+=w*w*((a1-pa)**2+(pb-b0)**2)
    return estimate,max(-1.,estimate-math.sqrt(vl)),min(1.,estimate+math.sqrt(vu))


def decode(model,counts):
    f=model['facilities'];c=model['customers'];mask=(1<<f)-1
    result={'shots':0,'feasible_count':0,'optimal_count':0}
    for bits,count in counts.items():
        assert len(bits)==30 and set(bits)<=set('01') and isinstance(count,int) and count>0
        result['shots']+=count;value=int(bits,2);opened=value&mask
        assignments=[(value>>(f+i*f))&mask for i in range(c)]
        if any(a.bit_count()!=1 or a&~opened for a in assignments):continue
        cost=sum(model['opening'][j] for j in range(f) if opened>>j&1)
        cost+=sum(model['service'][i][a.bit_length()-1] for i,a in enumerate(assignments))
        result['feasible_count']+=count
        if abs(cost-model['optimum'])<=1e-8:result['optimal_count']+=count
    return result


def export(workspace,destination):
    inputs={}
    def read(path):
        inputs[path.relative_to(workspace).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    def readcsv(path):
        inputs[path.relative_to(workspace).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
        return list(csv.DictReader(path.open()))
    folder=workspace/'outputs/depth200_optimum_study'
    model=read(workspace/'outputs/aqc_tensor_qaoa_v2/cases/flp30.json')['model']
    assert model['n_qubits']==30 and model['facilities']==model['customers']==5
    jobs=read(folder/'jobs.json')
    assert len(jobs)==4 and all(j['backend']=='ibm_rensselaer' and j['processor']=={'family':'Nighthawk','revision':'1'} for j in jobs)
    plan=read(folder/'hardware_plan.json')
    resources={r['method']:r['resources']['cz_depth'] for r in plan['rows'] if r['case_id']=='flp30'}
    assert sorted(resources.values())==[0,16,48,96,192,292]
    recorded=readcsv(folder/'hardware_by_block.csv')
    pooled=read(folder/'hardware_analysis.json')['rows']
    existing=readcsv(workspace/'outputs/hardware_width_figures_2026/plot_data.csv')
    indexed={};checked=[]
    for path in sorted((folder/'raw_counts').glob('b*.json')):
        raw=read(path)
        job=next(j for j in jobs if j['job_id']==raw['job_id'])
        assert (raw['block'],raw['arm'])==(job['block'],job['arm'])
        for r in raw['rows']:
            if r['case_id']!='flp30':continue
            values=decode(model,r['counts'])
            row=next(v for v in recorded if v['case_id']=='flp30' and v['arm']==raw['arm'] and v['method']==r['method'] and int(v['block'])==raw['block'])
            assert all(values[k]==int(row[k]) for k in values)
            assert values['shots']==6144
            key=(raw['arm'],r['method'],raw['block']);assert key not in indexed
            indexed[key]=values
            checked.append(dict(arm=raw['arm'],method=r['method'],block=raw['block'],job_id=raw['job_id'],**values))
    assert len(indexed)==24
    rates=[];effects=[]
    for arm in ['raw','suppressed']:
        for method in resources:
            blocks=[indexed[arm,method,b] for b in [0,1]]
            original=next(r for r in pooled if (r['case_id'],r['arm'],r['method'])==('flp30',arm,method))
            assert all(sum(b[k] for b in blocks)==original[k] for k in ['shots','feasible_count','optimal_count'])
            p,lo,hi=rate_interval(blocks,'feasible')
            rates.append(dict(study='qaoa_output',case_id='flp30',family='FLP',n_qubits=30,arm=arm,method=method,cz_depth=resources[method],shots=original['shots'],feasible_count=original['feasible_count'],optimal_count=original['optimal_count'],block_feasible_counts=[b['feasible_count'] for b in blocks],p_feasible_percent=100*p,ci_low_percent=100*lo,ci_high_percent=100*hi))
        a=[indexed[arm,'compressed_l2',b] for b in [0,1]]
        for reference,method in [('source','original'),('refined_product','product')]:
            b=[indexed[arm,method,i] for i in [0,1]]
            for event in ['feasible','optimal']:
                p,lo,hi=difference_interval(a,b,event)
                prior=next(r for r in existing if r['case_id']=='flp30' and r['arm']==arm and r['comparison']==('source' if reference=='source' else 'product_reference') and r['event']==event)
                assert all(abs(x-float(prior[k]))<1e-12 for x,k in [(100*p,'improvement_pp'),(100*lo,'ci_low_pp'),(100*hi,'ci_high_pp')])
                effects.append(dict(study='qaoa_output',case_id='flp30',family='FLP',n_qubits=30,arm=arm,event=event,reference=reference,compiled_method='compressed_l2',cz_depth=16,improvement_pp=100*p,ci_low_pp=100*lo,ci_high_pp=100*hi,compiled_count=sum(r[event+'_count'] for r in a),reference_count=sum(r[event+'_count'] for r in b),shots_per_method=sum(r['shots'] for r in a),retrospective_depth_choice=True))
    result={'scope':'One FLP30 instance, five customers and five facilities, complete p=2 QAOA output. Hardware results from Nighthawk. Separate from complete-annealing studies.','depth_choice':'The qubit-width plots show the depth-16 comparison descriptively; its choice is retrospective. All four compressed depths and both treatment arms appear in the depth-feasibility figure. No new significance test or circuit selection.','intervals':'Pointwise 95% Wilson-based additive-variance block means; differences use MOVER, equal weights 1/2. Intervals do not account for retrospective depth choice or multiplicity.','backend':'ibm_rensselaer','processor':{'family':'Nighthawk','revision':'1'},'raw_count_sets_independently_decoded':len(checked),'rates':rates,'effects':effects,'by_block':checked,'input_sha256':inputs,'all_saved_counts_reproduced':True,'all_eight_difference_estimates_and_intervals_match_prior_export':True}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print('Verified and exported FLP30:',len(checked),'raw count sets,',len(rates),'rates,',len(effects),'descriptive comparisons.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace-root',type=Path,default=HERE.parent)
    parser.add_argument('--output',type=Path,default=HERE/'data/flp30_hardware_plot.json')
    args=parser.parse_args();export(args.workspace_root.resolve(),args.output.resolve())
