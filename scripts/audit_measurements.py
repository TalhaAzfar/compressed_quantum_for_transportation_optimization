"""Decode original hardware counts and verify recorded event totals, without Qiskit."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from unpack_recorded_data import entries,read_record
from revision.width_sweep_models import metrics

@lru_cache(maxsize=None)
def load(path):
    return json.loads(read_record(path))

def model_for(study,cid):
    if study in ['hardware_width_sweep_2026','native_depth_refinement_2026_10']:
        return load(f'outputs/hardware_width_sweep_2026/cases/{cid}/model.json')
    if study in ['reproducibility_2026_10','long_anneal_hardware_2026_10','cost_adaptation_2026_10']:
        return load(f'outputs/{study}/cases/{cid}/model.json')
    if study in ['larger_width_feasibility_2026_10','flp42_depth_extension_2026_10']:
        return load(f'outputs/{study}/case.json')['model']
    if study in ['depth200_optimum_study','aqc_tensor_qaoa_v2']:
        return load(f'outputs/aqc_tensor_qaoa_v2/cases/{cid}.json')['model']
    if study=='full_circuit_mitigation':
        return load('outputs/aqc_tensor_qaoa_v2/cases/flp20.json')['model']
    raise ValueError(f'Unrecognized study {study}')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'results/hardware_counts_audit.json')
    args=p.parse_args()
    checked=[];calibrations=0
    for entry in entries():
        path=Path(entry['path'])
        if 'raw_counts' not in path.parts or path.suffix!='.json':continue
        study=path.parts[1]
        # Do not retain large count dictionaries across input files.
        data=json.loads(read_record(entry['path']))
        for row in data['rows']:
            counts=row['counts']
            if not counts or any(not isinstance(n,int) or n<=0 for n in counts.values()):
                raise ValueError(f'Invalid counts in {path}')
            if row.get('kind','experiment')!='experiment':
                if sum(counts.values())!=row['shots']:raise ValueError(f'Invalid calibration shots: {path}')
                calibrations+=1
                continue
            cid=row.get('case_id','flp42' if '42' in study or study=='larger_width_feasibility_2026_10' else 'flp20')
            model=model_for(study,cid)
            if any(len(s.replace(' ',''))!=model['n_qubits'] or set(s.replace(' ',''))-{'0','1'} for s in counts):
                raise ValueError(f'Invalid bit order/width in {path}')
            values=metrics(model,counts)
            recorded=row.get('metrics',{})
            for name in ['shots','feasible_count','good_count','optimal_count']:
                if name in recorded and recorded[name]!=values[name]:
                    raise ValueError(f'{path} {cid} {row["method"]} {name}: decoded {values[name]}, recorded {recorded[name]}')
            if 'shots' in row and row['shots']!=values['shots']:
                raise ValueError(f'Shot total mismatch: {path}')
            checked.append(dict(file=entry['path'],case_id=cid,method=row['method'],
                                block=data.get('block'),phase=data.get('phase',data.get('stage')),
                                **{k:values[k] for k in ['shots','feasible_count','good_count','optimal_count']}))
    report=dict(status='passed',hardware_distributions_decoded=len(checked),
                calibration_distributions_checked=calibrations,
                shots_decoded=sum(r['shots'] for r in checked),rows=checked,
                scope='All included acquisition phases are audited. Screening and confirmation remain separate. No new significance tests or circuit selection.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
