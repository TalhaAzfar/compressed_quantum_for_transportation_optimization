"""Recover declared best-iterate results when Numba wraps the deadline exception.

The optimization, seed and 90-second deadline are unchanged. Only a SystemError
whose exception chain contains BudgetExpired is normalized. All other errors
remain failures. Original attempts and their costs stay in the release.
"""
import argparse
import json
import sys
from pathlib import Path
from .full_circuit_compression import ROOT,EXTRA,settings,save


def is_deadline_exception(exception):
    from .steps_native_sweep import BudgetExpired
    seen=set();stack=[exception]
    while stack:
        current=stack.pop()
        if id(current) in seen:continue
        seen.add(id(current))
        if isinstance(current,BudgetExpired):return True
        stack += [e for e in [current.__cause__,current.__context__] if e is not None]
    return False


def normalize_deadline(function):
    def wrapped(*args,**kwargs):
        from .steps_native_sweep import BudgetExpired
        try:return function(*args,**kwargs)
        except SystemError as exception:
            if is_deadline_exception(exception):raise BudgetExpired("Normalized Numba-wrapped fitting deadline") from exception
            raise
    return wrapped


def worker(source,layers,index):
    import scipy.optimize
    from .full_circuit_explicit import worker as original_worker
    scipy.optimize.minimize=normalize_deadline(scipy.optimize.minimize)
    original_worker(source,layers,index)


def run():
    from .corrected_study import file_hash
    from .nighthawk_study import now
    from .memory_guard import guarded_run
    c,out=settings();assert (out/"explicit_complete.json").exists()
    assert not (out/"jobs.json").exists()
    path=out/"deadline_recovery_protocol.json"
    source="revision/recover_full_circuit_deadline.py"
    if path.exists():
        protocol=json.loads(path.read_text());assert protocol["source_sha256"]==file_hash(ROOT/source)
    else:
        tasks=[]
        for p in sorted((out/"explicit_fits").glob("*/*.json")):
            r=json.loads(p.read_text())
            if r.get("verified") or r.get("status")!="worker_failed":continue
            name=f"explicit_{r['source']}_l{r['layers']}_r{r['index']}"
            log=(out/"logs"/(name+".log")).read_text()
            if "BudgetExpired: Declared wall-time cap reached" in log and "SystemError:" in log:
                tasks.append(dict(source=r["source"],layers=r["layers"],index=r["index"],original_result_sha256=file_hash(p)))
        protocol=dict(frozen_utc=now(),tasks=tasks,source_sha256=file_hash(ROOT/source),
            new_hardware_outcomes_seen=False,reason=__doc__,same_fit_seconds_cap=90,maximum_retries_per_failed_seed=1)
        save(path,protocol);(out/"source_snapshot/recover_full_circuit_deadline.py").write_bytes((ROOT/source).read_bytes())
    for task in protocol["tasks"]:
        source,layers,index=task["source"],task["layers"],task["index"]
        name=f"recovered_{source}_l{layers}_r{index}"
        p=out/"explicit_fits"/source/f"l{layers}_r{index}.json"
        archive=out/"deadline_failures"/source/p.name
        process_path=out/"logs"/(name+"_process.json")
        if process_path.exists():continue
        if not archive.exists():
            assert file_hash(p)==task["original_result_sha256"]
            archive.parent.mkdir(parents=True,exist_ok=True);p.rename(archive)
        command=[sys.executable,"-m","revision.local_runner","--extra-site",EXTRA,
                 "revision.recover_full_circuit_deadline","worker",source,str(layers),str(index)]
        result=guarded_run(command,out/"logs"/(name+".log"),address_gib=4,rss_gib=3,reserve_gib=12,seconds=180)
        save(process_path,result)
        if not p.exists():save(p,dict(source=source,layers=layers,index=index,verified=False,status="worker_failed",process=result))
        if result["status"]=="system_memory_reserve":raise RuntimeError(result)
        r=json.loads(p.read_text());print(name,r.get("exact_fidelity"),r.get("verified"),flush=True)
    save(out/"deadline_recovery_complete.json",dict(completed_utc=now(),protocol_sha256=file_hash(path)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=["run","worker"])
    p.add_argument("source",nargs="?");p.add_argument("layers",nargs="?",type=int);p.add_argument("index",nargs="?",type=int)
    a=p.parse_args()
    if a.action=="run":run()
    else:worker(a.source,a.layers,a.index)
