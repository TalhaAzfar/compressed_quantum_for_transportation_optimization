"""Run one subprocess with a hard address limit and a whole-tree RSS watchdog."""
import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
import psutil

GIB = 1024**3


def impose_limits(address_gib=4):
    cap = int(address_gib*GIB)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    # Restrict numerical thread pools before JAX/BLAS initialize.
    if hasattr(os,"sched_getaffinity"):
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    for name in ["OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ[name]="1"
    os.environ["JAX_PLATFORMS"]="cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"


def guarded_run(command, log_path, address_gib=4, rss_gib=3, reserve_gib=12, seconds=600):
    if psutil.virtual_memory().available < reserve_gib*GIB:
        return dict(status="system_memory_reserve", launched=False, peak_tree_rss_mib=0.)
    wrapped=[sys.executable,"-m","revision.memory_guard","--address-gib",str(address_gib),"--",*command]
    start=time.monotonic();peak=0;status=None
    with open(log_path,"a") as log:
        child=subprocess.Popen(wrapped,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            try:
                process=psutil.Process(child.pid)
                tree=[process]+process.children(recursive=True)
                total=sum(p.memory_info().rss for p in tree if p.is_running())
                peak=max(peak,total)
                if total > rss_gib*GIB:status="rss_limit"
                elif psutil.virtual_memory().available < reserve_gib*GIB:status="system_memory_reserve"
                elif time.monotonic()-start > seconds:status="wall_time_limit"
                if status:
                    os.killpg(child.pid,signal.SIGKILL)
                    break
            except psutil.NoSuchProcess:pass
            time.sleep(.2)
        returncode=child.wait()
    return dict(status=status or ("completed" if returncode==0 else "worker_failed"),launched=True,
        returncode=returncode,seconds=time.monotonic()-start,peak_tree_rss_mib=peak/2**20,
        hard_address_space_gib=address_gib,rss_watchdog_gib=rss_gib,system_reserve_gib=reserve_gib)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--address-gib",type=float,default=4)
    p.add_argument("command",nargs=argparse.REMAINDER)
    a=p.parse_args();impose_limits(a.address_gib)
    command=a.command[1:] if a.command and a.command[0]=="--" else a.command
    os.execvp(command[0],command)
