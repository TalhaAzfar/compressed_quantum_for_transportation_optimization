"""Configurable CPU allocation with one aggregate memory watchdog.

The earlier single-core guard is retained unchanged for frozen studies.
All descendants share this guard's RSS budget, including parallel fits.
"""
import argparse
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import psutil

GIB=1024**3
THREAD_ENV=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS','QUIMB_NUM_THREAD_WORKERS')

def physical_cpu_ids(allowed=None):
    allowed=sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    selected=[];seen=set()
    for cpu in allowed:
        base=Path('/sys/devices/system/cpu')/f'cpu{cpu}'/'topology'
        try:key=((base/'physical_package_id').read_text().strip(),(base/'core_id').read_text().strip())
        except OSError:key=('unknown',cpu)
        if key not in seen:selected.append(cpu);seen.add(key)
    return selected

def impose_limits(address_gib=4,cpu_ids=None,threads=1):
    allowed=set(os.sched_getaffinity(0));cpu_ids=list(cpu_ids or [min(allowed)])
    if not set(cpu_ids)<=allowed:raise ValueError('CPU allocation exceeds inherited affinity')
    if threads<1 or threads>len(cpu_ids):raise ValueError('Thread count must fit CPU allocation')
    if address_gib<=0:raise ValueError('Address-space cap must be positive')
    resource.setrlimit(resource.RLIMIT_AS,(int(address_gib*GIB),)*2)
    os.sched_setaffinity(0,set(cpu_ids))
    for name in THREAD_ENV:os.environ[name]=str(threads)
    os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['OMP_DYNAMIC']='FALSE';os.environ['MKL_DYNAMIC']='FALSE'

def guarded_run(command,log_path,cores=4,threads=1,address_gib=4,rss_gib=3,reserve_gib=12,seconds=600):
    available=physical_cpu_ids()
    if cores<1 or cores>len(available):raise ValueError('Unavailable physical-core count')
    if threads<1 or threads>cores:raise ValueError('Invalid threads/core allocation')
    if min(address_gib,rss_gib,seconds)<=0 or reserve_gib<0:raise ValueError('Invalid resource limits')
    cpus=available[:cores]
    if psutil.virtual_memory().available<reserve_gib*GIB:
        return dict(status='system_memory_reserve',launched=False,peak_tree_rss_mib=0.,cpu_ids=cpus)
    wrapped=[sys.executable,'-m','revision.cpu_memory_guard','--address-gib',str(address_gib),
        '--cpus',','.join(map(str,cpus)),'--threads',str(threads),'--',*command]
    started=time.monotonic();peak=0;status=None;cpu_times={};minimum_available=float('inf')
    Path(log_path).parent.mkdir(parents=True,exist_ok=True)
    with open(log_path,'a') as log:
        child=subprocess.Popen(wrapped,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while child.poll() is None:
                total=0
                try:tree=[psutil.Process(child.pid)];tree+=tree[0].children(recursive=True)
                except psutil.NoSuchProcess:tree=[]
                for proc in tree:
                    try:
                        with proc.oneshot():
                            total+=proc.memory_info().rss;cpu=proc.cpu_times()
                            key=(proc.pid,proc.create_time());cpu_times[key]=cpu.user+cpu.system
                    except psutil.NoSuchProcess:continue
                peak=max(peak,total);system_available=psutil.virtual_memory().available
                minimum_available=min(minimum_available,system_available)
                if total>rss_gib*GIB:status='rss_limit'
                elif system_available<reserve_gib*GIB:status='system_memory_reserve'
                elif time.monotonic()-started>seconds:status='wall_time_limit'
                if status:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    break
                time.sleep(.1)
        except BaseException:
            try:os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            child.wait();raise
        returncode=child.wait()
    elapsed=time.monotonic()-started;cpu_seconds=sum(cpu_times.values())
    return dict(status=status or ('completed' if returncode==0 else 'worker_failed'),launched=True,
        returncode=returncode,seconds=elapsed,peak_tree_rss_mib=peak/2**20,cpu_ids=cpus,
        numerical_threads=threads,sampled_tree_cpu_seconds=cpu_seconds,mean_busy_cores=cpu_seconds/elapsed,
        minimum_system_available_gib=None if minimum_available==float('inf') else minimum_available/GIB,
        hard_address_space_gib=address_gib,aggregate_rss_watchdog_gib=rss_gib,system_reserve_gib=reserve_gib)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--address-gib',type=float,default=4);p.add_argument('--cpus',required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();impose_limits(a.address_gib,[int(x) for x in a.cpus.split(',')],a.threads)
    command=a.command[1:] if a.command and a.command[0]=='--' else a.command
    os.execvp(command[0],command)
