import json
import subprocess
import sys
from types import SimpleNamespace
from revision.cpu_memory_guard import guarded_run,physical_cpu_ids

def test_core_allocation_and_thread_limit_reach_child(tmp_path):
    cpus=physical_cpu_ids()[:4]
    code="import os,json,resource;print(json.dumps({'cpus':sorted(os.sched_getaffinity(0)),'threads':os.environ['OPENBLAS_NUM_THREADS'],'address':resource.getrlimit(resource.RLIMIT_AS)[0]}))"
    log=tmp_path/'child.log'
    result=guarded_run([sys.executable,'-c',code],log,cores=len(cpus),threads=len(cpus),address_gib=.5,reserve_gib=0,seconds=5)
    assert result['status']=='completed'
    got=json.loads(log.read_text());assert got['cpus']==cpus and got['threads']==str(len(cpus))
    assert got['address']==512*1024**2

def test_aggregate_watchdog_counts_parallel_descendants(tmp_path):
    worker="import time; x=bytearray(48*1024**2);time.sleep(10)"
    parent=f"import subprocess,sys; a=[subprocess.Popen([sys.executable,'-c',{worker!r}]) for _ in range(2)]; [p.wait() for p in a]"
    result=guarded_run([sys.executable,'-c',parent],tmp_path/'parallel.log',cores=min(2,len(physical_cpu_ids())),
        address_gib=.5,rss_gib=.09,reserve_gib=0,seconds=5)
    assert result['status']=='rss_limit' and result['returncode']!=0

def test_cpu_guard_does_not_launch_without_system_reserve(monkeypatch,tmp_path):
    monkeypatch.setattr('revision.cpu_memory_guard.psutil.virtual_memory',lambda:SimpleNamespace(available=0))
    result=guarded_run([sys.executable,'-c','raise AssertionError'],tmp_path/'reserve.log',cores=1)
    assert not result['launched'] and result['status']=='system_memory_reserve'

def test_cpu_guard_preserves_hard_allocation_limit(tmp_path):
    code="try:\n x=bytearray(512*1024**2)\nexcept MemoryError:\n print('limited')\nelse:\n raise AssertionError('missing limit')"
    log=tmp_path/'address.log'
    result=guarded_run([sys.executable,'-c',code],log,cores=1,address_gib=.25,reserve_gib=0,seconds=5)
    assert result['status']=='completed' and 'limited' in log.read_text()

def test_cpu_guard_terminates_deadline(tmp_path):
    result=guarded_run([sys.executable,'-c','import time;time.sleep(10)'],tmp_path/'time.log',cores=1,reserve_gib=0,seconds=.2)
    assert result['status']=='wall_time_limit' and result['returncode']!=0
