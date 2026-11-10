import subprocess
import sys
from revision.memory_guard import guarded_run


def test_hard_address_limit_rejects_allocation_without_consuming_ram():
    code="import resource; assert resource.getrlimit(resource.RLIMIT_AS)[0]==256*1024**2\ntry:\n x=bytearray(512*1024**2)\nexcept MemoryError:\n print('limited')\nelse:\n raise AssertionError('limit missing')"
    result=subprocess.run([sys.executable,"-m","revision.memory_guard","--address-gib","0.25","--",sys.executable,"-c",code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert "limited" in result.stdout


def test_guard_does_not_launch_without_system_reserve(monkeypatch,tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr("revision.memory_guard.psutil.virtual_memory",lambda:SimpleNamespace(available=0))
    result=guarded_run([sys.executable,"-c","raise AssertionError('must not launch')"],tmp_path/"log")
    assert result["status"]=="system_memory_reserve" and not result["launched"]


def test_wall_limit_stops_the_guarded_process(tmp_path):
    result=guarded_run([sys.executable,"-c","import time; time.sleep(10)"],tmp_path/"log",seconds=.2,reserve_gib=0)
    assert result["status"]=="wall_time_limit" and result["returncode"]!=0
