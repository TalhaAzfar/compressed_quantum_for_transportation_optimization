import sys
from revision.memory_guard import guarded_run


def test_rss_guard_accounts_for_descendant_allocations(tmp_path):
    child="import time; allocation=bytearray(96*1024**2); time.sleep(10)"
    parent=f"import subprocess,sys; subprocess.run([sys.executable,'-c',{child!r}])"
    result=guarded_run([sys.executable,"-c",parent],tmp_path/"log",address_gib=.5,
                       rss_gib=.06,reserve_gib=0,seconds=5)
    assert result["status"]=="rss_limit"
    assert result["returncode"]!=0
    assert result["peak_tree_rss_mib"] > .06*1024
