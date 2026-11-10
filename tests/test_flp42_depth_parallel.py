import json
from pathlib import Path
import subprocess
import sys


def test_depth_coverage_precedes_second_starts_and_keeps_original_budgets():
    from revision.flp42_depth_parallel import ordered_tasks
    from revision.contribution_hardware_core import ROOT,read
    c = read(ROOT/'configs/flp42_depth_extension_2026_10.json')
    tasks = ordered_tasks(c)
    assert len(tasks) == 14
    assert len({r['result'] for r in tasks}) == 14
    # If both running L4 starts are adopted, different depths fill their slots.
    done_or_active = {'fit_l2_r2','fit_l2_r3','fit_l4_r2','fit_l4_r3'}
    pending = [r for r in tasks if r['name'] not in done_or_active]
    assert [r['args'] for r in pending[:2]] == [['fit','6','2'],['fit','10','2']]
    assert [r['args'][1] for r in tasks[:7]] == ['2','4','6','10','14','18','22']
    assert all(r['seconds'] == c['fit_seconds']+1800 for r in tasks)


def test_adopted_workers_and_new_workers_share_two_slots(tmp_path,monkeypatch):
    import psutil
    from revision import flp42_depth_parallel as scheduler
    from revision.cpu_memory_guard import physical_cpu_ids
    cpus = physical_cpu_ids()[:2]
    assert len(cpus) == 2
    for folder in ['fits','logs']:
        (tmp_path/folder).mkdir()
    script = tmp_path/'fake_worker.py'
    script.write_text('''import json,sys,time\nfrom pathlib import Path\n_,layer,index=sys.argv[-3:]\ntime.sleep(.15)\nPath(sys.argv[1],"fits",f"l{layer}_r{index}.json").write_text(json.dumps({"verified":True}))\n''')
    c = dict(workers=2,threads_per_worker=1,fit_seconds=20,address_gib=8,
             aggregate_rss_gib=8,system_reserve_gib=0,layers=[2,4])
    monkeypatch.setattr(scheduler.extension,'settings',lambda:(c,tmp_path))
    monkeypatch.setattr(scheduler.extension,'runner',lambda:[sys.executable,str(script),str(tmp_path)])
    # A process launched by an older scheduler remains alive through adoption.
    child = subprocess.Popen([sys.executable,'-c',
        'import time;time.sleep(1.5)', 'revision.flp42_depth_extension'],
        start_new_session=True)
    process = psutil.Process(child.pid)
    process.cpu_affinity([cpus[0]])
    try:
        scheduler.worker_pool([dict(name='fit_l2_r2',pid=child.pid,cpus=[cpus[0]])])
        assert child.wait() == 0
        adopted = json.loads((tmp_path/'logs/fit_l2_r2_process.json').read_text())
        assert adopted['adopted'] is True
        assert adopted['result_present'] is False
        # Missing output is retained as failure; each other task runs once.
        assert json.loads((tmp_path/'fits/l2_r2.json').read_text())['status'] == 'worker_failed'
        for name in ['l4_r2','l2_r3','l4_r3']:
            assert json.loads((tmp_path/'fits'/f'{name}.json').read_text())['verified'] is True
        progress = json.loads((tmp_path/'progress.json').read_text())
        assert progress['running'] == []
        assert progress['pending'] == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
