"""Work-conserving, depth-first scheduling for the frozen FLP42 extension.

Existing workers are adopted without interruption. This only changes when
fits start; their objective, numerical checks, budgets and hardware selection
remain in the frozen flp42_depth_extension module.
"""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time

from .contribution_hardware_core import ROOT,read,save,digest,now
from . import flp42_depth_extension as extension


def ordered_tasks(config):
    """Cover the depth range with one start before scheduling second starts."""
    return [dict(name=f'fit_l{layers}_r{index}',
                 result=f'fits/l{layers}_r{index}.json',
                 args=['fit',str(layers),str(index)],
                 seconds=config['fit_seconds']+1800)
            for index in [2,3] for layers in config['layers']]


def process_is_live(process):
    import psutil
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def worker_pool(adopted):
    import psutil
    from .cpu_memory_guard import physical_cpu_ids
    config,out = extension.settings()
    cpus = physical_cpu_ids()[:config['workers']*config['threads_per_worker']]
    groups = [cpus[w::config['workers']] for w in range(config['workers'])]
    tasks = ordered_tasks(config)
    byname = {t['name']:t for t in tasks}
    active = []
    for row in adopted:
        task = byname[row['name']]
        process = psutil.Process(row['pid'])
        assert process_is_live(process), 'Adopted worker has exited; reconcile results first'
        assert 'revision.flp42_depth_extension' in process.cmdline()
        assert process.cpu_affinity() == row['cpus']
        active.append(dict(task=task,process=process,popen=None,log=None,
                           cpu=row['cpus'],started=process.create_time(),adopted=True))
    occupied = {tuple(r['cpu']) for r in active}
    available = [g for g in groups if tuple(g) not in occupied]
    assert len(active)+len(available) == config['workers']
    active_names = {r['task']['name'] for r in active}
    pending = [t for t in tasks if t['name'] not in active_names and
               not (out/t['result']).exists()]
    peak = 0
    try:
        while pending or active:
            for row in list(active):
                process,task = row['process'],row['task']
                if process_is_live(process):
                    continue
                code = row['popen'].wait() if row['popen'] is not None else None
                if row['log'] is not None:
                    row['log'].close()
                result = out/task['result']
                save(out/'logs'/f"{task['name']}_process.json",dict(
                     returncode=code,returncode_known=code is not None,
                     adopted=row['adopted'],result_present=result.exists(),
                     seconds=time.time()-row['started'],cpus=row['cpu'],
                     threads=config['threads_per_worker']))
                if not result.exists():
                    save(result,dict(status='worker_failed',verified=False,task=task,
                                     returncode=code,adopted=row['adopted']))
                print('Finished',task['name'],'result present',result.exists(),flush=True)
                active.remove(row)
                available.append(row['cpu'])
            while pending and available:
                task = pending.pop(0)
                cpu = available.pop(0)
                command = [sys.executable,'-m','revision.cpu_memory_guard',
                           '--address-gib',str(config['address_gib']),'--cpus',
                           ','.join(map(str,cpu)),'--threads',str(config['threads_per_worker']),
                           '--',*extension.runner(),*task['args']]
                log = (out/'logs'/f"{task['name']}.log").open('a')
                child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                         start_new_session=True)
                active.append(dict(task=task,process=psutil.Process(child.pid),popen=child,
                                   log=log,cpu=cpu,started=time.time(),adopted=False))
                print('Launched',task['name'],'CPUs',cpu,flush=True)
            rss,observations = 0,[]
            for row in active:
                if time.time()-row['started'] > row['task']['seconds']:
                    raise RuntimeError('Worker exceeded original time budget: '+row['task']['name'])
                try:
                    tree = [row['process']]+row['process'].children(recursive=True)
                    for process in tree:
                        with process.oneshot():
                            resident = process.memory_info().rss
                            rss += resident
                            observations.append(dict(pid=process.pid,rss_gib=resident/2**30,
                                threads=process.num_threads(),
                                cpu_seconds=sum(process.cpu_times()[:2]),
                                affinity=process.cpu_affinity()))
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak,rss)
            memory = psutil.virtual_memory()
            save(out/'progress.json',dict(updated_utc=now(),
                 scheduling='one start per depth first; two workers across depths',
                 running=[r['task']['name'] for r in active],pending=len(pending),
                 pending_tasks=[t['name'] for t in pending],rss_gib=rss/2**30,
                 peak_rss_gib=peak/2**30,available_system_gib=memory.available/2**30,
                 processes=observations))
            if rss > config['aggregate_rss_gib']*2**30:
                raise RuntimeError('Aggregate RAM watchdog reached')
            if memory.available < config['system_reserve_gib']*2**30:
                raise RuntimeError('System memory reserve reached')
            time.sleep(.5)
    finally:
        for row in active:
            if process_is_live(row['process']):
                os.killpg(row['process'].pid,signal.SIGKILL)
            if row['popen'] is not None:
                row['popen'].wait()
            if row['log'] is not None:
                row['log'].close()


def verify_amendment(out):
    amendment = read(out/'scheduling_amendment.json')
    for path,sha in amendment['source_sha256'].items():
        assert digest(ROOT/path) == sha, 'Scheduling source changed after amendment'
    assert not (out/'jobs.json').exists() or (out/'hardware_protocol.json').exists()
    return amendment


def run():
    config,out = extension.settings()
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        amendment = verify_amendment(out)
        base = extension.pilot()
        base.verify_hashes(read(out/'preparation_protocol.json'))
        if not (out/'hardware_protocol.json').exists():
            adopted = read(out/'scheduler_handoff.json')['adopted_workers']
            worker_pool(adopted)
            extension.freeze_hardware()
            protocol = read(out/'hardware_protocol.json')
            protocol['source_sha256'].update(amendment['source_sha256'])
            name = str((out/'scheduling_amendment.json').relative_to(ROOT))
            protocol['input_sha256'][name] = digest(out/'scheduling_amendment.json')
            protocol['scheduling_amendment'] = name
            save(out/'hardware_protocol.json',protocol)
        for phase in ['screen','confirm']:
            if phase == 'confirm' and not (out/'confirmation_protocol.json').exists():
                base.select()
            for block in [0,1]:
                base.submit(phase,block)
            while not base.collect(phase):
                save(out/'pipeline_status.json',dict(status='waiting_for_hardware',
                     phase=phase,updated_utc=now()))
                time.sleep(30)
        extension.analyze()
        save(out/'pipeline_status.json',dict(status='complete',completed_utc=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run'])
    args = parser.parse_args()
    run()
