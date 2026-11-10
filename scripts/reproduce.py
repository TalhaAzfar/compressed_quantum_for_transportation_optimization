"""Recreate manuscript plots and budget diagnostics offline, without modifying inputs."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'results')
    p.add_argument('--skip-archive-check', action='store_true', help='Skip verification of compressed original measurements.')
    args = p.parse_args()
    env = dict(os.environ, MPLBACKEND='Agg', MPLCONFIGDIR=str(args.output.resolve()/'matplotlib-cache'))
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
        env.setdefault(key, '1')
    def run(*cmd):
        subprocess.run([sys.executable,*map(str,cmd)], cwd=ROOT, env=env, check=True)
    archive_verified = False
    if not args.skip_archive_check:
        if (ROOT/'recorded_data/index.json').is_file():
            run(ROOT/'scripts/unpack_recorded_data.py', '--verify-only')
            archive_verified = True
        else:
            print('Using compact figure data. Install the optional original-data archive for raw-measurement verification.', flush=True)
    paper = args.output.resolve()/'paper'
    paper.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'paper/data',paper/'data',dirs_exist_ok=True)
    (paper/'figs').mkdir(exist_ok=True)
    plots = ['plot_hardware_comparisons.py','plot_depth_feasibility.py',
             'plot_hardware_validation.py','plot_candidate_quality.py','plot_source_duration.py']
    for name in [*plots,'export_revision_diagnostics.py']:
        shutil.copy2(ROOT/'paper'/name,paper/name)
    for name in plots:
        run(paper/name)
    run(paper/'export_revision_diagnostics.py','--workspace-root',ROOT,
        '--output',paper/'data/revision_diagnostics.json')
    files=[str(p.relative_to(args.output.resolve())) for p in sorted((paper/'figs').glob('*'))]
    report=dict(status='passed',offline=True,hardware_jobs_submitted=0,generated_figures=files,
                archive_verified=archive_verified)
    (args.output/'reproduction.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
