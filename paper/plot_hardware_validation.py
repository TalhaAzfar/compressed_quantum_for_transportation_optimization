"""Combine calibration-day and longer-annealing comparisons in one figure.

Plots frozen probability differences and pointwise intervals without pooling
experiments or recomputing their estimates or multiple-comparison corrections.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
COLORS = {'FLP': '#0072B2', 'VRP': '#009E73', 'TSP': '#D55E00'}
MARKERS = {'FLP': 'o', 'VRP': 's', 'TSP': '^'}


def read(name):
    return json.loads((ROOT / 'data' / name).read_text())


def main():
    replication = read('replication_analysis.json')['primary_day_specific']
    longer = read('long_source_analysis.json')['comparisons']
    sources = ['replication_analysis.json', 'long_source_analysis.json']
    plotted = []
    with plt.rc_context({'font.size': 13, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(3, 2, figsize=(10.3, 10.1), squeeze=False)
        fig.subplots_adjust(left=.11, right=.985, bottom=.07, top=.91,
                            hspace=.55, wspace=.29)
        for study_index in range(3):
            if study_index < 2:
                selected = [r for r in replication
                            if r['reference'] == 'source' and r['day'] == study_index]
                title = f'Calibration day {study_index + 1}'
                study = f'calibration_day_{study_index + 1}'
                ticks, limits = [16, 20], (15.2, 20.8)
            else:
                selected = [r for r in longer if r['reference'] == 'source']
                title = 'Longer annealing: 24 steps, duration 8'
                study = 'longer_annealing'
                ticks, limits = [20, 25], (19.2, 25.8)
            for column, event in enumerate(['feasible', 'optimal']):
                ax = axes[study_index, column]
                for row in selected:
                    if row['event'] != event:
                        continue
                    offset = (int(row['case_id'][-1]) - 2) * .2
                    delta = row['improvement_pp']
                    color = COLORS[row['family']]
                    ax.errorbar(row['n_qubits'] + offset, delta,
                                yerr=[[delta - row['ci_low_pp']],
                                      [row['ci_high_pp'] - delta]],
                                fmt=MARKERS[row['family']], color=color,
                                capsize=3, ms=6, elinewidth=1.1,
                                mfc='white' if row['identical_circuit_self_control'] else color)
                    plotted.append(dict(study=study, panel=chr(97 + 2*study_index + column),
                                        x_plotted=row['n_qubits'] + offset, **row))
                ax.axhline(0, color='#666666', ls='--', lw=.85)
                ax.grid(axis='y', color='#dddddd', lw=.6)
                ax.set_axisbelow(True)
                ax.set(xticks=ticks, xlim=limits, xlabel='Number of qubits',
                       ylabel=('Feasible' if event == 'feasible' else 'Optimum')
                              + ' improvement (pp)')
                ax.set_title(f'({chr(97 + 2*study_index + column)}) {title}',
                             loc='left', fontsize=12, pad=12)
        # Use common limits only across repeat executions of the same cases.
        # The longer-schedule experiment retains its own probability scale.
        for column in range(2):
            lo = min(axes[row, column].get_ylim()[0] for row in [0, 1])
            hi = max(axes[row, column].get_ylim()[1] for row in [0, 1])
            for row in [0, 1]:
                axes[row, column].set_ylim(lo, hi)
        handles = [Line2D([], [], color=COLORS[f], marker=MARKERS[f],
                          ls='', label=f, ms=7) for f in COLORS]
        fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.99, .995),
                   ncol=3, frameon=False)
        fig.suptitle('Selected native AQC versus full annealing',
                     x=.11, ha='left', y=.975, fontsize=15, weight='bold')
        for ext in ['pdf', 'png']:
            fig.savefig(ROOT / 'figs' / f'new_hardware_validation.{ext}',
                        bbox_inches='tight', pad_inches=.06, dpi=300)
        plt.close(fig)
    assert len(plotted) == 36
    provenance = {'description': 'Frozen source comparisons; experiments and correction families remain separate.',
                  'source_sha256': {name: hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()
                                    for name in sources},
                  'rows': plotted}
    (ROOT / 'data/hardware_validation_plot.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
