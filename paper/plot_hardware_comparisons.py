"""Render Figures 3--4, distinguishing FLP30 and both FLP42 acquisitions.

The combined Figure 5 is rendered by ``plot_hardware_validation.py``.

The exported probability differences and confidence limits are plotted directly;
this script does not repeat circuit selection or statistical estimation.
"""
from pathlib import Path
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


def save(fig, name, dpi=300):
    for ext in ['pdf', 'png']:
        fig.savefig(ROOT / 'figs' / f'{name}.{ext}', bbox_inches='tight',
                    pad_inches=.06, dpi=dpi)
    plt.close(fig)


def main_comparisons():
    effects = (read('central_analysis.json')['comparisons']
               + read('larger_width_analysis.json')['comparisons']
               + read('flp42_depth_sweep.json')['effects']
               + read('flp30_hardware_plot.json')['effects'])
    labels = {'source': 'uncompressed source preparation',
              'refined_product': 'fitted product preparation'}
    handles = [Line2D([], [], marker=MARKERS[f], color=COLORS[f], ls='',
                      label=f, ms=7) for f in COLORS]
    study_handles = [
        Line2D([], [], marker='o', color=COLORS['FLP'], mfc='white', ls='', ms=7,
               label='FLP42: initial comparison'),
        Line2D([], [], marker='s', color=COLORS['FLP'], mfc='white', ls='', ms=7,
               label='FLP42: depth sweep'),
        Line2D([], [], marker='D', color=COLORS['FLP'], ls='', ms=6,
               label='FLP30: QAOA output, raw'),
        Line2D([], [], marker='D', color=COLORS['FLP'], mfc='white', ls='', ms=6,
               label='FLP30: QAOA output, error suppressed')]
    plotted = []
    for event in ['feasible', 'optimal']:
        for reference, label in labels.items():
            rows = [r for r in effects
                    if r['event'] == event and r['reference'] == reference]
            fig, axes = plt.subplots(1, 2, figsize=(10.3, 4.8),
                                     gridspec_kw={'width_ratios': [1.25, 1]})
            fig.subplots_adjust(left=.115, right=.985, bottom=.30, top=.80,
                                wspace=.28)
            for ax, large in zip(axes, [False, True]):
                for row in rows:
                    extension = row.get('study') == 'larger_width'
                    sweep = row.get('study') == 'flp42_depth_sweep'
                    qaoa = row.get('study') == 'qaoa_output'
                    if (extension or sweep or qaoa) and not large:
                        continue
                    if large and row['n_qubits'] < 14:
                        continue
                    offset = ((-.4 if row['arm'] == 'raw' else .4) if qaoa else
                              -.35 if extension else .35 if sweep else
                              (int(row['case_id'].rsplit('r', 1)[1]) - 2) * .15)
                    if row['n_qubits'] == 20:
                        offset += -.27 if row['family'] == 'FLP' else .27
                    delta = row['improvement_pp']
                    color = COLORS[row['family']]
                    ax.errorbar(row['n_qubits'] + offset, delta,
                                yerr=[[delta - row['ci_low_pp']],
                                      [row['ci_high_pp'] - delta]],
                                fmt='s' if sweep else 'D' if qaoa else MARKERS[row['family']], color=color,
                                capsize=2, ms=7 if (extension or sweep) else 6 if qaoa else 5, elinewidth=1,
                                mfc='white' if (extension or sweep or row.get('identical_circuit_self_control') or
                                               (qaoa and row['arm'] == 'suppressed')) else color)
                    plotted.append(dict(event=event, reference=reference,
                        panel='larger_widths' if large else 'main', study=row.get('study', 'main'),
                        case_id=row['case_id'], arm=row.get('arm'), n_qubits=row['n_qubits'],
                        x_plotted=row['n_qubits']+offset, improvement_pp=delta,
                        ci_low_pp=row['ci_low_pp'], ci_high_pp=row['ci_high_pp']))
                ax.axhline(0, color='#666666', ls='--', lw=.8)
                ax.grid(axis='y', color='#dddddd', lw=.5)
                ax.set_axisbelow(True)
                ax.set_xlabel('Number of qubits')
                ax.set_title('Larger widths: 14–42 qubits' if large else 'Main comparison: 27 instances', fontsize=10)
                ax.set_xticks([14, 16, 20, 25, 30, 42] if large else [6, 9, 10, 12, 14, 16, 20, 25])
                ax.set_xlim((13, 43) if large else (5, 26))
            axes[0].set_ylabel(('Feasible' if event == 'feasible' else 'Optimum')
                               + ' probability improvement (pp)')
            fig.suptitle('Native AQC versus ' + label, x=.08, ha='left',
                         y=.98, fontsize=13, weight='bold')
            fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .945),
                       ncol=3, frameon=False)
            fig.legend(handles=study_handles, loc='lower left', bbox_to_anchor=(.07, .13),
                       ncol=2, frameon=False, fontsize=8, columnspacing=1.1)
            fig.text(.08, .08, 'Nighthawk hardware · pointwise 95% confidence intervals · pp: percentage points', fontsize=8)
            fig.text(.08, .035, 'FLP30: retrospective CZ depth 16; all depths in Figure 2(c). Both FLP42 acquisitions use the same instance.', fontsize=8)
            save(fig, f'new_{event}_vs_' + ('source' if reference == 'source' else 'product'))
    (ROOT/'data/hardware_comparison_plot_rows.json').write_text(
        json.dumps({'rows':plotted,
                    'flp30_scope':'Retrospective CZ-depth-16 QAOA-output comparison; both arms, separate from annealing.',
                    'flp42_scope':'Two separate acquisitions of one FLP instance; original and depth-sweep confirmation are not pooled.'}, indent=2)+'\n')


def calibration_comparison():
    primary = read('replication_analysis.json')['primary_day_specific']
    # Retain the original figure's default font and line sizes.
    with plt.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams['pdf.fonttype'] = 42
        fig, axes = plt.subplots(2, 2, figsize=(9, 7), squeeze=False)
        for day in [0, 1]:
            for index, event in enumerate(['feasible', 'optimal']):
                ax = axes[index, day]
                for row in primary:
                    if (row['day'], row['event'], row['reference']) != (day, event, 'source'):
                        continue
                    offset = (int(row['case_id'][-1]) - 2) * .2
                    color = COLORS[row['family']]
                    delta = row['improvement_pp']
                    ax.errorbar(row['n_qubits'] + offset, delta,
                                yerr=[[delta - row['ci_low_pp']],
                                      [row['ci_high_pp'] - delta]],
                                fmt=MARKERS[row['family']], color=color, capsize=3,
                                mfc='white' if row['identical_circuit_self_control'] else color)
                ax.axhline(0, color='grey', ls='--', lw=.8)
                ax.set(xticks=[16, 20], xlim=(15, 21), xlabel='Number of qubits',
                       ylabel=event.capitalize() + ' improvement (pp)', title=f'Day {day+1}')
                ax.grid(axis='y', alpha=.2)
                ax.spines[['top', 'right']].set_visible(False)
        fig.legend(handles=[Line2D([], [], color=COLORS[f], marker=MARKERS[f],
                                   ls='', label=f) for f in ['FLP', 'TSP']],
                   loc='upper right', ncol=2, frameon=False)
        fig.suptitle('Selected native AQC versus full annealing', x=.08, ha='left', y=.99)
        fig.text(.08, .015, 'Nighthawk · independently generated instances · pointwise 95% confidence intervals', fontsize=8)
        fig.tight_layout(rect=[0, .035, 1, .93])
        save(fig, 'new_two_day_confirmation', dpi=200)


def main():
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 300})
    main_comparisons()


if __name__ == '__main__':
    main()
