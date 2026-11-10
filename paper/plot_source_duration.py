"""Render the source-duration figure from the supplied exact probabilities."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / 'data/source_duration_grid.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 36
    colors = {'TSP': '#D55E00', 'VRP': '#009E73'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for case in sorted({row['case_id'] for row in rows}):
        for steps in [12, 24]:
            selected = sorted(
                (row for row in rows if row['case_id'] == case and int(row['steps']) == steps),
                key=lambda row: float(row['duration']),
            )
            assert [float(row['duration']) for row in selected] == [2, 4, 8]
            family = selected[0]['family']
            for ax, event in zip(axes, ['feasible', 'optimal']):
                ax.plot(
                    [float(row['duration']) for row in selected],
                    [100 * float(row['p_' + event]) for row in selected],
                    color=colors[family], ls='--' if steps == 12 else '-',
                    marker='^' if family == 'TSP' else 's', ms=4, alpha=.75,
                )
    for ax, event in zip(axes, ['feasible', 'optimal']):
        ax.set(xlabel='Annealing duration T', ylabel=f'Ideal {event} probability (%)',
               xticks=[2, 4, 8], yscale='log')
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [
        Line2D([], [], color=colors['TSP'], marker='^', label='TSP: 25 qubits'),
        Line2D([], [], color=colors['VRP'], marker='s', label='VRP: 20 qubits'),
        Line2D([], [], color='black', ls='--', label='12 steps'),
        Line2D([], [], color='black', label='24 steps'),
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .91),
               ncol=4, frameon=False)
    fig.suptitle('Annealing duration and ideal solution probabilities', y=.99)
    fig.text(.08, .01, 'Three cost realizations per problem family.', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .82])
    fig.savefig(ROOT / 'figs/new_larger_source_diagnosis.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
