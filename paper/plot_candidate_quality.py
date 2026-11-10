"""Plot conditional cost-gap distributions from saved confirmation samples."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ['source', 'product', 'shallow', 'selected_native']
LABELS = {'source': 'Full annealing', 'product': 'Product',
          'shallow': 'Shallow AQC', 'selected_native': 'Selected AQC'}
COLORS = {'source': '#555555', 'product': '#D55E00',
          'shallow': '#7E57C2', 'selected_native': '#0072B2'}
STYLES = {'source': '--', 'product': ':', 'shallow': '-.', 'selected_native': '-'}


def main():
    data = json.loads((ROOT/'data/candidate_quality_analysis.json').read_text())
    rows = {(r['group'],r['method']): r for r in data['groups']}
    titles = {'FLP20': 'FLP20: main study, three instances',
              'TSP25': 'TSP25: longer annealing, three instances',
              'VRP20': 'VRP20: longer annealing, three instances',
              'FLP42': 'FLP42: one instance'}
    with plt.rc_context({'font.size': 11, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig,axes = plt.subplots(2,2,figsize=(10.3,7.6),squeeze=False)
        for i,group in enumerate(titles):
            ax = axes.flat[i]
            handles = []
            end = max(r['gap_pct'] for method in METHODS
                      for r in data['group_distributions'][group+'|'+method])*1.04
            for method in METHODS:
                dist = data['group_distributions'][group+'|'+method]
                count = rows[group,method]['feasible_count']
                label = LABELS[method]+f' (n={count})'
                if dist:
                    x = np.array([r['gap_pct'] for r in dist])
                    y = np.cumsum([r['count'] for r in dist])/count
                    ax.step(np.r_[0,x,end],np.r_[0,y,1.],where='post',
                            color=COLORS[method],ls=STYLES[method],
                            lw=2 if method=='selected_native' else 1.5)
                handles.append(Line2D([],[],color=COLORS[method],ls=STYLES[method],
                                      lw=2 if method=='selected_native' else 1.5,label=label))
            ax.set(title=f'({chr(97+i)}) '+titles[group],
                   xlabel='Optimality gap of feasible sample (%)',
                   ylabel='Cumulative fraction of feasible samples',
                   ylim=(0,1.02),xlim=(0,end))
            ax.grid(alpha=.2)
            ax.legend(handles=handles,loc='lower right',fontsize=8.5,framealpha=.95)
        fig.tight_layout()
        for ext in ['pdf','png']:
            fig.savefig(ROOT/'figs'/f'new_candidate_quality.{ext}',
                        bbox_inches='tight',pad_inches=.06,dpi=300)
        plt.close(fig)


if __name__=='__main__':
    main()
