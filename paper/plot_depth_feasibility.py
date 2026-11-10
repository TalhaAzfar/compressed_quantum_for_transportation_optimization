"""Render measured depth--feasibility, separating screening and confirmation."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parent
COLORS = {'source':'#D55E00', 'refined_product':'#777777',
          'fixed_shallow':'#009E73', 'selected_native':'#0072B2'}
METHODS = {'source':'Uncompressed source', 'refined_product':'Fitted product',
           'fixed_shallow':'Shallow AQC', 'selected_native':'Native AQC'}
MARKERS = {'FLP':'o', 'VRP':'s', 'TSP':'^'}


def main():
    with (ROOT/'data/depth_feasibility_confirmation.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 136
    qaoa = json.loads((ROOT/'data/flp30_hardware_plot.json').read_text())['rates']
    assert len(qaoa) == 12
    sweep = json.loads((ROOT/'data/flp42_depth_sweep.json').read_text())['rates']
    assert len(sweep) == 14
    plotted = []
    plt.rcParams.update({'font.size':10, 'axes.titlesize':11, 'axes.labelsize':11,
                         'pdf.fonttype':42, 'ps.fonttype':42})
    fig = plt.figure(figsize=(10.6,7.4))
    grid = fig.add_gridspec(2,2,height_ratios=[1,1],hspace=.80,wspace=.30)
    axes = [fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
    for ax, study, title in zip(axes, ['short','long'],
            [r'(a) $T_{\mathrm{ann}}=2$, 12 steps', r'(b) $T_{\mathrm{ann}}=8$, 24 steps']):
        selected = [r for r in rows if r['study'] == study or
                    (study=='short' and r['study']=='larger_width')]
        for method in METHODS:
            for r in selected:
                if r['method'] != method: continue
                p, lo, hi = [float(r[k]) for k in ['p_feasible_percent','ci_low_percent','ci_high_percent']]
                ax.errorbar(int(r['cz_depth']), p, yerr=[[p-lo],[hi-p]],
                    fmt=MARKERS[r['family']], color=COLORS[method],
                    mfc='white' if method=='fixed_shallow' else COLORS[method],
                    ms=7 if r['study']=='larger_width' else 4.8 if method=='selected_native' else 4.2,
                    mew=.8, elinewidth=.7, capsize=2, capthick=.7,
                    alpha=.85 if method=='selected_native' else .65,
                    zorder=4 if method=='selected_native' else 3)
                plotted.append(dict(panel=study,study=r['study'],case_id=r['case_id'],
                    method=method,arm=None,n_qubits=int(r['n_qubits']),
                    cz_depth=int(r['cz_depth']),p_feasible_percent=p,
                    ci_low_percent=lo,ci_high_percent=hi))
        ax.set_xscale('symlog',linthresh=64,linscale=1.5)
        ax.set_xlim(-4,11500)
        ticks=[0,16,32,64,256,1024,8192]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel('Executed two-qubit depth (CZ)')
        ax.set_ylabel('Hardware feasibility (%)')
        ax.set_title(title,loc='left',pad=34)
        note=('27 instances; 6–25 qubits; 8,192 shots\nLarger circles: FLP42; 131,072 shots' if study=='short' else
              '6 instances; 20/25 qubits; 65,536 shots')
        ax.text(0,1.025,note,transform=ax.transAxes,fontsize=8)
        ax.grid(axis='y',alpha=.2)
        ax.axhline(0,color='#777777',lw=.65,ls='--',zorder=1)
        ax.spines[['top','right']].set_visible(False)
        if study=='short':
            ax.set_yscale('symlog',linthresh=.1,linscale=.8)
            ax.set_ylim(-.006,22)
            ax.yaxis.set_major_locator(FixedLocator([0,.05,.1,.5,1,5,20]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y,pos:f'{y:g}'))
            ax.yaxis.set_minor_locator(NullLocator())
        else:
            ax.set_ylim(-.006,.25)
    ax = fig.add_subplot(grid[1,0])
    for r in qaoa:
        method = ('source' if r['method']=='original' else
                  'refined_product' if r['method']=='product' else 'selected_native')
        p,lo,hi = [r[k] for k in ['p_feasible_percent','ci_low_percent','ci_high_percent']]
        suppressed = r['arm']=='suppressed'
        ax.errorbar(r['cz_depth'],p,yerr=[[p-lo],[hi-p]],
                    fmt='D' if suppressed else 'o',color=COLORS[method],
                    mfc='white' if suppressed else COLORS[method],ms=6,
                    mew=1,elinewidth=.9,capsize=3)
        plotted.append(dict(panel='qaoa_output',study=r['study'],case_id=r['case_id'],
                    method=r['method'],arm=r['arm'],n_qubits=r['n_qubits'],
                    cz_depth=r['cz_depth'],p_feasible_percent=p,
                    ci_low_percent=lo,ci_high_percent=hi))
    ax.set(xlim=(-5,310),ylim=(-.025,1.0),xticks=[0,16,48,96,192,292],
           xlabel='Executed two-qubit depth (CZ)',ylabel='Hardware feasibility (%)')
    ax.set_title('(c) FLP30: QAOA-output compression',loc='left',pad=10)
    ax.tick_params(axis='x',labelsize=8)
    ax.grid(axis='y',alpha=.2)
    ax.axhline(0,color='#777777',lw=.65,ls='--',zorder=1)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[
        Line2D([],[],marker='o',color='#333333',ls='',ms=6,label='Raw'),
        Line2D([],[],marker='D',color='#333333',mfc='white',ls='',ms=6,label='Error suppressed')],
        loc='upper right',frameon=False,ncol=1,fontsize=8)
    ax = fig.add_subplot(grid[1,1])
    for r in sweep:
        method = ('fixed_shallow' if r['selected_original_shallow'] else
                  'source' if r['method']=='source' else
                  'refined_product' if r['method']=='product' else 'selected_native')
        p,lo,hi = [r[k] for k in ['p_feasible_percent','ci_low_percent','ci_high_percent']]
        confirm = r['phase']=='confirm'
        ax.errorbar(r['cz_depth'],p,yerr=[[p-lo],[hi-p]],
                    fmt='D' if confirm else 'o',color=COLORS[method],
                    mfc=COLORS[method] if confirm else 'white',ms=5.7,
                    mew=1,elinewidth=.8,capsize=2.2,
                    zorder=4 if confirm else 3)
        plotted.append(dict(panel='flp42_depth_sweep',study=r['study'],
                    phase=r['phase'],case_id=r['case_id'],method=r['method'],
                    arm=None,n_qubits=42,cz_depth=r['cz_depth'],
                    p_feasible_percent=p,ci_low_percent=lo,ci_high_percent=hi))
    ax.set_xscale('symlog',linthresh=64,linscale=1.5)
    ax.set(xlim=(-4,3700),ylim=(-.003,.068),
           xlabel='Executed two-qubit depth (CZ)',ylabel='Hardware feasibility (%)')
    ax.xaxis.set_major_locator(FixedLocator([0,18,54,198,2815]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis='x',labelsize=8)
    ax.set_title('(d) FLP42: annealing depth sweep',loc='left',pad=10)
    ax.grid(axis='y',alpha=.2)
    ax.axhline(0,color='#777777',lw=.65,ls='--',zorder=1)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(handles=[
        Line2D([],[],marker='o',color='#333333',mfc='white',ls='',ms=5.7,label='Screening'),
        Line2D([],[],marker='D',color='#333333',ls='',ms=5.7,label='Confirmation')],
        loc='upper right',frameon=False,ncol=1,fontsize=8)
    family_handles = [Line2D([],[],ls='',marker=m,color='#222222',mfc='white',ms=5,label=f)
                      for f,m in MARKERS.items()]
    method_handles = [Line2D([],[],ls='',marker='o',color=COLORS[m],
                      mfc='white' if m=='fixed_shallow' else COLORS[m],ms=5,label=label)
                      for m,label in METHODS.items()]
    fig.legend(handles=family_handles,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.01),frameon=False,fontsize=9)
    fig.legend(handles=method_handles,ncol=4,loc='upper center',bbox_to_anchor=(.5,.96),frameon=False,fontsize=9)
    fig.subplots_adjust(left=.077,right=.989,bottom=.085,top=.815)
    for ext in ['pdf','png']:
        fig.savefig(ROOT/'figs'/f'new_depth_feasibility.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    (ROOT/'data/depth_feasibility_plot_rows.json').write_text(json.dumps({'rows':plotted},indent=2)+'\n')
    print('Rendered depth versus measured hardware feasibility.')


if __name__=='__main__':
    main()
