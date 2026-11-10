"""Standalone figures for the larger-instance prefix pilot."""
import csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .tensor_prefix_prepare import settings


def plot():
    config,out=settings()
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    colors={.25:"#2a9d8f",.5:"#457b9d",1.:"#e76f51"}
    markers={"path_1":"o","path_3":"s","patch_1":"^"}
    for spec,ax in zip(config["cases"],axes.flat):
        path=out/"cases"/(spec["id"]+".json")
        if not path.exists():continue
        case=json.loads(path.read_text())
        for r in case["candidates"]:
            label=f"endpoint {r['endpoint']:g}; {r['graph']} L{r['layers']}"
            ax.scatter(r["resources"]["cz_depth"],r["fidelity"],color=colors[r["endpoint"]],
                       marker=markers[f"{r['graph']}_{r['layers']}"],s=65,label=label)
        ax.axhline(.98,color="gray",ls="--",lw=1,label="0.98 fit target")
        ax.set(title=f"{case['model']['problem'].upper()}, {case['model']['n_qubits']} qubits",
               xlabel="Native prefix CZ depth",ylabel="Fidelity to converged MPS target")
        ax.set_ylim(.60,1.0)
        ax.grid(alpha=.2)
    handles=[Line2D([],[],color=color,lw=3,label=f"Endpoint {endpoint:g}") for endpoint,color in colors.items()]
    handles += [Line2D([],[],color="black",marker=marker,ls="",label=label) for marker,label in [("o","Path, 1 layer"),("s","Path, 3 layers"),("^","Patch, 1 layer")]]
    handles.append(Line2D([],[],color="gray",ls="--",label="0.98 fit target"))
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,-.07),ncol=4,fontsize=9)
    fig.savefig(out/"prefix_tradeoffs.pdf",bbox_inches="tight");fig.savefig(out/"prefix_tradeoffs.png",dpi=170,bbox_inches="tight");plt.close(fig)
    depth_rows=[]
    for spec in config["cases"]:
        path=out/"cases"/(spec["id"]+".json")
        if not path.exists():continue
        case=json.loads(path.read_text())
        for name,method in case["methods"].items():
            if not name.startswith("aqc_") or not name.endswith("_prefix"):continue
            tag=name.removeprefix("aqc_").removesuffix("_prefix")
            depth_rows.append((spec["id"]+"\n"+tag,[case["methods"][key]["resources"]["cz_depth"] for key in
                [name,"aqc_"+tag,"refined_"+tag,"standard_p1"]]))
    fig,ax=plt.subplots(figsize=(9,4.7),layout="constrained")
    styles=[("Native prefix only","o","#2a9d8f"),("Native prefix + QAOA","s","#e76f51"),
            ("Uncompressed prefix + QAOA","^","#264653"),("Standard QAOA p=1","x","#457b9d")]
    for i,(label,marker,color) in enumerate(styles):
        ax.scatter(np.arange(len(depth_rows))+(i-1.5)*.08,[r[1][i] for r in depth_rows],marker=marker,color=color,label=label,s=65)
    ax.set_xticks(range(len(depth_rows)),[r[0] for r in depth_rows]);ax.set_yscale("log")
    ax.set_ylabel("CZ depth (log scale)");ax.grid(axis="y",alpha=.2)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,1.2),ncol=2,fontsize=9)
    fig.savefig(out/"circuit_depths.pdf",bbox_inches="tight");fig.savefig(out/"circuit_depths.png",dpi=170,bbox_inches="tight");plt.close(fig)
    path=out/"hardware_final.csv"
    if not path.exists() or not path.read_text().strip():return
    rows=list(csv.DictReader(path.open()))
    index={(r["case_id"],r["method"]):r for r in rows}
    if not index:return
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout="constrained")
    for spec,ax in zip(config["cases"],axes.flat):
        for prefix,label,color in [("standard_p1","Standard QAOA p=1","#457b9d"),("aqc_prefix_qaoa","AQC prefix + QAOA p=1","#e76f51")]:
            values=[float(index[spec["id"],prefix+"_"+stage]["cvar"]) for stage in ["initial","refined"]]
            ax.plot([0,1],values,"o-",color=color,label=label)
        ax.set_xticks([0,1],["Before refinement","After 3 SPSA steps"])
        ax.set_title(spec["id"]);ax.set_ylabel("Held-out CVaR; lower is better")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/"hardware_refinement.pdf");fig.savefig(out/"hardware_refinement.png",dpi=170);plt.close(fig)
    methods=[("standard_p1_initial","QAOA p=1, initial"),("standard_p1_refined","QAOA p=1, refined"),
             ("standard_p2","QAOA p=2"),("shallow_prefix_qaoa","Shallow prefix + QAOA"),
             ("uncompressed_prefix_qaoa","Uncompressed prefix + QAOA"),
             ("aqc_prefix_qaoa_initial","AQC prefix + QAOA, initial"),
             ("aqc_prefix_qaoa_refined","AQC prefix + QAOA, refined")]
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout="constrained")
    for spec,ax in zip(config["cases"],axes.flat):
        for event,offset,marker,color,label in [("feasible",-.13,"o","#457b9d","Feasible"),
                                               ("good",.13,"D","#e76f51","Within 5% of optimum")]:
            values=[index[spec["id"],method] for method,_ in methods]
            p=np.array([float(r["p_"+event]) for r in values])*100
            lo=np.array([float(r[event+"_ci_low"]) for r in values])*100
            hi=np.array([float(r[event+"_ci_high"]) for r in values])*100
            ax.errorbar(p,np.arange(len(methods))+offset,xerr=np.maximum(0,np.array([p-lo,hi-p])),
                        fmt=marker,color=color,ms=4,capsize=2,label=label)
        ax.set_yticks(range(len(methods)),[label for _,label in methods],fontsize=8)
        ax.invert_yaxis();ax.set_xlim(left=0);ax.set_xlabel("Probability per shot (%)")
        ax.set_title(spec["id"]);ax.grid(axis="x",alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.5,-.05),ncol=2)
    fig.suptitle("Held-out hardware samples; pointwise 95% Wilson intervals",fontsize=12)
    fig.savefig(out/"hardware_success.pdf",bbox_inches="tight");fig.savefig(out/"hardware_success.png",dpi=170,bbox_inches="tight");plt.close(fig)


if __name__=="__main__":plot()
