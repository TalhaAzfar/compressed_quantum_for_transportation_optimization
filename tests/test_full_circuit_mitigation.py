import numpy as np
import pytest

from revision.readout_indicator_mitigation import assignment_matrices,apply_local,event_weights,weighted_estimate,bit_mapping
from revision.full_circuit_hardware import select_fits,existing_submission


def test_asymmetric_readout_inverse_recovers_events_with_little_endian_order():
    a=np.array([[[.9,.2],[.1,.8]],[[.96,.3],[.04,.7]],[[.8,.07],[.2,.93]]])
    p=np.array([.03,.06,.11,.21,.04,.08,.17,.3]);p/=p.sum()
    full=np.kron(a[2],np.kron(a[1],a[0]))
    measured=full@p
    np.testing.assert_allclose(apply_local(p,a),measured,atol=1e-14)
    events=np.column_stack([np.array([0,1,0,1,1,0,0,0]),np.arange(8)==6])
    weights=event_weights(events,a)
    np.testing.assert_allclose(measured@weights,p@events,atol=1e-14)
    assert weights.min()<0,"Signed event weights must not be clipped"


def test_calibration_and_weighted_uncertainty_are_independent_of_count_order():
    matrices=assignment_matrices({"00":72,"01":8,"10":18,"11":2},
                                {"00":6,"01":14,"10":24,"11":56},2)
    np.testing.assert_allclose(matrices,[[[.9,.3],[.1,.7]],[[.8,.2],[.2,.8]]])
    mean,se=weighted_estimate({"00":3,"11":1},np.array([-.2,0.,0.,1.4]))
    assert mean==pytest.approx(.2)
    assert se==pytest.approx(np.std([-.2,-.2,-.2,1.4],ddof=1)/2)


def test_physical_readout_mapping_tracks_routed_classical_bits():
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(9,3);qc.measure(7,0);qc.measure(2,1);qc.measure(4,2)
    assert bit_mapping(qc,[2,4,7])==[2,0,1]
    a=np.array([[[.9,.2],[.1,.8]],[[.96,.3],[.04,.7]],[[.8,.07],[.2,.93]]])
    physical=np.array([0,1,0,0,0,0,0,0.])
    # Logical 010 corresponds to physical 001 for this mapping.
    logical=np.array([0,0,1,0,0,0,0,0.])
    assert apply_local(logical,a[[2,0,1]])[2]==pytest.approx(apply_local(physical,a)[1])


def test_selection_requires_both_members_at_matched_gate_budget():
    def r(source,index,f,depth=2,verified=True):
        return dict(source=source,index=index,exact_fidelity=f,layers=depth,verified=verified,
                    resources=dict(cz_count=50*depth,cz_depth=8*depth))
    rows=[r("a",0,.8),r("a",1,.9),r("b",0,.7),r("a",0,.99,5),r("b",0,1.,5,False)]
    selected=select_fits(rows,[2,5],["a","b"])
    assert [(v["source"],v["index"]) for v in selected]==[("a",1),("b",0)]


def test_ambiguous_hardware_submission_cannot_be_repeated(tmp_path):
    (tmp_path/"intent_0_raw.json").write_text("{}")
    with pytest.raises(RuntimeError,match="reconcile"):existing_submission(tmp_path,0,"raw")
    (tmp_path/"jobs.json").write_text('[{"block":0,"arm":"raw"}]')
    assert existing_submission(tmp_path,0,"raw")[1]


def test_sampler_pub_honors_individual_calibration_shots():
    from qiskit import QuantumCircuit
    from qiskit.primitives.containers import SamplerPub
    qc=QuantumCircuit(2);qc.measure_all()
    assert SamplerPub.coerce((qc,None,4096),shots=8192).shots==4096


def test_full_target_mps_conversion_matches_asymmetric_entangled_state():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from revision.tensor_prefix_core import mps_state
    from revision.full_circuit_compression import mps_dense_qiskit
    qc=QuantumCircuit(5);qc.x(0);qc.ry(.41,3);qc.cx(3,1);qc.rz(.72,1);qc.rx(-.22,4)
    mps=mps_state(qc,16)
    dense=mps_dense_qiskit(mps,5)
    np.testing.assert_allclose(dense,Statevector.from_instruction(qc).data,atol=1e-12)


def test_explicit_mps_gradient_matches_exact_directional_derivative():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
    from revision.tensor_prefix_core import native_ansatz,mps_state
    from revision.full_circuit_explicit import explicit_settings
    qc=QuantumCircuit(4);qc.h(range(4));qc.ry(.4,1);qc.rzz(.7,0,3);qc.rx(-.3,2)
    ansatz=native_ansatz(4,[(0,1),(1,2),(2,3)],2)
    target=mps_state(qc,16);objective=MaximizeStateFidelity(target,ansatz,explicit_settings(16,cutoff=0.))
    rng=np.random.default_rng(94501);x=rng.normal(0,.2,ansatz.num_parameters);direction=rng.normal(size=len(x))
    state=Statevector.from_instruction(qc).data
    def exact_loss(theta):return 1-abs(np.vdot(state,Statevector.from_instruction(ansatz.assign_parameters(theta)).data))**2
    loss,gradient=objective.loss_function(x);epsilon=1e-6
    assert loss==pytest.approx(exact_loss(x),abs=1e-10)
    assert np.dot(gradient,direction)==pytest.approx((exact_loss(x+epsilon*direction)-exact_loss(x-epsilon*direction))/(2*epsilon),abs=2e-7)


def test_analysis_keeps_raw_counts_and_identity_readout_correction_consistent(tmp_path,monkeypatch):
    import csv,json
    from revision import analyze_full_circuit_mitigation as analysis
    from revision.tensor_prefix_models import make_model,decode
    from revision.prefix_diagnostic import feasible_assignments
    model=make_model(dict(problem="flp",customers=1,facilities=2,seed=94502));n=model["n_qubits"]
    feasible,costs=feasible_assignments(model);chosen=int(feasible[np.argmin(costs)])
    out=tmp_path/"experiment";(out/"raw_counts").mkdir(parents=True)
    (tmp_path/"revision").mkdir()
    for name in ["analyze_full_circuit_mitigation.py","readout_indicator_mitigation.py"]:
        (tmp_path/"revision"/name).write_text("# synthetic test provenance\n")
    methods=[dict(method=f"full_{source}_l2",ideal=dict(p_feasible=.3,p_good=.3)) for source in ["aqc_qaoa","product_qaoa"]]
    total=0
    for block in range(2):
        for arm in ["raw","suppressed"]:
            rows=[]
            for index,r in enumerate(methods):
                k=24+index*8+block
                rows.append(dict(**r,kind="experiment",shots=128,measurement_mapping=list(range(n)),
                    counts={format(chosen,f"0{n}b"):k,"0"*n:128-k}))
            if arm=="raw":
                for index,value in enumerate([0,(1<<n)-1,5,10,3,12]):
                    rows.append(dict(method=f"readout_{index}",kind="calibration" if index<2 else "calibration_check",
                        prepared=value,shots=512,counts={format(value,f"0{n}b"):512}))
            total+=sum(r["shots"] for r in rows)
            (out/"raw_counts"/f"b{block}_{arm}.json").write_text(json.dumps(dict(block=block,arm=arm,rows=rows,metrics={})))
    (out/"hardware_plan.json").write_text(json.dumps(dict(rows=methods,model=model,physical=list(range(n)),
        arms=["raw","suppressed"],expected_shots=total)))
    config=dict(hardware_blocks=2,hardware_layers=[2,5],bootstrap_replicates=12,bootstrap_seed=94503)
    monkeypatch.setattr(analysis,"ROOT",tmp_path);monkeypatch.setattr(analysis,"settings",lambda:(config,out))
    analysis.analyze()
    result=list(csv.DictReader((out/"mitigated_events.csv").open()))
    for row in result:
        assert float(row["corrected_estimate"])==pytest.approx(float(row["raw_probability"]),abs=1e-14)
        assert float(row["correction_ci_low"])==pytest.approx(0.,abs=1e-14)
        assert float(row["correction_ci_high"])==pytest.approx(0.,abs=1e-14)
    summary=json.loads((out/"analysis_summary.json").read_text())
    assert summary["shots"]==total and summary["independently_checked_count_sets"]==8
    contrasts=list(csv.DictReader((out/"primary_contrasts.csv").open()))
    assert len(contrasts)==8 and sum(r["status"]=="unavailable_pair" for r in contrasts)==4


def test_wrapped_deadline_is_normalized_without_hiding_other_system_errors():
    from revision.recover_full_circuit_deadline import normalize_deadline
    from revision.steps_native_sweep import BudgetExpired
    def wrapped_deadline():
        try:raise BudgetExpired("declared limit")
        except BudgetExpired as exc:raise SystemError("Numba wrapper") from exc
    with pytest.raises(BudgetExpired,match="Normalized"):normalize_deadline(wrapped_deadline)()
    def unrelated():raise SystemError("unrelated implementation error")
    with pytest.raises(SystemError,match="unrelated"):normalize_deadline(unrelated)()
