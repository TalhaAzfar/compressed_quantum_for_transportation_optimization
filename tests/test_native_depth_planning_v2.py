from pathlib import Path
import runpy


def test_execution_planner_matches_exact_values_and_all_gradients(monkeypatch):
    from revision import native_depth_objective as original
    from revision.native_depth_objective_v2 import make_objective
    monkeypatch.setattr(original,'make_objective',make_objective)
    tests=runpy.run_path(str(Path(__file__).with_name('test_native_depth_refinement.py')))
    tests['test_sliced_and_unsliced_aqc_gradient_against_exact_finite_difference']()


def test_resume_preserves_history_and_consumes_remaining_budget(tmp_path):
    import numpy as np
    import pytest
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from revision.hardware_width_sweep import dense_target
    from revision.tensor_prefix_core import native_ansatz
    from revision.native_depth_objective_v2 import make_objective
    from revision.native_depth_refinement_v2 import fit_one
    from revision.native_depth_refinement import settings
    from revision.contribution_hardware_core import save,read
    from revision.width_sweep_models import make_model
    model=make_model(dict(id='test',problem='flp',seed=4,customers=1,facilities=1))
    qc=QuantumCircuit(2);qc.h(0);qc.cx(0,1);qc.rz(.4,1)
    reference=Statevector(qc).data;target=dense_target(reference,2,4)
    ansatz=native_ansatz(2,[(0,1)],1)
    objective,_=make_objective(target,ansatz,repeats=1)
    x=np.random.default_rng(123).normal(0,.4,ansatz.num_parameters)
    loss,gradient=objective.loss_function(x)
    first=dict(evaluation=1,seconds=.2,loss=float(loss),gradient_inf_norm=float(np.max(np.abs(gradient))))
    checkpoint=tmp_path/'checkpoint.json'
    prior=dict(parameters=x.tolist(),loss=float(loss),evaluations=1,seconds=.2,history=[first])
    save(checkpoint,prior)
    config=dict(settings(),fit_max_evaluations=12,fit_maxiter=10,minimum_evaluations_if_not_converged=2)
    _,row=fit_one(objective,ansatz,np.zeros_like(x),config,checkpoint,
                  reference=reference,target_dense=reference,model=model)
    assert row['recorded_evaluations_before_resume']==1
    assert row['recorded_seconds_before_resume']==.2
    assert row['history'][0]==first and 1<row['evaluations']<=12
    assert len(read(checkpoint)['history'])==row['evaluations']
    assert row['objective_difference']<1e-8
    prior['loss']+=.2;save(checkpoint,prior)
    with pytest.raises(AssertionError):
        fit_one(objective,ansatz,x,config,checkpoint,reference=reference,target_dense=reference,model=model)


def test_saved_path_reuse_preserves_preprocessing_and_gradient(tmp_path):
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from revision.hardware_width_sweep import dense_target
    from revision.tensor_prefix_core import native_ansatz
    from revision.native_depth_objective_v2 import make_objective
    from revision.contribution_hardware_core import save
    circuit=QuantumCircuit(4);circuit.h(range(4));circuit.rzz(.4,0,3);circuit.rx(.7,1)
    reference=Statevector(circuit).data;target=dense_target(reference,4,16)
    ansatz=native_ansatz(4,[(0,1),(1,2),(2,3),(0,3)],3)
    first,diag=make_objective(target,ansatz,repeats=1)
    path=tmp_path/'path.json';save(path,dict(contraction=diag))
    reused,check=make_objective(target,ansatz,repeats=1,warm_path_file=path)
    assert check['archived_benchmark_path_sha256'] is not None
    assert check['trials'][check['selected_trial']]['source']=='archived benchmark path'
    x=np.random.default_rng(44).normal(0,.3,ansatz.num_parameters)
    a,ga=first.loss_function(x);b,gb=reused.loss_function(x)
    assert abs(float(a)-float(b))<1e-10
    assert np.max(np.abs(np.asarray(ga)-np.asarray(gb)))<1e-8
