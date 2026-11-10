"""Checks that matter for interpreting an equal-budget parameter-transfer test."""
import numpy as np


def test_old_parameters_reconstruct_the_saved_training_circuit():
    from revision.cost_adaptation_study import training
    from revision.tensor_prefix_core import native_ansatz
    from revision.nighthawk_study import qpy_read
    from revision.hardware_width_sweep import exact_state
    from revision.contribution_hardware_core import ROOT
    for cid in ['flp20_r1','tsp16_r1','vrp20_r1']:
        case,old=training(cid)
        circuit=native_ansatz(case['model']['n_qubits'],case['edges'],2).assign_parameters(old['parameters'])
        saved=qpy_read(ROOT/old['logical_qpy'])
        assert abs(abs(np.vdot(exact_state(circuit),exact_state(saved)))**2-1)<1e-8


def test_new_objectives_preserve_the_feasible_decision_set():
    from revision.cost_adaptation_study import settings,training
    from revision.width_sweep_models import make_model,feasible_costs,validate_model
    c=settings()
    assert len(c['warm_start_perturbations'])==len(c['cold_start_perturbations'])==3
    assert 6*3*c['hardware_blocks']*c['hardware_shots_per_circuit']==c['hardware_max_total_shots']
    for spec in c['cases']:
        old,_=training(spec['training_case'])
        new=make_model({k:v for k,v in spec.items() if k!='training_case'})
        assert new['seed']!=old['model']['seed']
        assert set(feasible_costs(new))==set(feasible_costs(old['model']))
        assert feasible_costs(new)!=feasible_costs(old['model'])
        assert validate_model(new,exhaustive=False)['penalty_expansion_checked']
