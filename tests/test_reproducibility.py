import math
import numpy as np
import pytest


def test_rare_event_projection_stays_stable_and_is_minimal():
    from revision.reproducibility_diagnostics import shots_for_hit, probability_of_hit
    for p in [1e-12, 1e-6, .001, .4, 1.]:
        n = shots_for_hit(p)
        assert probability_of_hit(p, n) >= .95-1e-14
        assert probability_of_hit(p, n-1) < .95+1e-14
    assert shots_for_hit(0) is None
    assert probability_of_hit(0,8192) == 0
    assert probability_of_hit(1,0) == 0
    assert abs(probability_of_hit(1e-12,8192)-8192e-12) < 1e-15
    with pytest.raises(ValueError): shots_for_hit(-.1)


def test_new_cost_models_preserve_feasible_set_and_independent_optimum():
    from revision.reproducibility_study import settings
    from revision.width_sweep_models import make_model, validate_model, feasible_costs
    from revision.contribution_hardware_core import ROOT, read
    previous = read(ROOT/'configs/native_depth_refinement.json')['cases']
    specs = settings()['cases']
    assert not {s['seed'] for s in previous} & {s['seed'] for s in specs}
    for spec in specs:
        model=make_model(spec); check=validate_model(model)
        assert check['penalty_expansion_checked'] and check['exact_optimum']==min(feasible_costs(model).values())
        assert model['n_qubits']==(20 if spec['problem']=='flp' else 16)


def test_no_unverified_extension_of_routing_encoding():
    from revision.width_sweep_models import make_model
    with pytest.raises(ValueError, match='certified only'):
        make_model(dict(id='unsupported',problem='vrp',seed=1,nodes=6,vehicles=2))
