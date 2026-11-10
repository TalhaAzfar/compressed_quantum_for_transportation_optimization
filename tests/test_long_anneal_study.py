"""Checks specific to the longer-source hardware experiment."""
import pytest


def test_same_six_models_and_prespecified_longer_sources():
    from revision.long_anneal_study import ROOT, settings
    from revision.contribution_hardware_core import read
    import json
    from revision.width_sweep_models import make_model
    c=settings()
    previous={s['id']:s for s in read(ROOT/'configs/native_depth_refinement.json')['cases']}
    assert len(c['cases'])==6
    for spec in c['cases']:
        assert spec==previous[spec['id']]
        model=read(ROOT/'outputs/hardware_width_sweep_2026/cases'/spec['id']/'model.json')
        assert json.loads(json.dumps(make_model(spec))) == model
        diag=read(ROOT/'outputs/reproducibility_2026_10/diagnostics'/spec['id']/'t8_s24.json')
        assert diag['duration']==c['annealing_time']==8
        assert diag['steps']==c['steps']==24
        assert model['n_qubits']==diag['n_qubits']==(25 if spec['problem']=='tsp' else 20)
        assert 0<diag['p_optimal']<diag['p_feasible']<1
    assert c['new_hardware_outcomes_seen'] is False


def test_shot_ceiling_includes_largest_screen_and_all_controls():
    from revision.long_anneal_study import settings
    c=settings()
    maximum_screen=6*9*c['screen_blocks']*c['screen_shots']
    confirmation=6*4*c['confirmation_blocks']*c['confirmation_shots']
    assert c['confirmation_blocks']*c['confirmation_shots']==65536
    assert maximum_screen+confirmation==1794048
    assert maximum_screen+confirmation<=c['hardware_max_total_shots']<=1800000
    assert c['confirmation_days']==1 and c['hardware_max_jobs']==8
    assert c['parallel_workers']*c['worker_rss_limit_bytes']<c['aggregate_rss_limit_bytes']<=35000000000


def test_native_depth_budgets_remain_within_32_and_64():
    from revision.long_anneal_study import depth_layers
    for matchings in [2,3,4]:
        rows=depth_layers(matchings)
        assert rows[:2]==[('product',0,0),('base',2,4*matchings)]
        for _,layers,cap in rows[2:]:
            assert 2*matchings*layers<=cap
            assert cap in [32,64]


@pytest.mark.campaign
def test_prior_frozen_records_remain_valid():
    from revision.reproducibility_study import check_frozen
    check_frozen()
