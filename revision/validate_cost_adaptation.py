"""Recompute adaptation fidelities and decode every hardware shot independently."""
import csv
import json
import pickle

import numpy as np

from .cost_adaptation_study import ROOT, OUT, settings, training
from .cost_adaptation_hardware import verify, analyze
from .contribution_hardware_core import read, save, digest, now, check_isa
from .hardware_width_sweep import exact_state
from .nighthawk_study import evolution, qpy_read
from .tensor_prefix_core import native_ansatz
from .tensor_prefix_models import numpy_model
from .width_sweep_models import decode, feasible_costs
from .native_depth_analysis import write_csv


def decoded_events(state, model):
    # Decode all basis strings, independently of the feasible-enumeration scorer.
    result = {k: 0. for k in ['feasible', 'good', 'optimal']}
    for start in range(0, len(state), 65536):
        indices = np.arange(start, min(start+65536, len(state)), dtype=np.uint64)
        decoded = decode(model, indices)
        probabilities = np.abs(state[indices])**2
        for event in result:
            result[event] += float(probabilities[decoded[event]].sum())
    return {'p_'+k: v for k, v in result.items()}


def main():
    verify(read(OUT/'hardware_protocol.json'))
    config = settings(); numerical = []; count_checks = 0; fits_checked = 0
    target = pickle.loads((OUT/'preflight/target.pkl').read_bytes())
    for spec in config['cases']:
        case = read(OUT/'cases'/spec['id']/'complete.json'); model = case['model']
        parent, old = training(spec['training_case'])
        assert set(feasible_costs(model)) == set(feasible_costs(parent['model']))
        assert model['seed'] != parent['model']['seed']
        ansatz = native_ansatz(model['n_qubits'], parent['edges'], config['layers'])
        reference = exact_state(evolution(numpy_model(model), config['source_duration'], config['source_steps']))
        old_reference = exact_state(evolution(numpy_model(parent['model']), config['source_duration'], config['source_steps']))
        overlap = float(abs(np.vdot(reference, old_reference))**2); del old_reference
        assert abs(overlap-case['target_validation']['source_overlap_with_training']) < 1e-10
        source_events = decoded_events(reference, model)
        for key, value in source_events.items():
            assert abs(value-case['target_validation']['ideal_source_events'][key]) < 1e-10
        for fit in case['fits']:
            state = exact_state(ansatz.assign_parameters(fit['parameters']))
            assert abs(float(abs(np.vdot(reference, state))**2)-fit['exact_fidelity']) < 1e-10
            assert fit['verified'] and fit['objective_difference'] < 1e-8
            assert fit['evaluations'] == 40 and len(fit['history']) == 40
            fits_checked += 1
        del state
        historical = read(ROOT/'outputs/native_depth_refinement_2026_10/cases'/spec['training_case']/'complete.json')
        old_starts = [r for r in historical['rows'] if r['layers'] == 2]
        assert len(old_starts) == 3
        for row in case['rows']:
            method = row['method']
            if method == 'reuse':
                assert row['parameters'] == old['parameters']
                evaluations = 0; fitting_seconds = 0.
            else:
                starts = [f for f in case['fits'] if f['method'] == method]
                selected = min(starts, key=lambda f: (-f['exact_fidelity'], f['start']))
                assert row['parameters'] == selected['parameters']
                assert row['selected_start'] == selected['start']
                evaluations = sum(f['evaluations'] for f in starts)
                fitting_seconds = sum(f['fitting_seconds'] for f in starts)
                assert evaluations == 120
            state = exact_state(qpy_read(ROOT/row['logical_qpy']))
            rebuilt = exact_state(ansatz.assign_parameters(row['parameters']))
            assert abs(float(abs(np.vdot(state, rebuilt))**2)-1) < 1e-10
            assert abs(float(abs(np.vdot(reference, state))**2)-row['exact_fidelity']) < 1e-10
            events = decoded_events(state, model)
            for key, value in events.items():
                assert abs(value-row['ideal_events'][key]) < 1e-10
            resources = row['resources']
            checked = check_isa(qpy_read(ROOT/row['isa_qpy']), target,
                resources['physical'], resources['final_logical_mapping'])
            assert checked['cz_depth'] == resources['cz_depth'] == 16
            assert resources['swap_count'] == 0
            numerical.append(dict(case_id=spec['id'], family=row['family'], n_qubits=row['n_qubits'],
                method=method, exact_fidelity=row['exact_fidelity'], source_overlap_with_training=overlap,
                new_cost_evaluations=evaluations, new_cost_fitting_seconds=fitting_seconds,
                prior_selected_fit_evaluations=old['evaluations'],
                prior_three_two_layer_starts_evaluations=sum(r['evaluations'] for r in old_starts),
                historical_initialization_cost_included=False, cz_depth=16, **events))
        print('Verified', spec['id'], flush=True)
    for block in [0, 1]:
        raw = read(OUT/'raw_counts'/f'block_{block}.json')
        for row in raw['rows']:
            model = read(OUT/'cases'/row['case_id']/'model.json')
            indices = np.array([int(s, 2) for s in row['counts']], dtype=np.uint64)
            weights = np.array(list(row['counts'].values()), dtype=np.int64)
            result = decode(model, indices)
            assert int(weights.sum()) == row['metrics']['shots'] == 8192
            for event in ['feasible', 'good', 'optimal']:
                assert int(weights @ result[event]) == row['metrics'][event+'_count']
            count_checks += 1
    before = read(OUT/'analysis.json')
    analyze(); after = read(OUT/'analysis.json')
    for field in ['summary', 'comparisons', 'raw_sha256', 'quantum_seconds']:
        assert before[field] == after[field]
    jobs = read(OUT/'jobs.json')
    assert len(jobs) == 2 and sum(j['total_shots'] for j in jobs) == 294912
    assert all(j['processor'] == {'family': 'Nighthawk', 'revision': '1'} for j in jobs)
    assert read(OUT/'protocol.json')['frozen_utc'] < read(OUT/'hardware_protocol.json')['frozen_utc'] < min(j['submitted_utc'] for j in jobs)
    assert fits_checked == count_checks == 36
    write_csv(OUT/'numerical_outcomes.csv', numerical)
    report = dict(verified_utc=now(), fits_exact_fidelity_recomputed=fits_checked,
        selected_circuits_reconstructed=18, selected_ideal_events_independently_decoded=18,
        raw_hardware_rows_independently_decoded=count_checks, total_shots=294912,
        inference_contrasts_recomputed=36, source_and_input_hashes_verified=True,
        native_connectivity_and_measurement_maps_verified=True, all_cz_depths=16,
        equal_new_cost_fit_budgets_verified=True, historical_training_cost_excluded_from_new_cost_budget=True,
        minimum_training_source_overlap=min(r['source_overlap_with_training'] for r in numerical),
        maximum_training_source_overlap=max(r['source_overlap_with_training'] for r in numerical),
        summary=after['summary'], quantum_seconds=after['quantum_seconds'],
        verifier_sha256=digest(ROOT/'revision/validate_cost_adaptation.py'))
    save(OUT/'validation.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
