# Hardware-native compression for transportation optimization

Research code for **Hardware-Native Compression of Quantum Annealing States for Transportation Optimization Models** by Talha Azfar, Ruimin Ke, Sean He, Cara Wang, and José Holguín-Veras.

The study fits shallow circuits to digitized annealing states and evaluates the resulting transportation candidates on superconducting hardware. It includes facility location (FLP), the traveling salesperson problem (TSP), and vehicle routing (VRP). The principal outcome is measured feasible-sample probability, with transportation costs, optimality gaps, approximation fidelity, and executed two-qubit depth as complementary measures. The method is a compilation strategy, rather than a complete transportation solver or a claim of quantum computational advantage.


## Quick start

Use Python 3.10 on Linux. The analysis installation recreates the figures from the included compact data without the original experiment archive.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/reproduce.py
```

`reproduce.py` regenerates the plots and sampling-budget diagnostics under `results/paper/`. If the optional experiment archive is installed, it verifies those records first. Its report states whether the archive was verified. Without that archive, it uses only the included compact data. These commands do not fit circuits, contact IBM, or submit quantum jobs.

Install the full environment for new fitting and circuit inspection. The complete numerical test suite also requires the optional data archive, as described below.

```bash
python -m pip install -r requirements.txt
python examples/small_compression.py --problem flp
python examples/small_compression.py --problem tsp
```

The small example compresses a complete 12-step annealing circuit using AQC-Tensor, with a chain as the coupling graph. It checks MPS fidelity against an exact state and reports ideal feasibility and transpiled CZ depths. Its output is a correctness example, not an additional hardware experiment. VRP is also available through `--problem vrp`. The distributed requirements record the campaign package versions. The test report describes the environment used to check this release.

## Contents

| Path | Contents |
| --- | --- |
| `revision/` | Model encodings, native ansatz, fitting, circuit compilation, acquisition, statistical analysis, mitigation, and cost analysis |
| `configs/` | Experiment specifications with personal IBM account settings removed |
| `examples/small_compression.py` | Small offline example for FLP, TSP, or VRP |
| `scripts/` | Offline figure reproduction, raw-measurement audit, and archive verification/unpacking |
| `paper/` | Figure scripts, exporters, and the compact numerical data used by the manuscript |
| `ORIGINAL_DATA.json` | Filename and SHA-256 checksum of the separately distributed original-data archive |
| `tests/` | Numerical, decoding, statistical, resource-control, and acquisition-guard tests |
| `SOURCE_PROVENANCE.json` | Original and distributed source hashes, identifying portability changes |
| `VALIDATION.md` | Checks performed on this release |

Generated files, virtual environments, credentials, optional `recorded_data/`, and unpacked `outputs/` are excluded through `.gitignore`.

## Studies and entry points

| Study | Main modules | Original output subtree |
| --- | --- | --- |
| Transportation encodings and decoding | `width_sweep_models`, `tensor_prefix_models` | `hardware_width_sweep_2026/cases` |
| Complete annealing compression, 6–25 qubits | `native_depth_refinement_v2`, `native_depth_objective_v2`, `native_depth_hardware`, `native_depth_analysis` | `native_depth_refinement_2026_10` |
| Repeated hardware evaluation on new cost instances | `reproducibility_study`, `reproducibility_hardware` | `reproducibility_2026_10` |
| Longer annealing sources, TSP25 and VRP20 | `long_anneal_study`, `long_anneal_hardware` | `long_anneal_hardware_2026_10` |
| Complete annealing compression, FLP42 | `larger_width_feasibility`, `flp42_fidelity_parallel` | `larger_width_feasibility_2026_10` |
| FLP42 native depth sweep to 198 CZ layers | `flp42_depth_extension`, `flp42_depth_parallel` | `flp42_depth_extension_2026_10` |
| QAOA prefix and continuation | `tensor_prefix_core`, `tensor_prefix_prepare`, `tensor_prefix_hardware`, `prefix_diagnostic` | `aqc_tensor_qaoa_v2`, `prefix_diagnostic` |
| Complete QAOA output compression, TSP25 and FLP30 | `depth200_optimum_fit`, `depth200_optimum_hardware` | `depth200_optimum_study` |
| FLP20 output compression and readout mitigation | `full_circuit_compression`, `full_circuit_hardware`, `analyze_full_circuit_mitigation`, `readout_indicator_mitigation` | `full_circuit_mitigation` |
| Cost changes, parameter adaptation, and candidate reuse | `cost_adaptation_study`, `cost_adaptation_hardware`, `cost_reuse_analysis` | `cost_adaptation_2026_10`, `cost_reuse_analysis_2026_10_11` |
| Feasible-candidate costs and gaps | `analyze_feasible_candidate_quality` | `feasible_candidate_quality_2026_10_14` |

Module names in this table are under `revision`. Use `python -m revision.<module> --help` where a command-line interface is provided. Some original campaign utilities use positional actions instead. Acquisition modules document the sequence of preparation, selection, submission, collection, and analysis. Several helper modules are libraries without command-line interfaces.

## Optional original experiment archive

For raw-bitstring audits, original circuit files, or tests that use recorded hardware targets, obtain the separately distributed `hardware-aware-transport-qaoa-data.zip`. Its SHA-256 checksum is in `ORIGINAL_DATA.json`. This archive can be hosted as a repository release asset or a separate data deposit. It is not required for figure reproduction from the compact data.

Place the data ZIP alongside the code folder and run these commands from the code folder.

```bash
python -m zipfile -e ../hardware-aware-transport-qaoa-data.zip .
python scripts/unpack_recorded_data.py --verify-only
python scripts/audit_measurements.py
```

Extraction creates `recorded_data/`. `audit_measurements.py` decodes original bitstrings and checks feasible, near-optimal, and optimal totals against the recorded metrics, writing `results/hardware_counts_audit.json`.

To reconstruct candidate costs directly from measurements, or run the full portable test suite after installing `requirements.txt`, unpack the original files into their historical `outputs/` paths.

```bash
python scripts/unpack_recorded_data.py
python -m revision.analyze_feasible_candidate_quality
python -m pytest -m "not campaign"
```

This reconstructs the cost distributions for the 27 main instances, the six longer-source comparisons, and the initial FLP42 comparison. It verifies original raw-count hashes, decodes feasibility and costs, and checks feasible outcomes with an independent scalar decoder. Results are written to `outputs/feasible_candidate_quality_2026_10_14/`. The later FLP42 depth-sweep confirmation is a separate acquisition and is covered by the measurement audit and the depth-sweep compact data.

To unpack only one study, pass `--study native_depth_refinement_2026_10`. To check compressed and uncompressed hashes without extracting, use `--verify-only`. Existing files with different hashes are never overwritten. QPY and pickle records should be loaded only when their provenance is trusted. The default plotting and measurement-audit commands do not deserialize pickles.

## New fitting and hardware work

The numerical components are reusable independently of the campaign orchestration.

- `width_sweep_models.make_model` builds transportation models and certified small-instance objective references.
- `nighthawk_study.evolution` generates the digitized annealing circuit. `duration` is total evolution time and `steps` is the Trotter step count. An annealing time of 2 does not mean two Trotter steps.
- `tensor_prefix_core.native_ansatz` restricts entangling interactions to supplied edges. Physical native compilation uses the selected hardware qubits and the backend target.
- `native_depth_objective_v2` and `native_depth_refinement_v2` implement the bounded tensor-contraction fitting and checkpointed optimization used in the depth studies.
- `tensor_prefix_sabre` and `depth200_native_cleanup` handle native compilation and gate cancellation.

The exact state is a benchmark validation tool on tractable instances, rather than a required representation of the fitting target. The 42-qubit experiment uses MPS targets. Larger-width MPS-only fitting and separate compression of circuit segments remain directions for further evaluation.

### Configure your own IBM credentials

Offline reproduction does not need an IBM account, API key, or instance CRN. For new hardware work, use your own IBM Cloud API key and IBM Quantum instance CRN. [IBM's account setup guide](https://quantum.cloud.ibm.com/docs/en/guides/initialize-account) explains how to obtain them. Keep both values out of source files, notebooks, experiment configurations, and Git commits.

After installing `requirements.txt`, run the local setup helper from this folder.

```bash
python scripts/configure_ibm_account.py
export IBM_QUANTUM_ACCOUNT=transport-aqc
```

The helper prompts privately for the key and CRN without echoing either value. It saves a named Qiskit account in `~/.qiskit/qiskit-ibm.json`, outside this repository. It does not contact IBM or submit a job, and it refuses to overwrite an existing account with the same name. Subsequent acquisition commands use this account through `IBM_QUANTUM_ACCOUNT`. If you already have a saved account, set that variable to its name and skip the setup helper.

As an alternative, provide `IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_INSTANCE` through your local shell environment or a private secret manager. Do not paste their values into commands retained in shell history. These environment variables override the saved-account selection when a token is supplied. `.env` files are not loaded automatically. The connection settings are:

| Variable | Purpose |
| --- | --- |
| `IBM_QUANTUM_ACCOUNT` | Optional saved Runtime account name |
| `IBM_QUANTUM_TOKEN` | Optional token for a new Runtime service connection |
| `IBM_QUANTUM_INSTANCE` | Optional instance CRN |
| `IBM_QUANTUM_BACKEND` | Optional backend override, default `ibm_rensselaer` |

The connection helper checks the required processor family. The paper's current hardware experiments used `ibm_rensselaer` with IBM Nighthawk revision 1. Recheck the live processor identity, coupling graph, calibration, qubit mapping, and resources for a new acquisition. A backend name alone is insufficient to establish that a later job used the same processor.

The distributed configurations contain the original scientific experiment settings, with account selectors set to `null`. Original acquisition records are immutable evidence. Historical campaign commands intentionally reject changed source hashes and existing frozen protocols. Run new experiments in a separate checkout with new output paths and newly frozen protocols. Do not edit original acquisition records or disable their guards to make an old campaign resume. Large fitted states and intermediate state-vector arrays are regenerated rather than distributed. Archived source/input hashes may therefore refer to original files absent from this upload. This does not prevent decoding the supplied measurements or recreating the manuscript plots.

## Interpretation of the measurements

Screening selects a candidate. Independent confirmation evaluates it. Keep those phases separate, including the multiple-testing families used by the original analyses. Two acquisitions on the same FLP42 instance do not count as two independent problem instances. QAOA output compression is distinct from complete annealing compression.

The main evidence is a consistent pattern of higher hardware feasibility relative to uncompressed annealing. Product-state and fixed shallow circuits are substantive controls. FLP42 demonstrates extension of the procedure and depth sensitivity, with gains against full annealing. It does not establish superiority over the product-state or shallow controls. Greater depth and greater ideal fidelity do not consistently produce better hardware samples.

Optimality gaps use the original transportation objective evaluated only for feasible samples, relative to the certified small-instance optimum. Conditional cost quality and the probability of obtaining a feasible candidate describe different aspects of the measured distribution. Equal-budget comparisons use the empirical distributions and do not introduce new hardware observations. Bootstrap uncertainty and mitigation results retain the assumptions described in the corresponding analysis modules and manuscript.

## Citation and repository updates

Author metadata is provided in `CITATION.cff`. Please cite the associated manuscript and the software packages relevant to the part of the workflow you use. This upload adds no new license terms. Retain the existing repository license when integrating it.
