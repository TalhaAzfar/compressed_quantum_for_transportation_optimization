# Release validation

The full code-and-data release was checked on 27 October 2026. Its original records are now distributed separately in `hardware-aware-transport-qaoa-data.zip`. The checks below describe that full release. No quantum jobs were submitted.

- **104 numerical and workflow tests passed.** One original-campaign hash test is opt-in through the `campaign` marker. It requires the complete original workspace and its unmodified source hashes. The public copy instead records its portability edits in `SOURCE_PROVENANCE.json`.
- All 4,251 compressed experiment files passed compressed-byte and original-byte SHA-256 checks.
- All 1,378 included experimental hardware distributions were decoded again, covering 7,839,744 shots across the archived acquisition phases. Twelve readout-calibration distributions also passed their shot-total checks. These totals describe the distributed archive, rather than a new pooled paper experiment. Screening and confirmation remain separate.
- The feasible-candidate analysis reproduced all 272 confirmation rows across 34 cases and 136 method/case combinations. It decoded 2,419,530 distinct recorded bitstrings and independently checked 3,247 feasible bitstrings. The reconstructed analysis is identical to the manuscript's compact cost-analysis input.
- Five plotting programs produced 15 PDF/PNG artifacts. Sampling-budget diagnostics were recomputed from the compact data.
- The offline six-qubit FLP example completed. Its MPS objective agreed with exact-state fidelity. The example reported fidelity 0.96175 and CZ depth 8 for the compiled circuit, compared with depth 126 for its source on the toy chain. These are ideal example results, not new hardware measurements.
- Five mocked checks covered default saved-account selection, named accounts, token/instance configuration, instance lookup, and processor-family rejection. No IBM service was contacted.
- Runnable source and configurations contain no author-specific environment paths or IBM account names. A scan of distributed and expanded text records found no credential values or private-key files.

The numerical suite sets double precision for JAX, as the original experiment runner does. Its exact-state gradient comparison uses an untruncated MPS so that truncation and floating-point precision are not confused with gradient error. These changes affect verification, rather than the recorded experiments or fitting algorithms.

The checked numerical environment is listed in `validation/environment.json`. Existing local environments supplied the dependencies. A new dependency installation and new hardware execution were not performed as part of release validation. The local environment emitted upstream Qiskit deprecation/plugin messages and a PyTorch tensor-copy warning. They did not cause test failures.

The separately distributed compressed records occupy approximately 98 MiB and expand to approximately 1.6 GiB. Large fitted-state arrays and MPS-state pickles are excluded. Original backend-target snapshots are retained for circuit-validation tests. The compact validation records are under `validation/`. Regenerated figures and expanded data are intentionally omitted from the upload folder and can be recreated using the README commands.

## GitHub credential check

Before refreshing the upload, the source, configurations, documentation, binary files, and all 4,251 compressed experiment records were scanned for cloud resource names, literal credentials, recognized token formats, private keys, credential-store files, and unexplained credential-like strings. No API keys, instance CRNs, or private-key material were found. Compressed records were scanned after decompression, including binary bytes without loading pickles. Original measurement records were not modified. The machine-readable report is `validation/credential_audit.json`.

The new `scripts/configure_ibm_account.py` was checked with a mocked Qiskit SDK. Checks covered hidden prompts, whitespace trimming, account settings, no credential values in output, rejection of empty inputs, and rejection of an invalid CRN. No actual account was saved and no IBM service was contacted. The helper saves user-supplied credentials only when explicitly run locally.

The README now describes the local Qiskit credential store and environment-variable alternative. Credential files and private settings remain excluded by `.gitignore`. The scientific code and recorded measurements are unchanged by this upload refresh.

## Lightweight code upload

The GitHub code upload excludes the bulk `recorded_data/` tree. All compact figure inputs, source, configurations, examples, tests, and credential-setup instructions are retained. `ORIGINAL_DATA.json` identifies the separate archive and its checksum. The plotting entry point works without the archive and explicitly reports that raw measurements were not reverified in that mode. Full bitstring audits and tests requiring hardware target snapshots need the optional archive. Scientific algorithms and original measurement files are unchanged.

The lightweight plotting command was run without `recorded_data/` and produced all 15 figure artifacts. Runtime, budget-quality, and block-variability diagnostics matched the bundled data within floating-point tolerance. Its report correctly records `archive_verified: false`. The lightweight upload also passed a fresh credential scan. See `validation/lightweight_reproduction.json` and `validation/lightweight_credential_audit.json`.
