# Experiment Index

This index is a triage map for future agents. Status labels are based on
`ieeg_lessons.md`, script docstrings, naming, and surrounding outputs. When the
evidence is indirect, treat the label as an informed inference rather than a
formal benchmark result.

| Path | Status | Role | Notes |
|---|---|---|---|
| `00_overview/ieeg_lessons.md` | successful | project memory | Main summary of v1-v7 findings, failure modes, and recommended settings. |
| `01_mode0_duffing/ieeg_example.py` | successful | primary discovery script | Best-known Mode-0 Duffing-ReLU workflow; ties directly to v6/v7 lessons. |
| `01_mode0_duffing/ieeg_relu_gated.py` | mixed | extended discovery script | Richer gated Duffing model with checkpointing; more complex and likely more brittle. |
| `02_simple_kuramoto_virtual_surgery/ieeg_kuramoto_virtual_surgery.py` | mixed | proof of concept | 2-node SOZ/NSOZ model for virtual surgery experiments. |
| `03_clustering_and_supporting_analysis/differential_embedding_clustering.py` | supporting-analysis | clustering foundation | Derives seizure-window channel clusters used by later Kuramoto models. |
| `03_clustering_and_supporting_analysis/differential_embedding_advanced.py` | supporting-analysis | clustering validation | Adds ARI, permutation tests, and time-resolved analyses. |
| `03_clustering_and_supporting_analysis/order_parameter_analysis.py` | supporting-analysis | synchrony analysis | Quantifies within/between-cluster order parameters. |
| `03_clustering_and_supporting_analysis/directed_connectivity_analysis.py` | supporting-analysis | causality analysis | oCSE-based directed flow analysis across SOZ subclusters. |
| `03_clustering_and_supporting_analysis/early_warning_analysis.py` | supporting-analysis | bifurcation analysis | Early-warning signals plus Stuart-Landau fits near seizure onset. |
| `03_clustering_and_supporting_analysis/ieeg_utils.py` | supporting-analysis | shared utilities | Common loaders, bandpass helpers, constants, and plotting utilities. |
| `03_clustering_and_supporting_analysis/run_log.txt` | supporting-analysis | execution notes | Historical run output for the Kuramoto analysis branch. |
| `03_clustering_and_supporting_analysis/Higher Kuramoto.ipynb` | mixed | exploratory notebook | Notebook-level exploration; not a clean automated entrypoint. |
| `04_cluster_kuramoto_models/kandy_cluster_kuramoto.py` | mixed | early cluster model | 4-oscillator E3Data reduction; precursor to later QM scripts. |
| `04_cluster_kuramoto_models/cluster_kuramoto_QM.py` | mixed | main QM cluster model | PatientQM alpha-band 3-cluster model; likely the cleanest QM fit in this family. |
| `04_cluster_kuramoto_models/cluster_kuramoto_QM_broadband.py` | mixed | ablation variant | Broadband counterpart to the alpha-band QM model. |
| `04_cluster_kuramoto_models/cluster_kuramoto_QM_lowfreq.py` | mixed | ablation variant | Low-frequency counterpart to the alpha-band QM model. |
| `04_cluster_kuramoto_models/envelope_phase_kuramoto.py` | mixed | slow-phase model | Envelope-of-envelope phase dynamics; useful but not highlighted as top-performing. |
| `04_cluster_kuramoto_models/global_svd_kandy.py` | mixed | alternative representation | Global SVD instead of cluster-first representation. |
| `04_cluster_kuramoto_models/kandy_3osc_multifreq.py` | mixed | amplitude cluster model | Explicitly framed as replacing failed phase-only approaches. |
| `04_cluster_kuramoto_models/rollout_all_models.py` | supporting-analysis | comparison harness | Rolls out several model families for side-by-side trajectory checks. |
| `04_cluster_kuramoto_models/extract_all_equations.py` | supporting-analysis | equation extraction | Re-trains models to produce symbolic equations and plots. |
| `05_multiband_reconstruction/multiband_kuramoto.py` | mixed | base multiband fitter | Fits cluster-reduced Kuramoto models band by band. |
| `05_multiband_reconstruction/multiband_kuramoto_v2.py` | mixed | richer multiband fitter | Later feature-expanded version; likely supersedes the base multiband script. |
| `05_multiband_reconstruction/parameter_sweep.py` | supporting-analysis | simulation sweep | Sweeps model parameters for the multiband reconstruction branch. |
| `05_multiband_reconstruction/reconstruct_ieeg.py` | mixed | data-guided reconstruction | Uses model phases plus data-derived amplitudes and offsets. |
| `05_multiband_reconstruction/reconstruct_model_only.py` | mixed | fully synthetic reconstruction | Uses static data summaries only; generates signal dynamics from the model. |
| `05_multiband_reconstruction/reconstruct_cfc.py` | mixed | iterative reconstruction v1 | First cross-frequency-coupled synthetic seizure generator. |
| `05_multiband_reconstruction/reconstruct_cfc_v2.py` | mixed | iterative reconstruction v2 | Refines v1 with more realistic onset behavior. |
| `05_multiband_reconstruction/reconstruct_cfc_v3.py` | mixed | iterative reconstruction v3 | Adds SOZ-led onset and OU amplitude fluctuations. |
| `05_multiband_reconstruction/reconstruct_cfc_v4.py` | mixed | iterative reconstruction v4 | Another v3-style iteration; keep for diffs, not as sole truth. |
| `05_multiband_reconstruction/reconstruct_cfc_v5.py` | mixed | iterative reconstruction v5 | Adds DC shift and pink-noise background. |
| `05_multiband_reconstruction/reconstruct_cfc_v6.py` | mixed | latest reconstruction variant | Latest numbered CFC generator in-tree; likely best handoff point for this branch. |
| `05_multiband_reconstruction/virtual_stimulation.py` | mixed | intervention simulation | Applies stimulation controls to the multiband model. |
| `05_multiband_reconstruction/model_schematic.py` | supporting-analysis | communication artifact | Generates the publication-style model schematic. |
| `99_archived_or_reference/kandy_cluster_kuramoto (copy).py` | archived | duplicate | Literal copy of the 4-oscillator cluster script. |
| `99_archived_or_reference/kuramoto_example.py` | archived | generic reference | Synthetic non-iEEG Kuramoto example from the broader repo. |
| `99_archived_or_reference/kuramoto_sivashinsky_example.py` | archived | generic reference | Synthetic PDE example; not part of the iEEG program. |

## Practical Triage

- If the task is equation discovery on real seizure data, start with `01_mode0_duffing/`.
- If the task is channel grouping or SOZ structure, start with `03_clustering_and_supporting_analysis/`.
- If the task is cluster-level Kuramoto fitting on PatientQM, start with `04_cluster_kuramoto_models/cluster_kuramoto_QM.py`.
- If the task is synthetic seizure generation or intervention, start with `05_multiband_reconstruction/reconstruct_cfc_v6.py` and `virtual_stimulation.py`.

## Portability Risks

- `E3Data.mat` is expected at `data/E3Data.mat`; QM text data is expected under `data/QM/`.
- The curated scripts now resolve `data/` and `src/` relative to the repository root.
- Several branches depend on `kandy` being importable either from the installed package or a manually injected `src` path.
