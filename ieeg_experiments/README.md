# iEEG Experiments

This directory is a curated copy of the code-centered parts of `ieeg_research/`.
The goal is to give future coding agents a clean starting point without making
them sort through plots, checkpoints, duplicate outputs, and abandoned branches.

## Status Legend

- `successful`: strong result or current best-known direction
- `mixed`: useful code path, but exploratory or not clearly dominant
- `failed`: important negative result to avoid repeating
- `supporting-analysis`: context, clustering, validation, or plotting support
- `archived`: duplicate, generic reference, or superseded variant

## Start Here

1. `00_overview/ieeg_lessons.md`
   This is the highest-signal summary of what worked, what failed, and why.
2. `01_mode0_duffing/ieeg_example.py`
   Best structural-ID path for the E3Data seizure data. The lessons file treats
   the v6/v7 Duffing-ReLU workflow as the main success track.
3. `01_mode0_duffing/ieeg_relu_gated.py`
   Extension of the Duffing idea with more ReLU-gated terms and checkpointing.
4. `03_clustering_and_supporting_analysis/`
   Differential embedding, order-parameter analysis, directed connectivity, and
   early-warning analyses that feed the Kuramoto-family modeling.
5. `04_cluster_kuramoto_models/` and `05_multiband_reconstruction/`
   Cluster-reduced and multiband follow-on models. These are more fragmented
   than the Duffing workflow, but they contain most of the later iEEG modeling.

## Folder Map

- `00_overview`: distilled lessons and experiment-level takeaways.
- `01_mode0_duffing`: strongest KANDy discovery scripts on E3Data.
- `02_simple_kuramoto_virtual_surgery`: 2-node SOZ vs NSOZ proof-of-concept.
- `03_clustering_and_supporting_analysis`: preprocessing, clustering, causality,
  early-warning, utilities, notebook, and run log.
- `04_cluster_kuramoto_models`: cluster-reduced KANDy models and comparison tools.
- `05_multiband_reconstruction`: multiband Kuramoto fitting and synthetic seizure
  reconstruction variants (`reconstruct_cfc*.py` are iterative generations).
- `99_archived_or_reference`: duplicate or generic references kept only for context.

## Important Caveats

- The curated copies now resolve data and `src/` relative to the repository root.
- `E3Data.mat` is expected at `data/E3Data.mat`; QM text data is expected under `data/QM/`.
- Most scripts save figures beside themselves or under `results/iEEG/...`.
- This reorganization copies code only. Original result plots and checkpoints stay in
  `ieeg_research/`.

Use `EXPERIMENT_INDEX.md` for file-by-file status and triage.
