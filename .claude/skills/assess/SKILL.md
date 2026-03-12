---
name: assess
description: Assess the quality of KANDy model results by examining generated plots. Use after training a model, running rollouts, or generating comparison plots.
argument-hint: [system-name]
---

# Assess Model Quality

Assess the quality of results for **$ARGUMENTS**.

## Current Results

!`ls results/ 2>/dev/null || echo "No results yet"`

## Instructions

Use the **model-quality-assessor** agent to:

1. Find all plots in `results/$ARGUMENTS/` (or search `results/` if the exact directory name differs)
2. Read and visually analyze each plot (PNG files)
3. Apply system-specific assessment criteria:
   - **ODEs** (Lorenz, Hopf): trajectory tracking, attractor geometry, divergence time
   - **PDEs** (Burgers, KS): space-time rollout accuracy, shock locations, spectral content
   - **Maps** (Henon, Ikeda): attractor structure, iterate accuracy
   - **Coupled** (Kuramoto): order parameter tracking, phase synchronization
4. Check for common failure modes (blowup, phase drift, amplitude mismatch, overfitting)
5. Rate overall quality: Excellent / Good / Fair / Poor
6. Provide specific recommendations for improvement if needed

## Output Format

- Plots reviewed (list each file)
- Per-plot analysis
- System-specific checks (pass/concern/fail)
- Overall rating with justification
- Actionable recommendations
