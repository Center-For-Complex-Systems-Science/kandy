---
name: run-experiment
description: Run a KANDy experiment on a dynamical system to discover its governing equations. Use when the user wants to train a KAN model, fit equations, or run the full KANDy pipeline on a system.
argument-hint: [system-name]
---

# Run KANDy Experiment

Run a KANDy experiment on the **$ARGUMENTS** dynamical system.

## Available Systems

!`uv run kandy --list 2>/dev/null`

## Current Results

!`ls results/ 2>/dev/null || echo "No results yet"`

## Instructions

Use the **kandy-researcher** agent to:

1. Run `uv run kandy $ARGUMENTS` (or `uv run python examples/<system>_example.py` if the system name doesn't match exactly)
2. Monitor the output for training loss convergence, discovered equations, and rollout metrics
3. After the experiment completes, use the **model-quality-assessor** agent to evaluate the generated plots in `results/`
4. Document findings in `notes/` with full experiment metadata
5. Report back with:
   - Discovered equations in clean symbolic format
   - Key metrics (training loss, one-step RMSE, rollout RMSE)
   - Quality assessment of the results
   - Any issues or recommendations

If no system name is provided, list available systems and ask the user which one to run.

## Critical Reminders

- KAN width is always `[m, n]` (single layer)
- Cross-terms must be in the lift, not learned by the KAN
- All results go to `results/<SystemName>/`
- Use `kandy.plotting.use_pub_style()` for publication-quality figures
