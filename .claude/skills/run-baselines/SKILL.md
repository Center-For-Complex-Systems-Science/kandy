---
name: run-baselines
description: Run baseline comparisons (OLS, LASSO, SINDy, PDEFind) against KANDy results for a dynamical system. Use when the user wants to compare KANDy's equation discovery against standard methods.
argument-hint: [baseline-name]
---

# Run Baseline Comparisons

Run baseline comparisons for **$ARGUMENTS**.

## Available Baselines

!`uv run kandy-baselines --list 2>/dev/null`

## Instructions

Use the **kandy-researcher** agent to:

1. Run `uv run kandy-baselines $ARGUMENTS` (or the appropriate baseline script)
2. Collect results for each method: discovered equations, number of terms, coefficients, RMSE/NRMSE
3. Compare against existing KANDy results in `results/`
4. Generate a comparison table in the report
5. Save baseline results to `results/<SystemName>/baselines/`
6. Document in `notes/` with full metadata

## Expected Output

A comparison table like:

| Method | Discovered Equation | # Terms | RMSE | NRMSE |
|--------|-------------------|---------|------|-------|
| KANDy  | ...               | ...     | ...  | ...   |
| OLS    | ...               | ...     | ...  | ...   |
| LASSO  | ...               | ...     | ...  | ...   |
| SINDy  | ...               | ...     | ...  | ...   |

If no baseline name is provided, list available baselines and ask the user which one to run.
