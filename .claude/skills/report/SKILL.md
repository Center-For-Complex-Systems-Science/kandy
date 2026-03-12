---
name: report
description: Generate a comprehensive research report for a KANDy experiment, including discovered equations, metrics, plots, and analysis. Use after an experiment is complete.
argument-hint: [system-name]
---

# Generate Research Report

Generate a comprehensive research report for the **$ARGUMENTS** experiment.

## Current Results

!`ls results/ 2>/dev/null`

## Current Notes

!`ls notes/ 2>/dev/null || echo "No notes yet"`

## Instructions

1. **Gather all data** for the system:
   - Read plots from `results/$ARGUMENTS/`
   - Read any existing notes from `notes/`
   - Check example scripts in `examples/` for configuration details
   - Check agent memory for prior findings

2. **Write the report** to `notes/$ARGUMENTS_report.md` with:

   ### System Description
   - Type (ODE/map/PDE), true equations, state variables, parameters

   ### KANDy Configuration
   - Lift design (phi), KAN width, grid, spline order, training steps
   - Derivative method, integrator, any special settings

   ### Discovered Equations
   - Full symbolic equations in clean format
   - Comparison with ground truth (coefficient accuracy, structural match)

   ### Edge Analysis
   - Table: Edge, Feature, Target, Fitted Function, R², Coefficient
   - Which edges are active vs zeroed

   ### Metrics
   - Training loss, one-step RMSE, rollout RMSE/NRMSE
   - Lyapunov time horizon (for chaotic systems)

   ### Baseline Comparisons (if available)
   - Table comparing KANDy vs OLS/LASSO/SINDy/PDEFind

   ### Quality Assessment
   - Use model-quality-assessor to rate the results
   - Key findings and observations

   ### Figures
   - Reference all generated plots with descriptions

3. **Print a summary** to the user with the key findings
