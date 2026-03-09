# Model Quality Assessor Memory

## Assessed Systems
- [Ikeda Map](ikeda-assessment.md) - Rating: Excellent

## Key Patterns

### Physics-Informed Lifts Are Critical for Discrete Maps
- Ikeda: precomputing [u*x*cos(t), u*y*cos(t), u*x*sin(t), u*y*sin(t)] with state-dependent phase reduces KAN to linear mixing
- When lift absorbs all nonlinearity, edge activations should be linear for active edges and ~zero for inactive edges
- Active/inactive edge amplitude ratio of 10^5-10^6 indicates clean sparsity recovery

### Expected Loss Ranges (Float32)
- One-step loss ~4e-8 is near float32 precision limit -- indicates essentially perfect fit
- 5-order-of-magnitude gap between one-step (~1e-7) and rollout (~1e-2) loss is normal for chaotic maps

### Chaotic Map Rollout Assessment
- Do NOT judge chaotic systems by point-wise RMSE over long horizons
- Instead check: attractor geometry, amplitude envelope, statistical properties, no blowup/collapse
- Short-term tracking (30-50 steps for Ikeda) + long-term attractor fidelity = success
- Rollout RMSE ~0.88 over 400 steps of chaotic Ikeda is expected/good

### Edge Activation Interpretation
- Linear edges = the lift captured the nonlinearity correctly
- Near-zero edges (1e-5 to 1e-6 scale) = correct sparsity, those features not needed for that output
- Slight nonlinear residual on zero edges is spline artifact, not model failure

### Increment Trick for Discrete Maps
- discrete_rhs(s) = map(s) - s, then Euler with dt=1 recovers exact map iteration
- Allows reuse of ODE rollout training infrastructure for discrete maps
