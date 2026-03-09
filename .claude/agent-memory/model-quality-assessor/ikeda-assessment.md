# Ikeda Map Assessment Summary

**Date:** 2026-03-08
**Rating:** Excellent
**Script:** examples/ikeda_example.py
**Results:** results/Ikeda/

## Configuration
- Lift: Physics-informed [u*x*cos(t), u*y*cos(t), u*x*sin(t), u*y*sin(t)]
- KAN: width=[4,2], grid=5, k=3, base_fun=RBF
- Training: Phase 1 (one-step, 200 LBFGS steps) + Phase 2 (rollout fine-tune, 100 steps, rollout_weight=0.2, horizon=15)
- Integrator: Euler with increment trick (dt=1)
- Symbolic: TRIG_LIB_CHEAP, R2 threshold 0.80

## Key Metrics
- One-step train loss: ~4e-8
- Rollout loss: ~2-4e-2
- Rollout RMSE (400 steps): 0.88
- Coefficient accuracy: 4+ significant figures
- Train/test gap: negligible (no overfitting)

## Discovered Equations
- x_{n+1} = 0.9*x*cos(t) + 0.9*y*sin(-t) + 1.0001 (true: 1 + 0.9*(x*cos(t) - y*sin(t)))
- y_{n+1} = 0.9*x*sin(t) + 0.9*y*cos(t) (exact match)

## Edge Sparsity Pattern (4x2 grid)
Active (linear, ~O(1)): (0,0,0), (0,3,0), (0,1,1), (0,2,1)
Inactive (~O(1e-5)): (0,1,0), (0,2,0), (0,0,1), (0,3,1)
Matches true equation structure exactly.
