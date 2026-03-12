# Kuramoto-Sivashinsky: KANDy Experiment Report

**Date:** 2026-03-10
**System:** Kuramoto-Sivashinsky PDE (chaotic regime, L=22)

## True Equation

    u_t + u*u_x + u_xx + u_xxxx = 0

equivalently:

    u_t = -u*u_x - u_xx - u_xxxx

## Discovered Equation

    u_t = -0.9966*u*u_x - 0.9959*u_xx - 0.9965*u_xxxx

All three coefficients within **0.4%** of the true values. **3 active terms, 0 spurious terms.**

## Setup

| Parameter | Value |
|---|---|
| Domain | [0, 22], periodic |
| Grid | N_x = 64 spatial points |
| Time step | dt = 0.25 |
| Total steps | 8000 |
| Burn-in | 1000 (discard transient) |
| Solver | ETDRK4 (Kassam & Trefethen, 2005) |
| Derivatives | Spectral (exact on periodic domains) |

### KAN Configuration

| Parameter | Value |
|---|---|
| Width | [6, 1] |
| Grid | 7 |
| Spline order k | 3 |
| Training | 100 LBFGS steps |
| Subsampling | 100K from ~448K total samples |
| Regularization | lamb = 0.0 |
| Patience | 0 |

### Koopman Lift (6 features)

| Feature | Description |
|---|---|
| u | Field value |
| u_x | First spatial derivative (spectral) |
| u_xx | Second spatial derivative (spectral) |
| u_xxxx | Fourth spatial derivative (spectral) |
| u*u_x | Nonlinear advection term |
| u*u_xx | Cross-term (included for completeness) |

## Results

### Metrics

| Metric | Value |
|---|---|
| Train loss | 1.1e-5 |
| Pointwise RMSE | 2.3e-3 |
| Rollout RMSE (200 steps) | 0.063 |

### Edge-by-Edge Symbolic Extraction

| Edge | Feature | Symbolic Fit | R^2 |
|---|---|---|---|
| (0,0,0) | u | 0 (zeroed) | -- |
| (0,1,0) | u_x | 0 (zeroed) | -- |
| (0,2,0) | u_xx | x (linear) | 0.999999 |
| (0,3,0) | u_xxxx | x (linear) | 0.999999 |
| (0,4,0) | u*u_x | x (linear) | 0.999989 |
| (0,5,0) | u*u_xx | 0 (zeroed) | -- |

**3/6 edges active (all linear, R^2 > 0.9999), 3/6 zeroed -- perfect sparsity.**

### Discovered Coefficients

| Term | True | Discovered | Error |
|---|---|---|---|
| u*u_x | -1.0000 | -0.9966 | 0.34% |
| u_xx | -1.0000 | -0.9959 | 0.41% |
| u_xxxx | -1.0000 | -0.9965 | 0.35% |

### Plots

- `spacetime.png` -- True vs KANDy space-time heatmaps (200-step rollout)
- `edge_activations.png` -- All 6 edge activations (3 linear, 3 zeroed)
- `loss_curves.png` -- Train/test convergence

## Key Technical Decisions

### Spectral Derivatives (Critical)

The KS equation is solved on a periodic domain, making spectral derivatives exact:
- u_x = IFFT(ik * u_hat)
- u_xx = IFFT(-k^2 * u_hat)
- u_xxxx = IFFT(k^4 * u_hat)

Finite differences introduced ~8% relative error on u_xxxx, biasing the learned coefficients.

### Minimal Feature Library (Critical)

**The original 12-feature library** included u_xx AND u_xx^2 (and other squared terms). This created a **library degeneracy** where the KAN split the u_xx signal across two edges:

- Edge for u_xx: learned psi ~ -0.288*u_xx^2 - 1.02*u_xx (quadratic!)
- Edge for u_xx^2: learned psi ~ +0.267*u_xx^2 (linear on squared feature)
- Net: ~+0.004*u_xx^2 - 1.02*u_xx (quadratic terms nearly cancel)

`auto_symbolic` then picked x^2 for the u_xx edge (R^2_quad = 0.999 >> R^2_lin = 0.869) because the spline itself was genuinely quadratic.

**The fix:** Remove all squared features. No feature should be a deterministic function of another feature in the library. Cross-terms (u*u_x, u*u_xx) are fine because they depend on two independent fields.

### ETDRK4 Solver

The Kassam-Trefethen ETDRK4 scheme handles the stiff linear part (u_xx + u_xxxx) exactly via exponential integrators. This is essential because:
- Forward Euler is unstable at dt=0.25
- The linear eigenvalues L_hat = k^2 - k^4 have large negative values for high wavenumbers
- ETDRK4 precomputes coefficients via contour integrals (M=32 points) to avoid cancellation

### IMEX Rollout

The learned model is integrated using the same ETDRK4 framework:
- Stiff linear part: handled exactly by precomputed ETDRK4 coefficients
- Nonlinear part: N_learned(u) = KANDy(phi(u)) - linear(u)
- This decomposition keeps the rollout stable at dt=0.25

### OOM Fix for Symbolic Extraction

`auto_symbolic()` calls a forward pass internally. On the full training set (100K samples) this causes an out-of-memory error. Fix: use a 2048-sample subset for the `save_act` forward pass before calling `auto_symbolic()`.

## Comparison: Before and After Fix

| Metric | Old (12-feat, FD) | New (6-feat, spectral) |
|---|---|---|
| Pointwise RMSE | 5.0e-3 | **2.3e-3** |
| Rollout RMSE (200 steps) | 0.408 | **0.063** |
| Active edges | 5/12 (2 spurious) | **3/6 (0 spurious)** |
| u_xx edge fit | x^2 (wrong!) | **x (correct)** |
| Coefficients | ~3% error | **~0.4% error** |

## Conclusion

KANDy successfully discovers the Kuramoto-Sivashinsky equation in closed form with sub-percent coefficient accuracy and perfect sparsity. The key ingredients are spectral derivatives (exact on the periodic domain) and a minimal feature library that avoids degeneracy from correlated features. The learned model also produces stable, accurate long-time rollouts via ETDRK4 integration.

**Rating: Excellent**
