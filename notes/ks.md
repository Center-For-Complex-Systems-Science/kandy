# Kuramoto-Sivashinsky Experiment Details

## Final Results (2026-03-10)

### Discovered Equation
```
u_t = -0.9966*u*u_x - 0.9959*u_xx - 0.9965*u_xxxx
TRUE: u_t = -u*u_x - u_xx - u_xxxx
```
Coefficients within 0.4% of true values. 3 terms, no spurious terms.

### Metrics
| Metric | Old (12-feat, FD) | New (6-feat, spectral) |
|---|---|---|
| Pointwise RMSE | 5.0e-3 | **2.3e-3** |
| Rollout RMSE (200 steps) | 0.408 | **0.063** |
| Active edges | 5/12 (2 spurious) | **3/6 (perfect)** |
| u_xx edge fit | x² (wrong!) | **x (correct)** |
| Coefficients | ~3% error | **~0.4% error** |

### Edge-by-Edge auto_symbolic (final)
| Edge | Feature | Fit | R² |
|---|---|---|---|
| (0,0,0) | u | 0 (zeroed) | — |
| (0,1,0) | u_x | 0 (zeroed) | — |
| (0,2,0) | u_xx | x (linear) | 0.999999 |
| (0,3,0) | u_xxxx | x (linear) | 0.999999 |
| (0,4,0) | u*u_x | x (linear) | 0.999989 |
| (0,5,0) | u*u_xx | 0 (zeroed) | — |

## Setup
- **PDE:** u_t + u*u_x + u_xx + u_xxxx = 0 (chaotic regime, L=22)
- **Grid:** N_x=64, dt=0.25, 8000 steps, BURN=1000
- **Derivatives:** Spectral (exact on periodic domain)
- **Lift:** 6 features: [u, u_x, u_xx, u_xxxx, u*u_x, u*u_xx]
- **KAN:** width=[6, 1], grid=7, k=3, 100 LBFGS steps, patience=0
- **Subsampling:** 100K from ~448K total samples

## Critical Bug Fix: Library Degeneracy

### The Problem
Original 12-feature library included u_xx AND u_xx² (and similar squared terms).
The KAN exploited this degeneracy by splitting the u_xx signal:
- Edge 2 (u_xx): learned ψ ≈ -0.288*u_xx² - 1.02*u_xx (quadratic!)
- Edge 7 (u_xx²): learned ψ ≈ +0.267*u_xx² (linear on squared feature)
- Net contribution: ≈ +0.004*u_xx² - 1.02*u_xx (quadratic terms nearly cancel)

auto_symbolic then picked x² for the u_xx edge (R²_quad=0.999 >> R²_lin=0.869).

### The Diagnosis
- NOT caused by FD derivative errors (spectral features showed same problem)
- NOT fixable by auto_symbolic complexity penalty (the spline IS genuinely quadratic)
- Caused by **optimization degeneracy**: with correlated features and no sparsity
  penalty, LBFGS finds a low-loss solution that spreads signal across edges

### The Fix
Remove all squared features. New 6-feature library:
[u, u_x, u_xx, u_xxxx, u*u_x, u*u_xx]

**Rule:** Include cross-terms (u*u_x, u*u_xx) but NOT independent powers of
existing features (u_xx², u_x², u², u³). No feature should be a deterministic
function of another feature in the library.

### L1 Regularization (partial fix)
- lamb=0.001 with 12-feature lib: u_xx R²_lin improved to 0.994 (from 0.867)
  but coefficients attenuated to ~0.95 and all 12 edges still non-zero
- lamb=0.01+: too aggressive, kills all signal
- Reduced library is the better fix (no coefficient attenuation, perfect sparsity)

## Technical Details

### ETDRK4 Solver (Kassam & Trefethen 2005)
- Original IMEX Euler blew up at dt=0.25 (NaN at step 394)
- ETDRK4 is rock-solid at dt=0.25
- Precomputes coefficients via contour integrals (M=32 points)
- Linear eigenvalues: L_hat = k² - k⁴
- Returns both real snapshots and Fourier coefficients (for spectral derivatives)

### Spectral Derivatives
- u_x = IFFT(ik * u_hat), u_xx = IFFT(-k² * u_hat), u_xxxx = IFFT(k⁴ * u_hat)
- Exact on periodic domains (vs FD: 8% relative error on u_xxxx)
- Require storing Fourier coefficients alongside real snapshots

### IMEX Rollout
- Forward Euler unstable for stiff KS at dt=0.25
- ETDRK4 with learned model: stiff linear part handled exactly
- learned_nl(v_hat) = FFT(model.predict(phi(v)) - (-v_xx - v_xxxx))

### OOM Fix for Symbolic Extraction
- auto_symbolic() does forward pass; OOM on full training set
- Fix: 2048-sample subset for save_act forward pass before auto_symbolic()

## Plots
- `results/KS/spacetime.png` — True vs KANDy space-time heatmaps (excellent match)
- `results/KS/edge_activations.png` — All 6 edges (3 linear, 3 zeroed)
- `results/KS/loss_curves.png` — Train/test convergence (no overfitting)
