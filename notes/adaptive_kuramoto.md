# Adaptive Kuramoto-Sakaguchi Experiment Details

## Results (2026-03-10)

### Rollout Metrics (Excellent)
| Metric | Value |
|---|---|
| Rollout RMSE θ | 0.035 |
| Rollout RMSE κ | 0.008 |
| Order parameter RMSE | 0.0003 |

Phase trajectories: near-perfect overlap for all 5 oscillators over 25 time units.
Order parameter: virtually indistinguishable from ground truth (r≈0.21).
Coupling weights: good match until t≈20, then divergence (some κ drift by ~0.01-0.1).

### Symbolic Extraction (Failed)
Both theta and kappa models produce extremely messy equations:
- **θ model (50 edges):** Each dθ_i/dt has ~30-40 terms (spurious cos, exp, nested sin)
- **κ model (1200 edges):** Each dκ_ij/dt has ~40-50 terms (spurious sqrt, exp, nested compositions)

True structure NOT recovered cleanly. The model is a good black-box predictor but
symbolic extraction fails due to library degeneracy (same root cause as KS u_xx² bug).

### Root Cause: Feature Correlation
The lift includes sin(θ_i-θ_j), cos(θ_i-θ_j), AND κ_ij — all correlated in training data.
With N=5 oscillators: theta model has 10 inputs, kappa model has 60 inputs. Far more
edges than the true equation needs, giving optimizer room for degenerate solutions.

## Setup
- **System:** Adaptive Kuramoto-Sakaguchi with N=5 oscillators
- **True equations:**
  - dθ_i/dt = ω_i + Σ_j κ_ij · sin(θ_i - θ_j - α), α = -π/4
  - dκ_ij/dt = -ε · [κ_ij + sin(θ_i - θ_j - β)], β = -π/2, ε = 0.1
- **Two-model approach:**
  - θ model: KAN([10, 5]), lift = [sin(θ_i-θ_j) for all j≠i, κ_ij for all j≠i]
  - κ model: KAN([60, 20]), lift = [sin(θ_i-θ_j), cos(θ_i-θ_j), κ_ij for all pairs]
- **Training:** 200 LBFGS steps, base_fun=sin, grid=5, k=3
- **Rollout:** RK4, 500 steps

## Bug Fixes
### CUDA/CPU Device Mismatch
- `KANDy.__init__` auto-detects CUDA; sets self.device="cuda" if available
- `fit()` overrides to CPU, but if model_ is manually assigned (not via fit()),
  device stays CUDA → `predict()` creates CUDA tensors but model grids are on CPU
- **Fix:** Pass `device="cpu"` to KANDy constructor when using manual model assignment

## Plots
- `results/Kuramoto/phase_trajectories.png` — 5 subplots, true vs KANDy (excellent)
- `results/Kuramoto/order_parameter.png` — r(t) true vs predicted (RMSE=0.0003)
- `results/Kuramoto/kappa_trajectories.png` — 4 coupling weights (diverge after t≈20)
- `results/Kuramoto/loss_phase.png` — Train/test + rollout loss (converges to ~10⁻⁸)

## Potential Improvements
- L1 regularization to encourage sparsity (but may attenuate coefficients)
- Pruning: remove low-magnitude edges before symbolic extraction
- Redesign lift to avoid correlated features (e.g., only sin(θ_i-θ_j-α) directly)
- Increase training data or reduce number of oscillators to reduce degeneracy
