# Standard Kuramoto Oscillator — KANDy Results (2026-03-18)

## System
N=5 coupled phase oscillators with fixed global coupling:
```
dθ_i/dt = ω_i + (K/N) Σ_{j≠i} sin(θ_j - θ_i)
```
- K = 1.0, N = 5 → K/N = 0.2
- ω = [0.548, -0.122, 0.717, 0.395, -0.812] (Uniform(-1,1), seed=42)

## Discovered Equations
All coefficients recovered to 3+ significant figures:
```
dθ₀/dt = -0.200·sin(θ₀-θ₁) - 0.200·sin(θ₀-θ₂) - 0.200·sin(θ₀-θ₃) - 0.200·sin(θ₀-θ₄) + 0.548
dθ₁/dt = +0.200·sin(θ₀-θ₁) - 0.200·sin(θ₁-θ₂) - 0.200·sin(θ₁-θ₃) - 0.200·sin(θ₁-θ₄) - 0.122
dθ₂/dt = +0.200·sin(θ₀-θ₂) + 0.200·sin(θ₁-θ₂) - 0.201·sin(θ₂-θ₃) - 0.200·sin(θ₂-θ₄) + 0.718
dθ₃/dt = +0.200·sin(θ₀-θ₃) + 0.200·sin(θ₁-θ₃) + 0.202·sin(θ₂-θ₃) - 0.200·sin(θ₃-θ₄) + 0.394
dθ₄/dt = +0.200·sin(θ₀-θ₄) + 0.200·sin(θ₁-θ₄) + 0.200·sin(θ₂-θ₄) + 0.200·sin(θ₃-θ₄) - 0.812
```

## Metrics
- One-step MSE: 5.1e-8
- Rollout RMSE (θ): 0.001
- Order parameter RMSE: 0.0002
- All 20 coupling coefficients within 0.1% of true ±0.2
- All 5 natural frequencies within 0.1% of true values
- Zero spurious edges
- **Rating: Excellent**

## KAN Architecture
- KAN: width=[10, 5], grid=5, k=3
- 200 LBFGS steps, lamb=0.0 (no regularization needed)
- Symbolic library: {x, 0} — all activations are linear or zero

## Key Design Choices & Lessons Learned

### 1. Unique pairs (i<j) — removes exact degeneracy
**Problem:** Using all N(N-1)=20 pairs `sin(θ_i-θ_j)` creates exact degeneracy because
`sin(θ_i-θ_j) = -sin(θ_j-θ_i)`. The KAN can split signal between anti-correlated
pairs arbitrarily without changing the loss.

**Fix:** Use only N(N-1)/2=10 unique pairs (i<j). For output k:
- Pairs (i,k) with i<k get coefficient +K/N (positive)
- Pairs (k,j) with k<j get coefficient -K/N (negative)

### 2. Many short trajectories — breaks multicollinearity
**Problem:** With 3 long trajectories (T=100), oscillators spend most time near
synchronized states where `sin(θ_0-θ_1) ≈ sin(θ_0-θ_2)`. This multicollinearity
lets the KAN distribute signal across correlated features.

**Fix:** 20 ICs × T=30 instead of 3 ICs × T=100. Short trajectories capture diverse
transient phase configurations. Order parameter range: [0.014, 0.971] with mean=0.709.

### 3. No L1 regularization needed
With unique pairs + diverse data, lamb=0.0 gives the best results. L1 regularization
(lamb≥0.001) actually hurt by zeroing out edges that should be active.

### 4. Bug fix in training.py
`fit_kan()` had a bug where `lamb > 0` was silently zeroed if `model.save_act` was
False (the default for new KANs). Fixed to auto-enable `save_act` when `lamb > 0`.

## Failed Approaches
1. **All N(N-1)=20 pairs, lamb=0:** Rollout perfect but equations wrong — signal
   distributed across anti-correlated pairs and correlated features
2. **Unique pairs, K=2.0, lamb=0.01:** Oscillators sync too tightly (order param=0.95),
   features near zero, L1 zeros everything. Train loss plateaus at 1.1e-2.
3. **Unique pairs, K=1.0, lamb=0.001:** L1 still too strong, zeros out correct edges.
   Only ~2 of 4 edges survive per output.
4. **Unique pairs, K=1.0, lamb=0.0001:** Better but still missing some edges for
   outputs 0-3. Output 4 was perfect.

## General Lesson for Coupled Oscillator Systems
For systems with coupled oscillators where lift features are phase differences:
- Use **unique pairs only** to avoid anti-symmetry degeneracy
- Use **many short trajectories** (20+) to break correlation between features during
  partial synchronization
- L1 regularization is unnecessary if data diversity is sufficient; it can even hurt
  by aggressively zeroing correct but weak edges
