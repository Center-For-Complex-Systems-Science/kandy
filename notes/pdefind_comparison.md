# PDE-FIND vs KANDy Baseline Comparison — Inviscid Burgers (2026-03-17)

## Summary

Head-to-head comparison of KANDy vs PDE-FIND (PySINDy PDELibrary) on inviscid Burgers
equation with random Fourier-mode initial conditions. **KANDy wins decisively**: sparsest
equation, best rollout, correct structural identification. PDE-FIND fails entirely on
Rusanov-generated shock data.

## Setup

Matched to `research_code/burgers_fourier_baselines.py`:

| Parameter | Value |
|-----------|-------|
| Grid | Nx=128, `np.linspace(-π, π, 128, endpoint=False)` |
| IC | K=10 Fourier modes, seed=0, σ_k = k^{-1.5} |
| Domain | [-π, π], t ∈ [0, 2], dt=0.004 |
| Solver | First-order Rusanov flux + RK45 (scipy, rtol=1e-6, atol=1e-8) |
| KANDy derivatives | TVD minmod (spatial), forward difference (temporal) |
| KANDy features | [u, u_x, u*u_x, u_xx] → KAN [4, 1], grid=7, k=3, 300 steps |
| PDE-FIND | PySINDy PDELibrary, PolynomialLibrary(degree=2), derivative_order=3, STLSQ |

## Results

| Method | Active Terms | NRMSE | Discovered Equation |
|--------|-------------|-------|---------------------|
| **KANDy** | **2** | **0.049** | `u_t = -1.084*u*u_x - 0.026` |
| OLS | 5 | 0.150 | `u_t = -1.083*u*u_x + 0.063*u - 0.082*u_x + 0.020*u_xx + 0.024` |
| LASSO | 5 | 0.150 | `u_t = -1.081*u*u_x + 0.062*u - 0.082*u_x + 0.020*u_xx + 0.024` |
| PDE-FIND FD | 1 | 1.562 | `u_t = +12.272*u_x` **(wrong)** |
| PDE-FIND SmoothedFD | 1 | 1.562 | `u_t = +12.272*u_x` **(wrong)** |
| PDE-FIND SavGol | 10 | N/A | 10 garbage terms **(wrong)** |

True equation: `u_t = -u*u_x`

All equations rounded with shared tolerance |coeff| > 0.01.

## Why PDE-FIND Fails

1. **PySINDy's PDELibrary computes spatial derivatives with its own internal finite differences** — there is no way to plug in TVD minmod or any other shock-aware scheme.
2. Standard central FD on Rusanov-generated shock data produces Gibbs-like oscillations near discontinuities, corrupting the feature matrix.
3. The `differentiation_method` parameter in PySINDy only controls the **temporal** derivative (u_t computation), not spatial derivatives. All three temporal methods (FiniteDifference, SmoothedFiniteDifference, SavitzkyGolay) were tried — none help because the spatial features are broken.
4. PDE-FIND discovers `u_t = 12.27*u_x` — a linear advection equation with a wildly wrong wave speed, not the nonlinear Burgers equation.

## Why KANDy Succeeds

1. **TVD minmod derivatives** handle shocks correctly: they limit the derivative magnitude at discontinuities rather than oscillating, producing clean feature values.
2. **KAN's nonlinear spline edges** learn the u*u_x → u_t mapping even with the ~8% coefficient bias inherent to TVD limiters.
3. **Sparsity via learned edge activations**: KANDy naturally zeros inactive edges (u, u_xx) through training, leaving only the dominant u*u_x term. No explicit thresholding needed.
4. **Product-form features** [u, u_x, u*u_x, u_xx] work for this single-IC Rusanov setup. (Note: the multi-IC `burgers_fourier_example.py` with `solve_burgers`/TVD-RK2 solver required conservation form d(u²/2)/dx instead.)

## Critical Settings That Differ Between Research Script and Baseline

| Setting | Research script (works) | Failed baseline |
|---------|------------------------|-----------------|
| **Nx** | 128 | 316 |
| **Grid** | `linspace(endpoint=False)` | `arange(x_min, x_max+dx, dx)` |
| **K_fourier** | 10 | 20 |
| **dt** | 0.004 | 0.002 |
| **t_end** | 2.0 | 3.0 |
| **Spatial deriv** | TVD minmod | Central FD |
| **Time deriv** | Forward diff `(U[n+1]-U[n])/dt` | Central diff `(U[n+1]-U[n-1])/(2dt)` |

The derivative method is the dominant factor. Grid and IC complexity are secondary.

## Symbolic Extraction Procedure

Extracting clean equations from trained KANDy models requires care:

```python
import copy, sympy as sp, torch
from kandy.symbolic import robust_auto_symbolic

# 1. Deep copy (fix_symbolic destroys learned splines)
kan = copy.deepcopy(model_kandy.model_).cpu()

# 2. Enable activation caching (PyKAN disables after training)
kan.save_act = True

# 3. Forward pass to populate caches
sym_input = torch.tensor(data[:5000], dtype=torch.float32)
kan(sym_input)

# 4. DO NOT call kan.prune() — breaks on small KANs

# 5. Robust symbolic extraction
robust_auto_symbolic(kan, lib=['x','x^2','x^3','0'],
                     r2_threshold=0.80, weight_simple=0.80,
                     topk_edges=8, set_others_to_zero=True)

# 6. Extract formula and clean
formulas, _ = kan.symbolic_formula()
subs = {sp.Symbol(f'x_{i+1}'): sp.Symbol(n) for i, n in enumerate(FEAT_NAMES)}
expanded = sp.expand(formulas[0].subs(subs))

# 7. Drop near-zero terms, round coefficients
COEFF_TOL = 0.01
terms = []
for term in sp.Add.make_args(expanded):
    coeff, rest = term.as_coeff_Mul()
    if abs(float(coeff)) > COEFF_TOL:
        terms.append(sp.Float(round(float(coeff), 3)) * rest)
cleaned = sum(terms)
```

**Why rounding matters:** `fix_symbolic` fits `x^2` to near-constant edges, producing
artifacts like `-2.38*(4.02 - 0.0007*u_xx)² + 38.60`. After expanding, the large constants
cancel (38.60 - 2.38*4.02² ≈ 0.03) leaving only tiny corrections. Without expand+round,
the equation looks complex but is actually near-trivial.

## Key Takeaways

1. **Derivative method is everything** for PDE identification on shock data. TVD minmod enables correct identification; standard FD prevents it entirely.
2. **KANDy achieves both sparsity and accuracy** — 2-term equation with best rollout. OLS/LASSO find the right dominant term but can't zero the spurious ones.
3. **PDE-FIND (PySINDy) is fundamentally limited** for shock PDEs because its spatial derivatives are hardcoded to internal FD with no user override.
4. **Apply the same rounding tolerance to all methods** for fair comparison.

## Script

`examples/pdefind_baseline.py` — runs PDE-FIND (3 diff methods) + KANDy + OLS/LASSO.
Results saved to `results/Burgers-Fourier/baselines/`.
