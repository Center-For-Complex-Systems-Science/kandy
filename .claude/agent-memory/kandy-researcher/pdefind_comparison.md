---
name: PDE-FIND vs KANDy Baseline Comparison
description: Results from head-to-head PDE-FIND (PySINDy PDELibrary) vs KANDy on inviscid Burgers with Fourier ICs, including key lessons about derivative methods and symbolic extraction
type: project
---

## PDE-FIND vs KANDy Comparison — Inviscid Burgers (2026-03-17)

### Setup that works (matches research_code/burgers_fourier_baselines.py)
- **Grid:** Nx=128, `np.linspace(x_min, x_max, Nx, endpoint=False)` — proper periodic grid
- **IC:** K=10 Fourier modes, seed=0, power-law decay p=1.5
- **Domain:** [-π, π], t ∈ [0, 2], dt=0.004
- **Solver:** First-order Rusanov flux + RK45 (scipy, rtol=1e-6, atol=1e-8)
- **KANDy derivatives:** TVD minmod (spatial), forward difference (temporal)
- **KANDy features:** [u, u_x, u*u_x, u_xx] → KAN [4, 1], grid=7, k=3, 300 steps

### Results

| Method | Terms | NRMSE | Equation |
|---|---|---|---|
| **KANDy** | **2** | **0.049** | `u_t = -1.084*u*u_x - 0.026` |
| OLS | 5 | 0.150 | `u_t = -1.083*u*u_x + 4 spurious` |
| LASSO | 5 | 0.150 | `u_t = -1.081*u*u_x + 4 spurious` |
| PDE-FIND (all 3 diff methods) | 1 | 1.562 | `u_t = +12.27*u_x` (**wrong equation**) |

### Why PDE-FIND fails on this data
- PySINDy's PDELibrary computes spatial derivatives with its own internal FD (not TVD)
- Standard central FD on Rusanov-generated shock data produces Gibbs-like oscillations
- All 3 PySINDy differentiation methods (FiniteDifference, SmoothedFiniteDifference, SavitzkyGolay) fail
- PDE-FIND discovers `u_t = 12.27*u_x` instead of `u_t = -u*u_x`
- **Why:** `differentiation_method` in PySINDy only controls the TEMPORAL derivative; PDELibrary always uses its own FD for SPATIAL derivatives — no way to plug in TVD minmod

### Why KANDy succeeds
- TVD minmod derivatives handle shocks correctly (no Gibbs oscillations)
- KAN's nonlinear spline edges can learn the relationship even with TVD derivative bias
- Product-form features [u, u_x, u*u_x, u_xx] work here (unlike the multi-IC burgers_fourier_example.py which needed conservation form)
- Forward-diff time derivative (matching research script) works better than central-diff

### Critical lessons for derivative method choice
1. **TVD minmod vs central FD:** Central FD fails on shock data (oscillations). TVD minmod introduces coefficient bias (~8%) but correctly identifies equation structure.
2. **Forward vs central time derivative:** Research script uses forward diff `(U[n+1]-U[n])/dt`, current examples used central diff `(U[n+1]-U[n-1])/(2dt)`. Forward diff matched better.
3. **Grid construction:** `np.linspace(endpoint=False)` for proper periodic grid vs `np.arange(x_min, x_max+dx, dx)` which includes a near-duplicate boundary point.
4. **Resolution matters:** Nx=128 with K=10 modes works. Nx=316 with K=20 modes produces more complex shocks that are harder for all methods.

### Symbolic extraction fixes (robust_auto_symbolic)
- **Export:** Added `robust_auto_symbolic` to `kandy.__init__` exports
- **Device mismatch:** Must call `kan.cpu()` before symbolic extraction (PyKAN grid tensors stay on CPU)
- **save_act=True:** Must set `kan.save_act = True` before forward pass or `spline_postacts` is empty
- **Deep copy:** Must use `copy.deepcopy(model_kandy.model_)` for symbolic extraction — `fix_symbolic` destroys learned splines, corrupting the model for rollout
- **No prune:** Don't call `kan.prune()` on [4,1] KAN — it clears internal caches and dimensions mismatch
- **Rounding:** Expand symbolic formula with `sp.expand()`, drop terms with |coeff| < COEFF_TOL (0.01), round to 3 decimal places. Apply same tolerance to ALL methods for fair comparison.

### Script location
`examples/pdefind_baseline.py` — runs PDE-FIND (3 diff methods) + KANDy + OLS/LASSO, generates spacetime rollout plots and equation comparison table. Results in `results/Burgers-Fourier/baselines/`.
