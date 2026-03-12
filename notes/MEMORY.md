# KANDy Project Memory

## Project Overview
KANDy = Kolmogorov-Arnold Networks for Dynamics. Replaces SINDy's sparse regression with a single-layer KAN + Koopman-lifted coordinates.

## Git History (3 commits on main)
1. `8bdb149` - Initial Commit
2. `cf4c107` - Adding KANE-lifting
3. `2528722` - Good adaptive Kuramoto Results (latest)

## Key Directories
- `src/kandy/` - Package: core, lifts, numerics, plotting, symbolic, training
- `examples/` - Clean example scripts (11 systems + baselines)
- `research_code/` - Raw experiments + comprehensive baselines
- `results/` - ALL outputs, organized by system: Burgers/, Burgers-Fourier/, Kuramoto/
- `model/` - Trained KAN checkpoints (v0.0-0.3)

## Output Conventions
- All results to `results/<SystemName>/`, baselines to `results/<SystemName>/baselines/`
- Plots: PNG + PDF at 300 dpi, `kandy.plotting.use_pub_style()`
- Reports MUST include discovered equations in clean symbolic format

## Burgers Experiment Results (2026-03-08)

### Final KANDy Results
| Experiment | Limiter | Discovered Equation | Rollout RMSE |
|---|---|---|---|
| Burgers (sinusoidal IC) | superbee | u_t = -0.92*d(u²/2)/dx | 0.0036 |
| Burgers-Fourier (5 random ICs) | minmod | u_t = -0.82*d(u²/2)/dx | 0.434 |

Both correctly identify ONLY flux derivative edge (u, u_x edges zeroed). R²≥0.95 on active edge.

### Derivative Method Experiments (2026-03-08)
Tried to fix coefficient bias. Only superbee worked for sinusoidal IC. All others fail for Fourier:
- **Spectral (raw):** Gibbs oscillations → KAN learns sin(u) instead (wrong equation)
- **Spectral (filtered, order=8):** All edges zero (u_t=0)
- **Superbee:** Sinusoidal: coeff -0.92 (best!). Fourier: ALL edges zero
- **Van Leer:** Fourier: ALL edges zero
- **Product form u*u_x:** Fourier: ALL edges zero (only conservation form d(u²/2)/dx works)
- **N=256 grid:** Fourier: ALL edges zero
- **4 features (+u_xx):** Fourier: ALL edges zero
- **Conclusion:** Conservation-form feature + minmod is the only working config for Fourier IC

### Baseline Results (sinusoidal IC)
- OLS/Ridge/LASSO: u*u_x ≈ -1.53 + spurious terms, NRMSE ≈ 0.10
- Ideal sparse (u*u_x only): coeff = -1.40, NRMSE = 0.03

### Baseline Results (Fourier IC)
- OLS: u*u_x ≈ -1.08, NRMSE≈0.15
- PDE-FIND STLSQ (deg 3): 9 terms, NRMSE=0.08
- Ideal sparse: coeff = -0.53, NRMSE=0.58

### Key Finding: TVD Derivative Bias
Coefficient bias is intrinsic to TVD limiters near shocks. Not fixable without fundamentally different derivative scheme. KANDy's strength is structural identification (1 term vs 4-9 in baselines).

## Ikeda Optical-Cavity Map Results (2026-03-09)
**Details:** See [ikeda.md](ikeda.md)

- Discovered exact equations: x_{n+1} = 1.0 + 0.9*x*cos(t) - 0.9*y*sin(t), y_{n+1} = 0.9*x*sin(t) + 0.9*y*cos(t) (4+ sig figs)
- One-step MSE ~4e-8, rollout RMSE 0.88 (chaotic, expected)
- KAN: [4, 2] with RBF base, two-phase training (LBFGS + rollout fine-tuning)
- **Critical bug fix:** Double-lifting -- `KANDy.fit()` applies lift internally, so pass pre-computed features with identity lift, not the physics lift
- **Pattern:** When pre-computing features, use identity lift for model; keep physics lift external (data prep / rollout dynamics_fn)
- **Plotting API:** `plot_attractor_overlay()` and `plot_loss_curves()` do NOT accept `title=` kwarg; use `save=` for output

## Kuramoto-Sivashinsky Results (2026-03-10)
**Details:** See [ks.md](ks.md)

- **Discovered:** `u_t = -0.9966*u*u_x - 0.9959*u_xx - 0.9965*u_xxxx` (within 0.4%)
- **True:** `u_t = -u*u_x - u_xx - u_xxxx`
- Train loss: 1.1e-5, Pointwise RMSE: 2.3e-3, Rollout RMSE: 0.063
- KAN: [6, 1], grid=7, k=3, 100 LBFGS steps on 100K subsampled points
- 3/6 edges active (all linear, R²>0.9999), 3/6 zeroed — **perfect sparsity**
- ETDRK4 solver + IMEX rollout (stiff linear implicit, learned NL explicit)
- **Rating: Excellent**

### Critical fix: Minimal library + spectral derivatives
- Old 12-feature lib included u_xx AND u_xx² → KAN split u_xx signal across both
  (quadratic on u_xx edge, linear on u_xx² edge, nearly cancelling). auto_symbolic
  picked x² instead of x on u_xx.
- **Fix:** Remove squared features. New 6-feature lib: [u, u_x, u_xx, u_xxxx, u*u_x, u*u_xx]
- Also switched from FD to spectral derivatives (exact on periodic domains)
- **Lesson:** Lift should include cross-terms (u*u_x) but NOT powers of existing features (u_xx²)

## Adaptive Kuramoto-Sakaguchi Results (2026-03-10)
**Details:** See [adaptive_kuramoto.md](adaptive_kuramoto.md)

- N=5 oscillators with adaptive coupling weights κ_ij
- True: dθ_i/dt = ω_i + Σ_j κ_ij·sin(θ_i-θ_j-α), dκ_ij/dt = -ε·[κ_ij + sin(θ_i-θ_j-β)]
- **Rollout RMSE:** θ=0.035, κ=0.008, order param=0.0003 — **excellent**
- **Symbolic extraction:** Failed — equations have 30-50 terms each with small spurious coefficients
- Root cause: Same library degeneracy as KS but worse. Lift has sin/cos(θ_i-θ_j) AND κ_ij (correlated features), KAN distributes signal across many edges
- Two-model approach: theta KAN [10,5] + kappa KAN [60,20], both with base_fun=sin
- **Bug fix:** KANDy constructor sets device=CUDA by default; must pass `device="cpu"` when model_ is assigned manually (not via fit())

## Code Changes Applied
- `core.py`: Added `patience` param, force CPU device
- `numerics.py`: Added `spectral_derivative()` (useful for smooth PDEs like KS)
- `__init__.py`: Exports spectral_derivative
- `burgers_example.py`: superbee limiter, BURN=100, patience=0, SSP-RK3 rollout
- `burgers_fourier_example.py`: minmod limiter, conservation form, patience=0, SSP-RK3 rollout
- Both Burgers examples: SSP-RK3 + CFL substeps (matches paper's method)
- `ikeda_example.py`: Identity lift fix, CPU device, corrected plotting API calls
- `kuramoto_sivashinsky_example.py`: ETDRK4 solver, spectral derivatives, reduced 6-feature lib, IMEX rollout, OOM fix (subset for auto_symbolic), subsampling, patience=0, steps=100
- `adaptive_kuramoto_example.py`: Fixed DEVICE=cpu, added device="cpu" to KANDy constructor

## Plotting Module (`kandy.plotting`)
Key functions: `plot_all_edges`, `plot_edge`, `plot_loss_curves`, `plot_attractor_overlay`, `use_pub_style`
Edge API: `get_edge_activation(model, l=0, i=input_idx, j=output_idx, X=tensor)`
plot_edge: `fits=["linear", "sech2"]`, plot_all_edges: `in_var_names=`, `out_var_names=`
**Note:** `plot_attractor_overlay()` and `plot_loss_curves()` do NOT accept `title=` kwarg
