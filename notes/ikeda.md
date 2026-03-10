# Ikeda Optical-Cavity Map Results (2026-03-09)

## Bug Fix: Double-Lifting in ikeda_example.py
- **Root cause:** `KANDy.fit()` applies the lift internally (core.py line 193: `theta = self.lift(X)`). The Ikeda example was passing pre-lifted normalized features AND using `ikeda_lift` as the model's lift, causing double-lifting. The KAN learned garbage (loss stuck at 0.173).
- **Fix:** Use identity lift `CustomLift(fn=lambda X: X, output_dim=4)` for the model. Keep `ikeda_lift` as a separate utility for the rollout `dynamics_fn`.
- **General pattern:** When passing pre-computed features to `model.fit()`, ALWAYS use an identity lift. The physics-informed lift should only be used externally (for data prep or rollout dynamics_fn).

## Final Results
| Metric | Value |
|---|---|
| One-step MSE | ~4e-8 |
| Rollout RMSE (400 steps) | 0.88 (chaotic, exponential error growth expected) |
| x_{n+1} coeff on x*cos(t) | 0.9000 (true: 0.9) |
| x_{n+1} coeff on y*sin(t) | -0.9000 (true: -0.9) |
| x_{n+1} constant | 1.0001 (true: 1.0) |
| y_{n+1} coefficients | 0.9/0.9 (exact match) |
| Rating | Excellent, publication-ready |

## Symbolic Discovery
Discovered equations (with normalization correction):
```
x_{n+1} = 1.0001 + 0.9*x*cos(t) - 0.9*y*sin(t)
y_{n+1} = 0.9*x*sin(t) + 0.9*y*cos(t)
```
Match true equations to 4+ significant figures.

Note: sympy may express phase as `6/(1+r^2) - 0.4` instead of `0.4 - 6/(1+r^2)`, which flips sin sign but not cos. Both are equivalent: `sin(6/(1+r^2) - 0.4) = -sin(0.4 - 6/(1+r^2))`.

## Architecture Details
- KAN: [4, 2], base_fun = RBF (exp(-x^2))
- Physics-informed lift: phi(x,y) = [u*x*cos(t), u*y*cos(t), u*x*sin(t), u*y*sin(t)]
- Two-phase training: Phase 1 = 200 steps LBFGS one-step, Phase 2 = rollout fine-tuning (rollout_weight=0.2, lr=1e-2)
- Discrete map "increment trick": dynamics_fn(s) = map(s) - s, Euler with dt=1
- Normalization substitution: KAN variables are (phi_i - mean_i) / std_i, must account for this in symbolic formula extraction
- Symbolic extraction: auto_symbolic_with_costs with TRIG_LIB_CHEAP, R^2 threshold=0.80

## Bugs Fixed in ikeda_example.py
1. Double-lifting (identity lift fix) -- root cause of training failure
2. DEVICE forced to CPU (PyKAN CUDA grid-update bugs)
3. plot_all_edges: changed input_names/output_names to in_var_names/out_var_names
4. plot_attractor_overlay: removed unsupported `title=` kwarg
5. plot_loss_curves: removed unsupported `title=` kwarg

## Important Pattern for Future Examples
- `plot_attractor_overlay()` does NOT accept `title` kwarg
- `plot_loss_curves()` does NOT accept `title` kwarg
- Both accept `save=` for output path
