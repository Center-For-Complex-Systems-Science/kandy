# KANDy — Kolmogorov-Arnold Networks for Dynamics

KANDy is a scientific Python library for data-driven identification of dynamical systems.
It replaces the sparse regression step of SINDy with a single-layer Kolmogorov-Arnold
Network (KAN) and augments the inputs with user-designed Koopman-lifted coordinates:

```
x_dot = A · Ψ(φ(x))
```

where **φ** is a Koopman lift encoding all cross-interaction terms, **Ψ** is a separable
spline map learned by the KAN, and **A** is the extracted linear mixing matrix.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Mathematical Background](#mathematical-background)
4. [The KANDy Algorithm](#the-kandy-algorithm)
5. [Koopman Lifts](#koopman-lifts)
6. [Training](#training)
7. [Symbolic Extraction](#symbolic-extraction)
8. [Finite-Volume Numerics](#finite-volume-numerics)
9. [Benchmark Systems](#benchmark-systems)
10. [API Reference](#api-reference)
11. [Research Automation Crew](#research-automation-crew)

---

## Reproduction Instructions

List the systems
```{bash}
uv run kandy --list 
```


```{bash}
uv run kandy burgers 
```
The original research code used to develop this entire system is available in `research_code`. The coding examples
were refactored into this format in order to use it with AI Agents using claude code. Our analysis posits that
AI agents are more stable for comparing methods that require hyperparameter tuning. The agent memore is stored in
.claude directory. One may install claude code and load the local agents using the following command:

```{bash}
claude code .
```


## Installation

```bash
pip install -e .
```

**Requirements:** Python 3.11–3.13 · PyTorch ≥ 2.0 · PyKAN ≥ 0.2.0 · SciPy ≥ 1.10 · SymPy ≥ 1.12 · NumPy ≥ 1.24 · Matplotlib ≥ 3.7

---

## Quick Start

```python
from kandy import KANDy, PolynomialLift

# Lorenz system: lift R^3 → R^9  (includes xy, xz cross-terms)
lift  = PolynomialLift(degree=2)
model = KANDy(lift=lift, grid=5, k=3, steps=500)

# X: (N, 3) state trajectory,  X_dot: (N, 3) time derivatives
model.fit(X, X_dot)

# Autoregressive rollout
traj = model.rollout(x0, T=5000, dt=0.005)

# Symbolic extraction
formulas = model.get_formula(var_names=["x", "y", "z"])
print(formulas)   # [SymPy expr for x_dot, y_dot, z_dot]
```

If you only have a state trajectory (no derivatives), pass `dt` for central-difference
estimation:

```python
model.fit(X, dt=0.01)
```

---

## Mathematical Background

### Kolmogorov-Arnold Representation Theorem

The Kolmogorov-Arnold Representation Theorem (KAT) states that any continuous function
f(x₁, …, xₙ) admits a decomposition into sums of univariate continuous functions:

```
f(x₁, …, xₙ) = Σ_{q=1}^{2n+1} Φ_q( Σ_{i=1}^{n} φ_{q,i}(xᵢ) ).
```

This motivates Kolmogorov-Arnold Networks (KANs), in which each edge carries a learnable
univariate spline activation, replacing the fixed nonlinearities of standard MLPs.

### Why Standard KANs Cannot Recover Dynamical Systems

**Bilinear Obstruction (Corollary).** There do not exist continuous functions h, u, v :
[0,1] → ℝ such that

```
x · y  =  h(u(x) + v(y))    for all (x, y) ∈ [0,1]².
```

*Proof sketch.* Setting y = 0 gives h(u(x) + v(0)) = 0 for all x, so h vanishes on
[u(x) + v(0) : x ∈ [0,1]]. For any small y\* > 0 one finds x\* such that u(x\*) + v(y\*)
falls in this zero set, forcing h = 0. But h(u(x\*) + v(y\*)) = x\*y\* > 0 for x\* > 0 —
a contradiction. ∎

**Deep KAN Corollary.** Adding layers is equivalent to composing functions. The bilinear
obstruction propagates through each layer, so arbitrarily deep KANs without width
expansion cannot represent bilinear terms x·y from raw scalar inputs.

**Non-injectivity Obstruction (Proposition).** Let the true dynamics be ṡ = f(s) on ℝⁿ,
observed through z = h(s) ∈ ℝᵈ. If there exist two states s₁ ≠ s₂ with h(s₁) = h(s₂)
but (d/dt)h(s₁) ≠ (d/dt)h(s₂), then no single-valued function F : ℝᵈ → ℝᵈ satisfying
ż = F(z) exists for all trajectories. Consequently, no deterministic model class
(including KANs) can recover the governing equations from such observations.

*Example.* For the harmonic oscillator ẋ = v, v̇ = −x observed only through y = x, two
states (x₀, v₁) and (x₀, v₂) with v₁ ≠ v₂ produce the same observable y = x₀ but
different derivatives ẏ = v₁, v₂. The regression ż = F(z) is structurally ill-posed.

---

## The KANDy Algorithm

### Formulation

Given a Koopman lift φ : X → X̃ defined by

```
φ(x₁, …, xₙ) = (θ₁, θ₂, …, θₘ),
```

KANDy learns the system

```
ẋ = A · Ψ(φ(x)),
```

where:
- `Ψ(θ) = (ψ₁(θ₁), …, ψₘ(θₘ))` is a separable feature map, each `ψᵢ` a univariate
  spline.
- `A` is a learned `(n × m)` linear mixing matrix.

The model factorises as:

```
x  ──►  φ(x) = θ  ──►  Ψ(θ)  ──►  A · Ψ(θ) = ẋ
```

### Structural Properties

1. **Nonlinear in x** through the composition Ψ(φ(x)).
2. **Linear in A** — A can be read off directly from the KAN output weights.
3. **Nonlinear in the spline parameters** defining ψᵢ.
4. **Shallow network** — always `width = [m, n]` (one hidden layer). Depth is never
   exposed as a user parameter.

### Separable Dictionary and Cross-Terms

Each ψᵢ depends on exactly one lifted coordinate θᵢ, so Ψ is separable. **Cross-interaction
terms such as x·y cannot be generated by the KAN from raw inputs — they must be encoded
explicitly in the lift φ.**

For the Lorenz system (which has xy and xz in its RHS), the correct lift is:

```
φ(x, y, z) = (x, y, z, x², xy, xz, y², yz, z²)   ← PolynomialLift(degree=2)
```

### Relation to SINDy and Koopman Methods

| Method | Dictionary | Regression | Evolution space |
|---|---|---|---|
| SINDy | Fixed Θ(x) | LASSO (sparse) | Original state x |
| EDMD  | Fixed Ψ(x) | Least-squares  | Lifted state Ψ   |
| **KANDy** | **Learned Ψ(φ(x))** | **KAN** | **Original state x** |

KANDy differs from classical EDMD in that it models `ẋ = A · Ψ` rather than `Ψ̇ = K · Ψ`.
It is a feature-based regression in the original state space, not a linear evolution law
in lifted space.

---

## Koopman Lifts

The lift is the most critical design choice. It must encode **all** cross-interaction terms
present in the target system's RHS. Missing cross-terms make the algorithm structurally
incorrect, not merely inaccurate.

| Class | Description | `output_dim` |
|---|---|---|
| `PolynomialLift(degree)` | All monomials up to given degree | Binomial(n+d, d) |
| `FourierLift(n_modes)` | DC + Re/Im parts of leading Fourier modes. For periodic PDE fields (u ∈ ℝᴺ). | 1 + 2·n\_modes |
| `RadialBasisLift(n_centers, sigma, center_method)` | Gaussian RBF dictionary. Auto σ via median-distance heuristic. Centres from random subsampling or k-means. | n\_centers |
| `DMDLift(n_modes, dictionary, sort_by)` | EDMD-based Koopman eigenfunctions from trajectory data. Separates real and complex-conjugate pairs. | n\_real + 2·n\_complex |
| `CustomLift(fn, output_dim)` | Wrap any hand-crafted feature function | user-specified |

```python
from kandy import PolynomialLift, FourierLift, RadialBasisLift, DMDLift, CustomLift
import numpy as np

# Polynomial — Lorenz, Holling, Hénon, ...
lift = PolynomialLift(degree=2)

# Fourier modes — Burgers, KS PDE, ...
lift = FourierLift(n_modes=16)

# Gaussian RBF dictionary
lift = RadialBasisLift(n_centers=50, center_method="kmeans")

# Data-driven Koopman eigenfunctions
lift = DMDLift(n_modes=10, dictionary=PolynomialLift(degree=2))

# Custom physics-informed lift (Ikeda optical-cavity map)
def ikeda_features(X):
    x, y = X[:, 0], X[:, 1]
    t  = 0.4 - 6.0 / (1.0 + x**2 + y**2)
    ct, st = np.cos(t), np.sin(t)
    return np.column_stack([0.9*x*ct, 0.9*y*ct, 0.9*x*st, 0.9*y*st])

lift = CustomLift(fn=ikeda_features, output_dim=4)
```

---

## Training

### `KANDy.fit`

```python
model.fit(
    X,                          # (N, n) state trajectory
    X_dot=None,                 # (N, n) time derivatives; omit if passing dt
    dt=None,                    # time step for central-difference estimation
    opt="LBFGS",                # "LBFGS" (default) or "Adam"
    lr=1.0,                     # learning rate (use ~1e-3 for Adam)
    batch=-1,                   # mini-batch size (-1 = full batch)
    lamb=0.0,                   # sparsity regularisation (L1 + entropy)
    rollout_weight=0.0,         # weight on trajectory rollout loss
    rollout_loss_fn=None,       # separate loss for rollout; defaults to MSE
    fit_steps=None,             # override self.steps for this call
    val_frac=0.15,
    test_frac=0.15,
)
```

Use **LBFGS** for most systems (default). Use **Adam** for large datasets or discrete-map
training with many parameters (e.g. Holling Type II, Ikeda with rollout).

### Discrete Maps

Pass the current state as `X` and the next state as `X_dot`. KANDy learns the one-step
map directly:

```python
model.fit(X_current, X_next, opt="Adam", lr=2e-3, batch=4096)
```

For long-horizon discrete rollout, use the **increment trick**:

```python
# dynamics_fn(s) = map(s) - s  →  Euler with dt=1 recovers exact map iteration
def discrete_rhs(state):
    return map_fn(state) - state

fit_kan(model.model_, dataset, integrator="euler", dynamics_fn=discrete_rhs, ...)
```

### Rollout Loss

`fit_kan` accepts a full trajectory dataset for multi-step loss:

```python
from kandy import fit_kan, make_windows

train_windows = make_windows(train_traj, window=16)   # (Nw, 16, state_dim)

dataset = {
    "train_input": Theta_train,  "train_label": Y_train,
    "test_input":  Theta_test,   "test_label":  Y_test,
    "train_traj":  train_windows, "train_t": t_window,
    "test_traj":   test_windows,  "test_t":  t_window,
}

fit_kan(
    model.model_,
    dataset,
    opt="LBFGS",
    steps=100,
    rollout_weight=0.6,
    rollout_horizon=15,
    dynamics_fn=my_dynamics_fn,   # state → derivative (must apply lift internally)
    integrator="rk4",             # "rk4" or "euler"
)
```

---

## Symbolic Extraction

### Basic Extraction

```python
# After fitting, populate activations
formulas = model.get_formula(
    var_names=["x", "y", "z"],   # lift feature names
    round_places=3,
    simplify=False,              # True: factor → together → nsimplify pipeline
)
# Returns a list of SymPy expressions
```

### Physics-Informed Extraction

`auto_symbolic_with_costs` assigns different complexity libraries to KAN edges based on
whether their input feature is a known-physics term. Preferred features get cheap costs so
the solver selects them when fit quality is comparable.

```python
from kandy import auto_symbolic_with_costs, TRIG_LIB_CHEAP, TRIG_LIB, POLY_LIB_CHEAP

model.model_.save_act = True
with torch.no_grad():
    model.model_(train_features)

# Ikeda: all 4 trig-product features are physics-informed
auto_symbolic_with_costs(
    model.model_,
    preferred_idx=set(range(4)),
    preferred_lib=TRIG_LIB_CHEAP,   # sin/cos at cost 2
    other_lib=TRIG_LIB,             # sin/cos at cost 4
    weight_simple=0.1,
    r2_threshold=0.80,
    verbose=1,
)
```

Pre-built libraries:

| Name | Description |
|---|---|
| `POLY_LIB_CHEAP` | x, x², x³, 0 at costs 1–3 |
| `POLY_LIB` | x, x², x³, 0 at costs 3–5 |
| `TRIG_LIB_CHEAP` | Polynomial + sin, cos at costs 1–2 |
| `TRIG_LIB` | Polynomial + sin, cos at costs 3–4 |

Custom libraries:

```python
from kandy import make_symbolic_lib
import torch, sympy as sp

my_lib = make_symbolic_lib({
    "x":      (lambda x: x,             lambda x: x,             1),
    "exp":    (torch.exp,               sp.exp,                  3),
    "sech2":  (lambda x: 1/torch.cosh(x)**2, lambda x: 1/sp.cosh(x)**2, 3),
})
```

### Scoring and Export

```python
from kandy import score_formula, formulas_to_latex, substitute_params

# R² of each formula on held-out data
r2 = score_formula(formulas, Theta_test, Y_test, var_names=FEATURE_NAMES)
print(r2)   # e.g. [0.9987, 0.9991, 0.9995]

# LaTeX output
tex = formulas_to_latex(
    formulas,
    lhs_names=[r"\dot{x}", r"\dot{y}", r"\dot{z}"],
)
print(tex)
# \begin{align*}
#   \dot{x} &= 10.0 y - 10.0 x  \\
#   \dot{y} &= 28.0 x - y - x z  \\
#   \dot{z} &= x y - 2.667 z
# \end{align*}

# Substitute known parameter values
sub = substitute_params(formulas, {"sigma": 10.0, "rho": 28.0, "beta": 2.667})
```

`KANDy.score_formula` is a convenience wrapper that applies the lift automatically:

```python
r2 = model.score_formula(formulas, X_test, Y_test)
```

---

## Finite-Volume Numerics

KANDy ships a self-contained finite-volume module for generating PDE training data on
periodic 1D domains (`u_t + ∂_x F(u) = 0`).

### Flux Schemes

| Scheme | Function | Notes |
|---|---|---|
| Rusanov (LLF) | `rusanov_flux` | Most diffusive; always stable |
| Roe upwind | `roe_flux` | Less diffusive; Harten–Hyman entropy fix at sonic points |
| HLL (HLLC) | `hllc_flux` | Two-wave solver; HLLC = HLL for scalar laws |

All schemes use MUSCL reconstruction (2nd-order) with a choice of slope limiter:
`minmod`, `van_leer`, or `superbee`.

### Time Integrators

| Name | Function | Order |
|---|---|---|
| SSP-RK2 (Heun) | `tvdrk2_step` | 2nd, TVD |
| SSP-RK3 (Shu–Osher) | `tvdrk3_step` | 3rd, TVD |

### CFL and Convenience Solvers

```python
from kandy import solve_burgers, solve_viscous_burgers, cfl_dt
import numpy as np

N  = 256
x  = np.linspace(0, 2*np.pi, N, endpoint=False)
u0 = np.sin(x)
dx = x[1] - x[0]

# Stable time step from CFL condition
dt = cfl_dt(u0, dx, cfl=0.4)

# Inviscid Burgers  u_t + (u²/2)_x = 0
U = solve_burgers(u0, n_steps=500, dt=dt, scheme="roe", limiter="van_leer")

# Viscous Burgers  u_t + (u²/2)_x = ν u_xx
# IMEX: explicit TVD-RK convection + exact spectral implicit diffusion
U = solve_viscous_burgers(u0, n_steps=500, dt=dt, nu=0.01)

print(U.shape)   # (500, 256)
```

`solve_scalar` accepts any flux and wave-speed functions for custom conservation laws:

```python
from kandy import solve_scalar

U = solve_scalar(
    u0, dx, n_steps=1000, dt=dt,
    flux_fn=my_flux,
    speed_fn=my_speed,
    roe_speed_fn=my_roe_speed,  # required for scheme='roe'
    scheme="roe",
    limiter="superbee",
    time_stepper="tvdrk3",
)
```

---

## Benchmark Systems

KANDy has been validated on ten dynamical systems spanning ODEs, discrete maps, and PDEs.
Example scripts for all ten are in [`examples/`](examples/).

| System | Type | Lift | Optimizer |
|---|---|---|---|
| Lorenz-63 | ODE (chaos) | `PolynomialLift(2)` on ℝ³ | LBFGS |
| Hopf fibration | Map S³ → S² | Identity / 5D engineered | LBFGS |
| Hénon map | Discrete map | [x, y, x²] | LBFGS |
| Ikeda optical cavity | Discrete map | 4D trig physics lift | LBFGS + rollout |
| Kuramoto–Sivashinsky | PDE | 12D local features | LBFGS |
| Inviscid Burgers | PDE | [u, u_x, ∂(u²/2)/∂x] | LBFGS |
| Burgers (Fourier ICs) | PDE | [u, u_x, ∂(u²/2)/∂x] | LBFGS |

---

## API Reference

### `KANDy`

```python
KANDy(lift, grid=5, k=3, steps=500, seed=42, device=None, base_fun=None)
```

| Method | Description |
|---|---|
| `fit(X, X_dot, ...)` | Fit the model |
| `predict(X)` | Predict derivatives / next state for X |
| `rollout(x0, T, dt, integrator='rk4')` | Autoregressive trajectory integration |
| `get_formula(var_names, round_places, simplify)` | Symbolic extraction via PyKAN |
| `score_formula(formulas, X, y_true, var_names)` | R² of symbolic formulas on data |
| `get_A()` | Extract linear mixing matrix A ∈ ℝⁿˣᵐ |

### Lifts

```python
PolynomialLift(degree, include_bias=True)
FourierLift(n_modes)
RadialBasisLift(n_centers, sigma=None, center_method='random')
DMDLift(n_modes, dictionary=None, sort_by='magnitude')
CustomLift(fn, output_dim, name='custom')
```

All lifts inherit from `Lift` (ABC) and implement `__call__(X)`, `output_dim`, and
optionally `fit(X)` (for data-dependent lifts such as `RadialBasisLift` and `DMDLift`).

### Symbolic

```python
auto_symbolic_with_costs(model, preferred_idx, preferred_lib, other_lib,
                          weight_simple=0.8, r2_threshold=0.90, verbose=1)
score_formula(formulas, theta, y_true, var_names)     # → list[float]
formulas_to_latex(formulas, lhs_names, environment='align*')  # → str
substitute_params(formulas, params)                   # → list[Expr]
make_symbolic_lib({name: (torch_fn, sympy_fn, cost)}) # → PyKAN lib dict
```

### Numerics

Additional CFD numerical integrators available.

### Training

```python
model = KANDy(lift, grid=5, k=3, steps=500, seed=42, device=None, base_fun=None)
model.fit(dataset)
```

# Maintainers
author: [Kevin Slote](www.kevin-slote.com)
email: kslote@clarkson.edu or kslote1@gmail.com
