# Adaptive Kuramoto-Sakaguchi: KANDy Experiment Report

**Date:** 2026-03-10
**System:** Adaptive Kuramoto-Sakaguchi coupled oscillators (N=5)

## True Equations

Phase dynamics:

    dtheta_i/dt = omega_i + sum_j kappa_ij * sin(theta_j - theta_i + alpha)

Coupling adaptation:

    dkappa_ij/dt = -epsilon * [kappa_ij + sin(theta_j - theta_i + beta)]

**Parameters:** alpha = -pi/4, beta = -pi/2, epsilon = 0.1, omega_i ~ Uniform(-0.5, 0.5)

## Setup

| Parameter | Value |
|---|---|
| Oscillators | N = 5 |
| State dimension | 25 (5 phases + 20 coupling weights) |
| Simulation | `solve_ivp`, T=300, dt=0.05, 6000 snapshots |
| Training samples | 5998 (central differences) |
| Train/Val/Test split | 70% / 15% / 15% |

### Model Architecture

Two separate KANDy models (the system separates into phase and coupling dynamics):

| | Phase Model (theta) | Coupling Model (kappa) |
|---|---|---|
| KAN width | [60, 5] | [60, 20] |
| Grid | 5 | 5 |
| Spline order k | 3 | 3 |
| Base function | sin | sin |
| Training | 200 LBFGS steps + rollout fine-tuning | 200 LBFGS steps |
| Rollout horizon | 8 steps (weight=0.3) | -- |
| Total edges | 300 | 1200 |

### Koopman Lift (60 features)

For each pair (i,j) where i != j (20 pairs):
- sin(theta_i - theta_j)  -- 20 features
- cos(theta_i - theta_j)  -- 20 features
- kappa_ij                 -- 20 features

Both sin and cos are needed because sin(Delta_theta + alpha) = sin(Delta_theta)cos(alpha) + cos(Delta_theta)sin(alpha).

## Results

### Rollout Metrics

| Metric | Value | Rating |
|---|---|---|
| Rollout RMSE theta | 0.035 | Excellent |
| Rollout RMSE kappa | 0.008 | Excellent |
| Order parameter RMSE | 0.0003 | Excellent |

### Training Convergence

Phase model (final step 199):
- Train loss: 3.17e-7
- Test loss: 3.65e-8
- Rollout train: 8.24e-10

### Rollout Quality

- **Phase trajectories:** Near-perfect overlap between true and predicted for all 5 oscillators over 25 time units (500 steps at dt=0.05)
- **Order parameter r(t):** Virtually indistinguishable from ground truth, stays at r ~ 0.21
- **Coupling weights:** Good match up to t ~ 20, then predicted weights begin to drift while true weights remain nearly constant. Drift is small (O(0.01)) but visible.

### Plots

- `phase_trajectories.png` -- 5 subplots showing true (solid) vs KANDy (dashed) phase evolution
- `order_parameter.png` -- r(t) true vs predicted (RMSE=0.0003)
- `kappa_trajectories.png` -- 4 representative coupling weights showing late-time divergence
- `loss_phase.png` -- Train/test + rollout loss convergence (~10^-8 final)

## Symbolic Extraction

**Status: Failed** -- equations are too messy to be scientifically useful.

The true equations have a clean, sparse structure:
- Each dtheta_i/dt has 1 + (N-1) = 5 terms
- Each dkappa_ij/dt has 2 terms

The discovered equations have 30-50 terms each, including many spurious terms with small coefficients (0.01-0.30). Example:

### Phase equations (showing dtheta_0/dt only)

**True:**

    dtheta_0/dt = omega_0 + kappa_01*sin(theta_1-theta_0+alpha) + kappa_02*sin(theta_2-theta_0+alpha) + kappa_03*sin(theta_3-theta_0+alpha) + kappa_04*sin(theta_4-theta_0+alpha)

**Discovered:**

    dtheta_0/dt = -0.08*cos(theta_1-theta_3) - 0.06*cos(theta_1-theta_4) + 0.06*cos(theta_2-theta_4) - 0.05*cos(theta_3-theta_1) + 0.02*cos(theta_3-theta_4) - 0.03*cos(theta_4-theta_1) - 0.06*cos(theta_4-theta_3) - 0.03*sin(theta_0-theta_1) - 0.09*sin(theta_0-theta_2) - 0.13*sin(theta_1-theta_0) + 0.14*sin(theta_2-theta_3) + 0.15*sin(theta_3-theta_1) + 0.03*sin(theta_3-theta_4) + 0.11*sin(theta_4-theta_2) - 0.06*sin(theta_4-theta_3) + 0.1*kappa_01 + 0.06*kappa_04 + 0.11*kappa_10 + 0.22*kappa_13 + 0.1*kappa_21 - 0.08*kappa_23 + 0.21*kappa_24 - 0.07*kappa_30 + 0.09*kappa_31 - 0.1*kappa_32 + 0.05*kappa_40 - 0.11*kappa_41 + 0.16*kappa_42 + 0.04*kappa_43 - 0.12*sin(1.0*kappa_03 + 0.01) + 0.3*sin(1.1*kappa_14 - 9.49) + 0.12*sin(1.18*kappa_20 + 3.01) - 0.24*sin(0.86*kappa_34 - 9.52) + 0.09*cos(1.42*kappa_12 + 5.0) - 0.6 + 0.26*exp(-0.94*kappa_02)

### Full Discovered Phase Equations

```
dtheta_0/dt = -0.08*cos(theta_1-theta_3) - 0.06*cos(theta_1-theta_4) + 0.06*cos(theta_2-theta_4) - 0.05*cos(theta_3-theta_1) + 0.02*cos(theta_3-theta_4) - 0.03*cos(theta_4-theta_1) - 0.06*cos(theta_4-theta_3) - 0.03*sin(theta_0-theta_1) - 0.09*sin(theta_0-theta_2) - 0.13*sin(theta_1-theta_0) + 0.14*sin(theta_2-theta_3) + 0.15*sin(theta_3-theta_1) + 0.03*sin(theta_3-theta_4) + 0.11*sin(theta_4-theta_2) - 0.06*sin(theta_4-theta_3) + 0.1*kappa_01 + 0.06*kappa_04 + 0.11*kappa_10 + 0.22*kappa_13 + 0.1*kappa_21 - 0.08*kappa_23 + 0.21*kappa_24 - 0.07*kappa_30 + 0.09*kappa_31 - 0.1*kappa_32 + 0.05*kappa_40 - 0.11*kappa_41 + 0.16*kappa_42 + 0.04*kappa_43 - 0.12*sin(1.0*kappa_03 + 0.01) + 0.3*sin(1.1*kappa_14 - 9.49) + 0.12*sin(1.18*kappa_20 + 3.01) - 0.24*sin(0.86*kappa_34 - 9.52) + 0.09*cos(1.42*kappa_12 + 5.0) - 0.6 + 0.26*exp(-0.94*kappa_02)

dtheta_1/dt = 0.07*cos(theta_1-theta_4) - 0.06*cos(theta_2-theta_4) - 0.09*cos(theta_3-theta_4) + 0.08*cos(theta_4-theta_2) - 0.06*cos(theta_4-theta_3) - 0.06*sin(theta_0-theta_1) - 0.11*sin(theta_0-theta_2) - 0.07*sin(theta_0-theta_3) + 0.03*sin(theta_1-theta_0) + 0.06*sin(theta_1-theta_2) + 0.14*sin(theta_2-theta_0) + 0.08*sin(theta_2-theta_1) - 0.12*sin(theta_3-theta_0) + 0.05*sin(theta_3-theta_1) - 0.05*sin(theta_3-theta_4) - 0.07*sin(theta_4-theta_0) + 0.07*sin(theta_4-theta_2) + 0.05*kappa_01 - 0.06*kappa_02 + 0.06*kappa_10 - 0.17*kappa_14 + 0.11*kappa_21 - 0.09*kappa_30 + 0.18*kappa_34 + 0.17*kappa_40 + 0.08*kappa_42 + 0.12*kappa_43 - 0.13*sin(1.22*cos(theta_4-theta_1) + 6.18) + 0.28*sin(0.94*kappa_04 - 0.05) + 0.2*sin(1.15*kappa_13 + 6.37) - 0.21*sin(1.12*kappa_24 - 3.07) + 0.46*cos(0.48*kappa_03 - 4.13) + 0.23*cos(0.93*kappa_20 + 7.84) + 0.22*cos(1.08*kappa_41 + 1.51) - 0.44

dtheta_2/dt = -0.04*cos(theta_1-theta_3) - 0.07*cos(theta_1-theta_4) + 0.13*cos(theta_3-theta_1) - 0.05*cos(theta_3-theta_4) + 0.05*cos(theta_4-theta_0) + 0.01*cos(theta_4-theta_1) + 0.05*cos(theta_4-theta_2) + 0.02*cos(theta_4-theta_3) + 0.05*sin(theta_0-theta_1) + 0.05*sin(theta_1-theta_0) - 0.09*sin(theta_1-theta_2) + 0.03*sin(theta_1-theta_4) + 0.07*sin(theta_2-theta_0) - 0.12*sin(theta_2-theta_4) + 0.07*sin(theta_3-theta_4) + 0.08*sin(theta_4-theta_0) - 0.08*sin(theta_4-theta_2) - 0.06*sin(theta_4-theta_3) - 0.04*kappa_02 + 0.09*kappa_04 + 0.13*kappa_10 - 0.03*kappa_23 - 0.05*kappa_30 - 0.14*kappa_32 + 0.1*kappa_40 + 0.14*kappa_42 + 0.08*kappa_43 - 0.09*sin(1.18*sin(theta_2-theta_3) + 3.0) + 0.14*sin(1.21*kappa_01 + 6.42) - 0.07*sin(0.83*kappa_12 - 9.56) - 0.05*sin(1.1*kappa_13 + 9.49) - 0.21*sin(1.34*kappa_14 - 0.35) - 0.23*sin(0.94*kappa_21 - 9.47) + 0.25*sin(1.04*kappa_41 + 9.36) + 0.39*cos(0.56*kappa_03 - 4.36) - 0.22*cos(0.94*kappa_20 + 4.63) + 0.54*cos(0.53*kappa_24 - 8.28) - 0.17*cos(0.98*kappa_34 + 1.56) + 0.18

dtheta_3/dt = 0.08*cos(theta_0-theta_4) + 0.05*cos(theta_1-theta_4) + 0.05*cos(theta_2-theta_4) - 0.05*cos(theta_4-theta_1) - 0.06*cos(theta_4-theta_2) - 0.11*cos(theta_4-theta_3) + 0.19*sin(theta_0-theta_3) - 0.14*sin(theta_1-theta_3) + 0.07*sin(theta_1-theta_4) + 0.1*sin(theta_2-theta_1) + 0.18*sin(theta_2-theta_3) - 0.13*sin(theta_2-theta_4) + 0.16*sin(theta_3-theta_1) + 0.08*sin(theta_3-theta_4) + 0.04*sin(theta_4-theta_0) + 0.09*sin(theta_4-theta_1) + 0.02*sin(theta_4-theta_3) - 0.05*kappa_02 + 0.13*kappa_12 + 0.14*kappa_21 - 0.14*kappa_23 + 0.11*kappa_31 + 0.09*kappa_40 - 0.11*kappa_41 + 0.18*kappa_42 + 0.06*kappa_43 + 0.23*sin(1.17*kappa_03 - 9.57) + 0.13*sin(1.28*kappa_14 + 2.95) - 0.22*sin(1.11*kappa_34 - 3.07) - 0.09*cos(0.99*kappa_01 + 1.57) - 0.1*cos(1.25*kappa_04 + 8.05) + 0.09*cos(1.3*kappa_13 + 4.88) - 0.23*cos(0.87*kappa_20 - 1.45) - 0.13*cos(1.15*kappa_24 + 7.95) - 0.76 + 0.2*exp(-0.77*kappa_30)

dtheta_4/dt = 0.06*cos(theta_0-theta_1) - 0.14*cos(theta_1-theta_3) + 0.07*cos(theta_1-theta_4) + 0.1*cos(theta_2-theta_4) + 0.13*cos(theta_3-theta_4) + 0.03*cos(theta_4-theta_0) + 0.07*cos(theta_4-theta_1) + 0.03*cos(theta_4-theta_3) + 0.04*sin(theta_0-theta_1) - 0.17*sin(theta_0-theta_2) - 0.04*sin(theta_0-theta_3) + 0.17*sin(theta_1-theta_2) - 0.17*sin(theta_2-theta_1) + 0.1*sin(theta_2-theta_3) - 0.11*sin(theta_3-theta_0) + 0.17*sin(theta_3-theta_1) - 0.17*sin(theta_3-theta_2) - 0.14*sin(theta_3-theta_4) + 0.2*sin(theta_4-theta_0) - 0.08*sin(theta_4-theta_1) + 0.04*sin(theta_4-theta_2) + 0.08*sin(theta_4-theta_3) - 0.09*kappa_02 + 0.18*kappa_04 + 0.12*kappa_10 + 0.1*kappa_12 - 0.13*kappa_23 + 0.12*kappa_31 - 0.05*kappa_32 + 0.2*kappa_34 + 0.17*kappa_40 + 0.12*kappa_42 + 0.11*sin(1.2*kappa_13 + 6.4) - 0.16*sin(1.08*kappa_14 - 0.06) - 0.05*sin(1.5*kappa_20 - 0.29) - 0.12*sin(1.1*kappa_24 + 9.49) + 0.26*sin(1.24*kappa_41 - 3.42) + 0.16*cos(0.75*kappa_01 - 8.04) - 0.14*cos(1.03*kappa_03 - 7.83) + 0.07*cos(1.51*kappa_21 + 5.09) - 0.62
```

### Coupling equations (showing first 4 only)

<details>
<summary>Click to expand dkappa equations</summary>

```
dkappa_01/dt = 0.08*cos(theta_0-theta_1) - 0.08*cos(theta_0-theta_2) + 0.05*cos(theta_0-theta_3) - 0.07*cos(theta_1-theta_0) + 0.06*cos(theta_1-theta_2) + 0.06*cos(theta_3-theta_2) - 0.09*cos(theta_3-theta_4) - 0.05*sin(theta_0-theta_1) - 0.08*sin(theta_0-theta_2) - 0.08*sin(theta_0-theta_3) - 0.08*sin(theta_0-theta_4) + 0.08*sin(theta_1-theta_0) + 0.09*sin(theta_1-theta_2) + 0.05*sin(theta_1-theta_4) - 0.06*sin(theta_2-theta_0) + 0.06*sin(theta_2-theta_1) + 0.04*sin(theta_2-theta_4) + 0.11*sin(theta_3-theta_0) + 0.01*sin(theta_3-theta_1) + 0.01*sin(theta_3-theta_2) + 0.03*sin(theta_4-theta_0) - 0.06*sin(theta_4-theta_2) + 0.09*sin(theta_4-theta_3) - 0.02*kappa_01 + 0.05*kappa_04 + 0.04*kappa_10 - 0.07*kappa_12 - 0.05*kappa_13 - 0.05*kappa_20 - 0.06*kappa_24 - 0.06*kappa_34 + 0.02*kappa_40 - 0.11*kappa_42 + ... + 0.06

dkappa_02/dt = 0.04*cos(theta_1-theta_3) + 0.06*cos(theta_1-theta_4) + 0.1*cos(theta_2-theta_0) + 0.07*cos(theta_3-theta_0) + ... - 0.09*sin(theta_0-theta_2) - 0.03*sin(theta_1-theta_0) + ... - 0.02

dkappa_03/dt = -0.05*cos(theta_0-theta_1) + 0.07*cos(theta_1-theta_2) + ... + 0.29 + 0.12*exp(-0.75*sin(theta_4-theta_3))

dkappa_04/dt = -0.08*cos(theta_0-theta_1) + 0.04*cos(theta_0-theta_4) + ... + 0.17
```

(Each of the 20 kappa equations has 40-50 terms. Full output available in `/tmp/ak_output.txt`.)

</details>

## Analysis: Why Symbolic Extraction Failed

The same **library degeneracy** problem encountered in the Kuramoto-Sivashinsky experiment, but much worse here due to scale:

1. **Too many correlated features.** The lift contains sin(theta_i - theta_j), cos(theta_i - theta_j), AND kappa_ij. These are correlated in the training data (kappa values evolve in response to phase differences). The KAN distributes signal across many edges rather than concentrating it on the correct ones.

2. **Too many edges vs true terms.** The theta model has 300 edges but only ~25 true terms across all 5 equations. The kappa model has 1200 edges but only ~40 true terms. With 10x more edges than needed, the optimizer has ample room for degenerate solutions.

3. **No sparsity pressure.** Training used lamb=0.0 (no L1 regularization). Without sparsity penalty, LBFGS finds low-loss solutions that use many small contributions rather than a few large ones.

### Potential Improvements

- **L1 regularization:** May encourage sparsity, but risks attenuating true coefficients
- **Pruning before symbolic extraction:** Remove low-magnitude edges to clean up
- **Smaller lift:** Reduce feature redundancy (e.g., only include sin(theta_i - theta_j - alpha) directly if alpha is known)
- **Fewer oscillators (N=3):** Reduces edges from 1200 to 108 for kappa model

## Conclusion

**The learned model is an excellent black-box predictor** (rollout RMSE theta=0.035, kappa=0.008, order parameter=0.0003), reproducing the adaptive Kuramoto dynamics almost perfectly over 500 integration steps. However, **symbolic extraction fails** to recover the true equation structure due to library degeneracy in the high-dimensional feature space. This represents a fundamental challenge for KANDy on systems with many coupled degrees of freedom and correlated features.
