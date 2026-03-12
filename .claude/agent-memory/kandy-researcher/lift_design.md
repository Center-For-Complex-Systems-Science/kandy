---
name: Lift Design Patterns
description: Successful lift configurations for different dynamical system types, including cross-terms and feature selection
type: project
---

## Core Principle
- Cross-terms (x*y, u*u_x) MUST be in the lift, NOT learned by the KAN
- Do NOT include squared versions of features already in the library (causes signal splitting)
- Keep library minimal: only terms that appear in the target equation + raw states

## ODE Systems
| System | Lift Features | Width | Notes |
|---|---|---|---|
| Lorenz | x, y, z, xy, xz | [5, 3] | Must include xy, xz cross-terms |
| Ikeda Map | x, y, cos(t), sin(t) where t=0.4-6/(1+x^2+y^2) | [4, 2] | Pre-compute trig of phase |
| Kuramoto | sin(theta_i - theta_j), cos(theta_i - theta_j), kappa_ij | varies | Pairwise phase differences |

## PDE Systems
| System | Lift Features | Width | Notes |
|---|---|---|---|
| Burgers (sinusoidal IC) | u, u_x, d(u^2/2)/dx | [3, 1] | Conservation form required; superbee limiter |
| Burgers (Fourier IC) | u, u_x, d(u^2/2)/dx | [3, 1] | Conservation form + minmod only working config |
| Kuramoto-Sivashinsky | u, u_x, u_xx, u_xxxx, u*u_x, u*u_xx | [6, 1] | Spectral derivatives; no squared features |

## Real/Noisy Data
| System | Lift Features | Width | Notes |
|---|---|---|---|
| iEEG saddle-focus v4 (3 SVD modes, 10s smooth, 1 Hz) | 1, z_i, z_i*z_j, z_i*(z0^2+z1^2), z_i*z2^2 | [15, 3] | Best: OLS R^2=0.55 (per-ep 0.68-0.91); KAN fails at 85 samples; OLS discovers structure |
| iEEG amplitude (3 SVD modes) | 1, z0, z1, z2, z0^2, z0*z1, z0*z2, z1^2, z1*z2, z2^2 | [10, 3] | include_bias=True critical for stability; OLS R^2=0.09 |
| iEEG phase diffs (3 SVD modes) | 1, z_i, sin(z_i), cos(z_i), sin(z_i-z_j), cos(z_i-z_j) | [16, 3] | OLS R^2=0.005, too stochastic; Phase at 50 Hz is noise |

## Second-Order ODEs (2-Jet Embedding)
| System | Lift Features | Width | Notes |
|---|---|---|---|
| iEEG Duffing-ReLU v6 (alpha, 4s avg, 5 Hz, Mode 0 only) | x, y, x^2, x^3, x*y, ReLU(x-theta), sin(omega*t), cos(omega*t) | [8, 1] | grid=3, lamb=0.001; 8/8 edges active, R^2=0.126, rollout NRMSE=1.04 (2x better than OLS) |

## Anti-Patterns
- Including u_xx AND u_xx^2: causes KAN to split signal across both edges (KS lesson)
- Purely quadratic lift without constant/linear: finite-time blowup on real data
- Product form (u*u_x) for Burgers with Fourier IC: all edges zero (use conservation form)
- Spectral derivatives for shocked solutions: Gibbs oscillations -> wrong equations
- Kuramoto-type lift on SVD modes of phases: sin/cos of SVD projections != sin/cos of raw phase diffs
- Alpha-band phase diffs at 50 Hz: carrier-frequency noise dominates, not coupling dynamics
- For saddle-focus: include constant for stability, z_i*(z0^2+z1^2) for amplitude saturation
- Transition windows around bifurcation onset concentrate deterministic signal (3x R^2 improvement)
- Ultra-heavy smoothing (10s avg) at 1 Hz: OLS R^2 jumps from 0.25 to 0.55 (bifurcation envelope)
- KAN splines need ~10x samples-to-parameters ratio; 85 samples with grid=3 -> KAN learns constants
- When KAN fails on small data, use OLS + LASSO as primary discovery (OLS has 15 params vs KAN's ~270)
