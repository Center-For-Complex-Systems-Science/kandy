---
name: iEEG Seizure Dynamics Lessons
description: Preprocessing, determinism checks, rollout stability, saddle-focus bifurcation modeling, and failure modes for real intracranial EEG data
type: project
---

## Determinism Check Protocol
Before running KANDy on real data, ALWAYS:
1. Plot x vs x_dot scatter for each state variable
2. Compute OLS R^2 on the lifted features -> derivatives
3. If OLS R^2 < 0.3, data is too stochastic for clean symbolic extraction
4. KANDy still provides structural identification (active/zeroed edges) even at R^2 ~ 0.09
5. OLS R^2 > 0.5 is needed for auto_symbolic to find nontrivial functions
6. Per-episode R^2 can be much higher than combined R^2 (cross-episode parameter variance)

## Experiment v4: Ultra-Heavy-Smoothed Bifurcation Envelope (2026-03-12) -- BEST

### Key Results
- OLS R^2 = 0.548 on transition windows (2.2x better than v3, 6.1x better than v1)
- Per-episode OLS R^2: 0.68-0.91 (very strong within-episode determinism)
- 10s running average (5000 samples at 500 Hz) + 1 Hz + SG derivatives
- KAN R^2 = -0.001 (learns constants -- insufficient data for splines)
- OLS reveals saddle-focus terms: z2 coupling, z1 linear decay, quadratic cross-terms
- LASSO zeros dz0/dt and dz2/dt entirely; keeps 6 terms for dz1/dt

### Configuration
- Lift: 15 features (1 + z_i + z_i*z_j + z_i*(z0^2+z1^2) + z_i*z2^2)
- KAN [15, 3], grid=3, k=3, lamb=0.005, patience=30
- Savitzky-Golay (window=11, polyorder=3) at 1 Hz
- DS 500x to 1 Hz (dt=1.0s)
- Transition windows only: 85 samples total (35+25+25)

### Critical Finding: KAN Needs More Data Than OLS
- 85 samples, 15 features: OLS succeeds (15 params/eq), KAN fails (~270 spline params for 63 train)
- OLS R^2=0.55 is above the symbolic extraction threshold, but KAN cannot exploit it
- Bottleneck shifted from signal quality to data quantity

## Experiment v3: Saddle-Focus (2026-03-12) -- Previous Best
- OLS R^2 = 0.25 on transition windows (3x better than v1, 50x better than v2)
- 3s running average, 10 Hz, SmoothedFiniteDifference
- 750 transition samples

## Experiment v2: Alpha-Band Phase (2026-03-11) -- FAILED
- OLS R^2 = 0.005, phase diffs at 50 Hz are 99.5% carrier noise
- Seizure onset detection via Kuramoto order parameter SUCCEEDED

## Experiment v1: Broadband Amplitude (2026-03-11) -- Baseline
- OLS R^2 = 0.09, 2 Hz too slow
- Rollout blowup fixed by include_bias=True

## Working Configuration: Best So Far (v6 for structural ID, v4 for OLS R^2)
- **For structural ID (KAN):** Mode 0 only, alpha 8-13 Hz, 4s avg, 5 Hz, grid=3, lamb=0.001
  - 1422 samples, 8 features, all edges active, R^2=0.126
- **For OLS R^2:** v4 config (10s avg, 1 Hz, transition windows, saddle-focus lift)
  - OLS R^2=0.55, per-ep 0.68-0.91, but only 85 samples (KAN fails)

## Critical Fix: Rollout Blowup
**Problem:** Purely polynomial ODE without constant/linear terms -> finite-time blowup.
**Solution:** Include constant feature (1) in lift. Enables bounded Stuart-Landau-type dynamics.
Works across all experiments: max |state| = 1.16 (v1), 2.07 (v2), 2.46 (v3), 1.58 (v4).

## Rollout Fine-Tuning DOES NOT WORK on Stochastic Data
Phase 2 rollout fine-tuning with Adam optimizer destroys spline shapes because:
- Optimal strategy for minimizing rollout loss on stochastic data is near-zero output
- Adam flattens all edges to minimize trajectory error
- The fix is architectural (constant feature), not numerical (fine-tuning)

## Key Failure Modes
- Raw 2-40 Hz + SVD at 500 Hz: derivatives 20-40x signal -> KAN learns nothing
- Alpha envelope at 50 Hz: derivatives ~5x signal -> carrier noise dominates
- Phase diffs at 50 Hz: OLS R^2 = 0.005, carrier noise only
- High regularization (lamb=0.01+) on noisy data: zeroes all real signal
- Low regularization (lamb=0.005) on noisy data: no sparsity (44/45 edges active) [v3]
- lamb=0 on 85 samples: massive overfitting (R^2=-51, nonsensical trig formulas)
- auto_symbolic_with_costs + POLY_LIB_CHEAP: returns R^2=0.0 for all edges on noisy data
- auto_symbolic on 80 edges with 2048 samples: hangs. Use N_SYM=512
- Phase 2 rollout fine-tuning: destroys learned spline structure on noisy data
- No constant feature: finite-time blowup in rollout
- 85 samples with 15-feature lift: KAN has ~270 spline params for 63 training samples -> learns constants
- LASSO too aggressive on small data: zeros entire equations

## General Lessons for Real/Noisy Data
- Always include constant feature in lift for real data
- Purely polynomial ODEs generically have finite-time blowup
- Linear damping/constant terms are the structural mechanism for bounded dynamics
- Clipping is a safety net, not a primary stability mechanism
- Phase dynamics at high sampling rate are dominated by carrier frequency
- Use N_SYM=512 (not 2048) for auto_symbolic on large KANs
- Transition windows concentrate deterministic signal (onset +/- 15-10s is good)
- SmoothedFiniteDifference (PySINDy) works well for noisy derivative estimation
- OLS R^2 > 0.5 is needed for KAN symbolic extraction to find structure
- Ultra-heavy smoothing (10s running avg) dramatically improves deterministic fraction
- Per-episode OLS R^2 can be 1.5-2x higher than combined (cross-episode parameter variance)
- KAN splines need more data than OLS: ~10x samples-to-parameters ratio minimum
- When KAN fails due to data scarcity, use OLS as primary discovery + LASSO for sparsification
- **Per-mode fitting >> pooled fitting** when SVD modes have heterogeneous dynamics (v5 lesson)
- Pooled models with shared normalization + regularization collapse to constants on mixed-scale data
- 2-jet embedding (x, y=x_dot, target=x_ddot) halves output dim for second-order ODEs
- Pre-computed features with identity lift cleanly handle time-dependent features (sin/cos)
- ReLU threshold estimated from data (mean mode value at onset) is a reasonable starting point
- Even at OLS R^2 ~ 0.1, per-mode KAN can achieve structural identification (correct active/zeroed edges)
- **grid=3, lamb=0.001 is the sweet spot** for ~1400 samples with 8 features (v6 finding)
- Lower grid = fewer spline params = less overfitting = more stable structural ID
- **auto_symbolic DESTRUCTIVELY replaces splines**: always do edge analysis BEFORE calling it
- auto_symbolic fails entirely when individual spline R^2 < 0.8 (all edges zeroed)
- Select models by **structural criteria** (most active edges), not just R^2
- KANDy can exceed OLS R^2 on noisy data by capturing nonlinear spline structure
- sin/cos features show near-linear splines (R^2_lin > 0.9) confirming linear forcing model

## Experiment v6: Mode 0 Focused Duffing-ReLU with Hyperparameter Sweep (2026-03-12) -- CURRENT BEST KAN

### Key Results
- Mode 0 only, hyperparameter sweep over grid={3,5,7} x lamb={0,1e-4,1e-3,5e-3,1e-2}
- **grid=3, lamb=0.001: ALL 8/8 edges active, R^2=0.126, NRMSE=1.04**
- grid=7, lamb=0.0: highest R^2=0.137 but only 4/8 edges active
- KANDy rollout beats OLS 2.1x (NRMSE 1.04 vs 2.19)
- KANDy R^2=0.126 exceeds OLS R^2=0.115 (captures nonlinear structure)
- auto_symbolic fails: R^2 too low for symbolic function matching on individual splines
- sin/cos edges are nearly linear (R^2_lin=0.90, 0.70), confirming linear forcing
- x^3, ReLU, x*y edges show clear nonlinear spline shapes

### Critical Findings
1. **lamb=0.001 is the sweet spot for noisy data** with ~1400 samples and 8 features
2. **grid=3 optimal for structural ID**: fewer params -> less overfitting -> more edges active
3. **Edge analysis BEFORE auto_symbolic is essential**: auto_symbolic destructively replaces
   splines with symbolic functions (all zeros on noisy data), destroying the model
4. **Select by structural criteria** (most active edges) not just R^2 for equation discovery
5. **auto_symbolic fundamentally fails at R^2 ~ 0.1**: individual spline shapes don't match
   any standard basis function above R^2=0.8 threshold. Need constrained symbolic fitting.

### Configuration
- KAN [8, 1], grid=3, k=3, lamb=0.001, steps=500, patience=50
- Mode 0 only: 1422 samples (3 episodes x 474)
- Same lift as v5

## Experiment v5: Duffing-ReLU with 2-Jet Embedding (2026-03-12)

### Key Results
- Alpha band (8-13 Hz), 4s avg, 5 Hz, full episodes (not just transitions)
- Pooled OLS R^2 = 0.095 (all modes combined -- too noisy for pooled KAN)
- Pooled KAN R^2 = -0.003 (learned constant, all edges zeroed)
- **Per-mode KAN Mode 0: R^2 = 0.135, correctly activates all 7 Duffing-ReLU terms**
  - x^3 (range=5.1), x*y (3.2), y (2.2), ReLU(x-theta) (1.5), x (0.7), sin/cos (~0.3)
  - Only x^2 zeroed (consistent with LASSO which also drops it)
- LASSO: R^2=0.09, 4 terms (x, x^2, x^3 + intercept -- sparse but misses y, xy, ReLU, sin)

### Critical Finding: Per-Mode Fitting Recovers Structure That Pooling Destroys
- Pooling heterogeneous modes with shared normalization and regularization -> constant
- Per-mode KAN for Mode 0 finds all 7 expected terms with good structural identification
- Modes 1 and 2 have lower R^2 (0.097, 0.022) -- dynamics are weaker in subordinate modes
- ReLU feature activates ONLY for Mode 0 (only leading mode reaches theta=2.4)

### Configuration
- Lift: 8 features [x, y, x^2, x^3, x*y, ReLU(x-theta), sin(omega*t), cos(omega*t)]
- KAN [8, 1], grid=5, k=3, lamb=0.01, patience=20
- 2-jet embedding: x = SVD mode, y = x_dot, target = x_ddot
- Identity lift (features pre-computed including time-dependent sin/cos)
- theta=2.40 (mean mode-0 at onset), omega=0.18 rad/s (35s period from FFT)
- 4266 total samples (3 modes x 3 eps x ~474 each)

### Lesson: 2-Jet Embedding Elegantly Handles Second-Order Systems
Since x_dot = y by construction, KANDy only needs to learn the x_ddot equation.
This halves the output dimension and the spline parameter count vs learning both equations.

## OLS R^2 Comparison Across Approaches
| Approach | OLS R^2 | Notes |
|----------|---------|-------|
| Ultra-smooth 10s avg (1 Hz, transition) | 0.548 | Best: 55% deterministic |
| Per-episode (10s avg, individual) | 0.68-0.91 | Very strong within-episode |
| Saddle-focus 10 Hz (transition) | 0.25 | Previous best |
| Duffing-ReLU alpha 5 Hz (all data, per-mode 0) | 0.115 | Per-mode better |
| Duffing-ReLU alpha 5 Hz (all data, pooled) | 0.095 | Pooling hurts |
| Saddle-focus 10 Hz (all data) | 0.08 | Baseline dilutes signal |
| Broadband amplitude (2 Hz) | 0.09 | Second best (pre-v4) |
| Alpha phase diffs (50 Hz) | 0.005 | Dominated by carrier noise |
