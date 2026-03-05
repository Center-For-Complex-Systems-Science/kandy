I have all context needed to write the complete formal peer review. Here it is:

---

# `outputs/review.md`

```markdown
# Peer Review — KANDy: Koopman-Augmented Neural Dictionary Dynamics

**Venue target:** Journal of Computational Physics (JCP) / SIAM Journal on Applied
Dynamical Systems (SIADS)
**Submission type:** Research Article
**Reviewer:** Anonymous Referee
**Recommendation:** Major Revision

---

## Contribution Summary

KANDy proposes a system-identification framework of the form **x_dot = A · Ψ(φ(x))**,
where φ is a user-specified Koopman lift, Ψ is the separable spline map of a
*single-layer* Kolmogorov–Arnold Network (KAN), and A is a linear mixing matrix
extracted analytically from output weights. The central theoretical claim — the
**Zero-Set Corollary** — asserts that bilinear terms such as x·y cannot be produced
by any sum of univariate functions of raw inputs, forcing all cross-products into the
lift φ and justifying the single-layer constraint. Benchmarks on the Lorenz attractor
and related systems demonstrate trajectory reconstruction and partial symbolic
recovery of governing equations. The package `kandy` exposes a clean API
(`PolynomialLift`, `KANDy.fit`, `get_A`, `get_formula`, `rollout`) that enforces
the single-layer constraint programmatically. The work occupies a productive niche
between sparse regression (SINDy), linear operator theory (EDMD), and black-box
neural ODEs, and the architectural constraint is rigorously motivated.

---

## Major Concerns

**M1. Baselines are absent — the paper cannot be accepted without them.**

The paper makes no quantitative comparison against SINDy (Brunton et al. 2016),
EDMD (Williams et al. 2015), or a neural ODE (Chen et al. 2018) on the same
benchmarks and metrics (RMSE, NRMSE, R², rollout horizon). Without these, claims
of superiority are unsubstantiated. The three methods occupy different design
philosophies and the paper's introduction rightly distinguishes them; the experiments
must reflect this.

*Remedy:* Add a Table 1 with columns [Method | Lorenz NRMSE | Hopf NRMSE |
Hénon NRMSE | #parameters | train time]. Include: SINDy (degree-2 library + LASSO,
threshold swept by cross-validation), EDMD (same polynomial library, DMD readout
for Ψ_dot = K Ψ), and a single-hidden-layer neural ODE (64 units, tanh) as black-box
baseline. Use identical train/test splits and initial conditions across all methods.

---

**M2. The Zero-Set Corollary proof is stated but not formally proved in the main text.**

The claim that "x·y is not additively separable — it cannot be written as
f(x) + g(y) for any univariate f, g" is correct and classical (follows from taking
mixed partial derivatives: ∂²(x·y)/∂x∂y = 1 ≠ 0 while ∂²(f(x)+g(y))/∂x∂y = 0).
However, the paper does not present this argument explicitly, does not cite the
Hilbert-space separability literature, and does not state precisely what function
class the splines belong to (L², C^k, etc.). The corollary is used to justify a
fundamental design choice — it must be proved to the standards of a mathematical
journal.

*Remedy:* Add a Theorem/Proof environment. State: "Let f: R^m → R be defined by
f(x) = Σ_{i=1}^{m} g_i(x_i) for measurable g_i. Then f cannot represent x_i · x_j
for i ≠ j." Proof: differentiate. Cite Kolmogorov–Arnold representation theorem
(1957) for context and distinguish from the general KA theorem (which applies to
*deep* compositions). Then state Corollary: a *single-layer* KAN is in this
additively-separable class, and any bilinear term must therefore be pre-computed in φ.

---

**M3. Noise robustness is not tested.**

Real dynamical systems data is noisy. SINDy's literature extensively studies
LASSO regularisation under measurement noise. KANDy's behaviour under noise is
entirely unknown from the submitted manuscript: Does x_dot computed by finite
differences degrade the spline fit? Does LBFGS overfit to noise? Does the symbolic
extraction of A break down?

*Remedy:* Add a noise experiment on Lorenz: inject Gaussian noise
σ_noise ∈ {0, 0.01, 0.05, 0.1} × std(X) to the trajectory. Report NRMSE of rollout
and the Frobenius error ||A_recovered − A_true||_F / ||A_true||_F at each noise
level. Compare the degradation curve against SINDy+LASSO and EDMD. This is a
critical practical question for the method's applicability.

---

**M4. The lift design requires undisclosed domain knowledge — this is a fundamental
limitation that must be prominently stated and quantified.**

The correct lift for Lorenz is φ = [x, y, z, x², x·y, x·z, y², y·z, z²] only
because the reviewer (or algorithm designer) already knows that the Lorenz RHS
contains x·y and x·z. For an unknown system, the user must choose φ without
knowing which cross-terms are needed. The paper does not address: (a) what happens
when φ is over-complete (too many monomials); (b) what happens when φ is
under-complete (missing a required cross-term); or (c) whether there is an automated
procedure for lift selection (e.g., iterative greedy addition of monomials until
fit improves).

*Remedy:* (a) Add an ablation table: KANDy with correct minimal lift vs. degree-1
lift (too small) vs. degree-3 full polynomial lift (over-complete) on Lorenz.
Report NRMSE and training time. (b) State explicitly in the Limitations section:
"The choice of φ currently requires knowledge of the expected nonlinearity structure.
For purely data-driven lift selection, one may use an over-complete polynomial
library of degree d, accepting increased m and compute cost." (c) Note that
automated lift selection is future work.

---

**M5. Symbolic recovery success conditions are not characterised.**

The paper calls `model.auto_symbolic()` and `model.symbolic_formula()` and reports
"exact recovery" for simple cases. However: (a) the conditions under which
auto_symbolic succeeds are not stated (it depends on PyKAN's internal fitting
thresholds); (b) the paper does not report whether the recovered symbolic A matches
the known Lorenz coefficients (σ=10, ρ=28, β=8/3) to within a stated tolerance;
(c) it is unclear whether the reported "symbolic" formulas are exact or
approximations.

*Remedy:* Report ||A_KANDy − A_true||_∞ explicitly for each benchmark where
the ground truth is known. For Lorenz: print the full 3×9 A matrix and compare
entry-by-entry against the known values (A[0,0]=−10, A[0,1]=+10, A[1,0]=+28,
A[1,2]=−1, A[1,5]=−1, A[2,4]=+1, A[2,2]=−8/3). State a threshold below which
coefficients are zeroed (sparsification criterion). Define "exact" recovery
precisely (e.g., ||A_KANDy − A_true||_∞ < 0.01).

---

## Minor Concerns

**m1. The relationship to EDMD is incompletely described.**

The paper correctly distinguishes KANDy (models x_dot = A · Ψ) from EDMD (models
Ψ_dot = K · Ψ). However, it does not note that EDMD in the continuous-time
formulation (gEDMD, Klus et al. 2020) also produces equations of the form
dΨ/dt = K Ψ, which for Ψ = φ (the lift itself) gives d(φ(x))/dt = K φ(x). A
careful reader will ask: is KANDy's A · Ψ(φ(x)) in the state space equivalent
to a Koopman operator approximation in lifted space? The answer is nuanced (KANDy
operates directly on x_dot, not on lifted observables) and deserves one paragraph.

**m2. Training details are inconsistent across experiments.**

The Lorenz script uses LBFGS with 200 steps and lamb=1e-4. The Hopf notebook
(audit finding) appears to use different optimiser settings. A table of
hyperparameters per experiment (optimiser, steps, grid, k, seed) should appear
in an appendix.

**m3. The rollout integration method is not standardised.**

Some experiments use an Euler step for integration; the Lorenz script correctly
uses RK4. Euler integration introduces O(dt) error per step and should not be
used in any final experiment. All rollout figures should use RK4 (or scipy
solve_ivp with KANDy's RHS) with a stated tolerance.

**m4. Figure quality is inconsistent.**

From the audit, most experiment figures are either not saved (plt.show() only),
saved without DPI specification, or saved as PNG only without PDF. A manuscript
submitted to JCP must include vector graphics (PDF/EPS) for all phase portraits
and symbolic equation panels. The `save_figure` helper in the package correctly
saves PNG (300 dpi) + PDF — this must be applied to every experiment figure.

**m5. The Hénon map is a discrete map, not a flow.**

Sections of the text conflate the continuous-time algorithm (x_dot = A · Ψ) with
the discrete-time Hénon application (x_{n+1} = A · Ψ). These are formally
different: in the discrete case there is no derivative, and the Koopman operator
acts on the map's iterate rather than its generator. The paper should either
(a) maintain a clean separation between continuous (ODE) and discrete (map)
formulations, or (b) introduce the map version formally as
x_{n+1} = A · Ψ(φ(x_n)) and prove/state analogous properties.

**m6. The `PolynomialLift` in the package lazily infers state dimension.**

`output_dim` raises `RuntimeError` until `__call__` has been called once. This
will cause errors if a user constructs `KANDy(lift=PolynomialLift(degree=2))`
and then calls `model.get_A()` before fitting. The dimension should be injectable
at construction time as an optional parameter `input_dim: Optional[int] = None`.

**m7. The package has no version pin for PyKAN.**

PyKAN's internal API (act_fun, spline_postacts, symbolic_fun layout) has changed
multiple times across 0.x releases. The A-extraction logic uses four different
fallback access patterns, which is a symptom of API instability. The pyproject.toml
should pin `pykan >= 0.2.0, < 0.3.0` (or the specific tested version) and this
should be documented.

---

## Comments on Proofs and Experimental Figures

### Bilinear Obstruction (Zero-Set Corollary)

The informal argument is correct: if h(x, y) = f(x) + g(y) then
∂²h/∂x∂y ≡ 0, but ∂²(x·y)/∂x∂y = 1, contradiction. This is
standard and can be tightened: the set of additively separable functions forms
a closed linear subspace of L²(Ω) with infinite codimension; bilinear monomials
lie outside it. The proof should be stated as a lemma with explicit function-space
assumptions (e.g., φ_i ∈ C²(Ω)). The current draft relies on the reader's
intuition rather than a proof.

The extension to deep KANs is also correct: a two-layer KAN can represent x·y
(compose a parabola with an affine function of x+y), so restricting to one layer
is necessary, not arbitrary. This point should be illustrated with a concrete
counterexample showing that width=[2,4,1] KAN can fit x·y from raw [x,y] inputs
while width=[2,1] cannot.

### Non-Injectivity Argument

The claim that the Koopman lift φ resolves non-injectivity of scalar observations
is attributed to the Takens embedding theorem (1981) in the delay-embedding case.
This is correct for generic smooth systems with embedding dimension 2d+1. However,
the polynomial lift used for Lorenz (φ: R³→R⁹) is not a delay embedding and the
injectivity argument is different — it relies on the map φ being injective on the
attractor, which should be verified (at minimum numerically) for each benchmark.
If two distinct attractor points x ≠ y satisfy φ(x) = φ(y), the method will fail
silently. The paper should either prove injectivity of the polynomial lift on the
Lorenz attractor or test it numerically.

### Lorenz Experiment Figures

Based on the audit and the provided lorenz_kandy.py script:

- **Loss curve (lorenz_loss.png/pdf):** Correct format (semilogy, train+val, 300
  dpi PNG + PDF). ✓
- **Trajectory overlay (lorenz_trajectory.png/pdf):** Three-panel time series with
  true (blue) vs. KANDy rollout (red dashed). This is the right visualisation but
  the manuscript should report the *time to first large deviation* (Lyapunov horizon)
  as a scalar metric, not just the visual overlay.
- **Phase portrait (lorenz_phase_portrait.png/pdf):** Side-by-side 3D projections.
  The reviewer notes that overlaying both attractors in a single panel (using
  transparency) is more informative than side-by-side for assessing attractor
  geometry preservation.
- **Error plot (lorenz_error.png/pdf):** Semilogy of pointwise absolute error per
  variable. This is good. The reviewer requests that the Lyapunov exponent of the
  true system (λ₁ ≈ 0.906 for standard parameters) be indicated as a reference
  slope: the error should grow as ~exp(λ₁ t) before rollout diverges.
- **A matrix table:** Must appear as a figure or table in the manuscript, not only
  in console output. The comparison between recovered A and ground-truth A
  (σ=10, ρ=28, β=8/3) is the single most compelling result and should be the
  centrepiece of the Lorenz section.

---

## Recommendation

**Major Revision.**

The core idea is original, theoretically motivated, and cleanly implemented.
The Zero-Set Corollary is a genuine insight that differentiates KANDy from both
SINDy (no KAN, no Koopman lift requirement) and EDMD (operator acts on Ψ, not x_dot).
The single-layer constraint is correctly enforced in the package API. However, the
paper cannot be accepted in its current form because: (1) quantitative baselines
against SINDy, EDMD, and neural ODEs are absent; (2) the proof of the bilinear
obstruction is informal; and (3) noise robustness is completely untested. These are
not cosmetic issues — they determine whether the method is competitive and practically
useful.

---

## Top-3 Highest-Impact Action Items

**Action 1 — Add SINDy/EDMD/neural-ODE comparison table (M1).**
This is the single highest-priority change. Without quantitative baselines on
identical data splits, the paper cannot make claims of advancement. Implement
SINDy with a degree-2 polynomial library + LASSO (PySINDy) and EDMD with the same
library (PyDMD) on all benchmarks. Report NRMSE,