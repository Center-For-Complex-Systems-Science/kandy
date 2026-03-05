# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
crewai install        # or: uv sync

# Run the full research crew
crewai run            # or: uv run kandy

# Target a specific dynamical system (edit INPUTS in main.py first)
# Supported: Lorenz, Henon, Burgers, Burgers-Fourier, Kuramoto-Sivashinsky, Navier-Stokes, Hopf

# Other entry points
uv run train <n_iterations> <output_filename>
uv run replay <task_id>
uv run test <n_iterations> <eval_llm>
```

## Algorithm

KANDy (Kolmogorov-Arnold Networks for Dynamics) is a reimplementation of SINDy where
sparse regression is replaced by a KAN and inputs are Koopman-lifted coordinates.

**The model:** `x_dot = A * Psi(phi(x))`
- `phi(x) = theta` — Koopman lift (user-designed; encodes cross-terms like x*y explicitly)
- `Psi(theta)` — separable spline map, `psi_i(theta_i)`, learned by a **single-layer** KAN
- `A` — learned linear mixing matrix (n × m)

**Critical constraints:**
- The KAN is always `width=[m, n]` (one hidden layer). Standard deep KANs cannot
  represent bilinear terms like x*y from raw inputs (proven by zero-set corollary).
- Cross-interaction terms must be in `phi`, not learned from raw states.
- Non-injective observations require the lift to expand the observable space.

**KANDy pipeline:**
1. Design `phi(x)` (include ALL cross-terms for the target system)
2. Apply lift: `theta = phi(x)` for all trajectory snapshots
3. Train: `KAN(width=[m, n], grid, k).fit({train_input: theta, train_label: x_dot}, ...)`
4. Extract `A` (output weights) and `psi_i` (splines) symbolically via `auto_symbolic()`
5. Validate: autoregressive rollout with RK4, compute RMSE/NRMSE

### Research code (`research_code/`)
Raw experimental notebooks and scripts. Each implements the KANDy pipeline:
- `henon.py` — Hénon map (discrete, 2D; lift includes x²)
- `hopf.ipynb` — Hopf fibration (S³ → S², topological)
- `Lorenz (3).ipynb` — Lorenz ODE (lift must include xy, xz)
- `Inviscid_Burgers (1).ipynb` — Burgers equation (lift includes u, u_x, u²)
- `Inviscd-Burgers-fourier-mode-ics.ipynb` — Burgers with Fourier-mode ICs
- `Kuramoto–Sivashinsky (1).ipynb` — KS PDE (lift includes u, u_x, u_xx, u_xxxx)
- `Navier-Stokes.ipynb` — 3D incompressible N-S

### 2. crewAI automation (`src/kandy/`)
A multi-agent system that automates research tasks. Uses **hierarchical process**.

**Agent hierarchy:**
- `phd_advisor` — Manager agent (not decorated with `@agent`); directs the crew
- `phd_student` — Cleans existing experiments and generates new ones
- `professional_coder` — Designs and builds the pip-installable `kandy` package
- `journal_reviewer` — Produces a formal referee report on results

**Task flow** (PhD Advisor delegates dynamically):
1. `clean_experimental_results` → phd_student → `outputs/audit_report.md`
2. `generate_new_results` → phd_student → `outputs/experiment_{system}.py`
3. `develop_package_code` → professional_coder → `outputs/package_design.md`
4. `review_results` → journal_reviewer → `outputs/review.md`

**Key files:**
- `src/kandy/config/agents.yaml` — Agent role, goal, backstory
- `src/kandy/config/tasks.yaml` — Task descriptions, expected outputs, context dependencies
- `src/kandy/crew.py` — Wires agents and tasks; `phd_advisor()` is passed as `manager_agent`
- `src/kandy/main.py` — Entry points; `INPUTS['system']` controls the target benchmark

### Environment
- `.env`: `MODEL` and `ANTHROPIC_API_KEY` (Anthropic Claude)
- Python 3.10–3.13, package manager: **uv**
- Core research dependencies: `pykan`, `torch`, `scipy`, `sympy`, `matplotlib`
- crewAI dependency: `crewai[tools]==1.9.3`
