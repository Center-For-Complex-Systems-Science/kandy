#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip -q install torch torchdiffeq pykan numpy scipy matplotlib sympy')


# In[2]:


import numpy as np
import torch
import copy
import random
from functools import reduce
from torch.optim import Optimizer
import kan.LBFGS as LBFGS


SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Make CUDA deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def fit(
    model, dataset, opt="LBFGS", steps=100, log=1,
    lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0.,
    update_grid=True, grid_update_num=10, loss_fn=None, lr=1.,
    start_grid_update_step=-1, stop_grid_update_step=50, batch=-1, metrics=None,
    in_vars=None, out_vars=None, beta=3, singularity_avoiding=False, y_th=1000.,
    reg_metric='edge_forward_spline_n', display_metrics=None, k=1,

    # ---- NEW: integrated rollout loss ------------------------------------------
    rollout_weight=0.0,          # set >0 to enable integration loss
    rollout_horizon=None,        # int number of steps (<= T-1). None => use full T-1
    traj_batch=-1,               # how many trajectories per step (-1 => all)
    dynamics_fn=None,            # optional: f(state)->dstate/dt. defaults to model.forward(state)
    integrator="rk4",            # "euler" or "rk4"
):
    """
    Adds an optional differentiable integration loss:
        if dataset contains 'train_traj' and 'train_t' and rollout_weight>0,
        we integrate ds/dt = f_theta(s) and match predicted states to train_traj.

    Notes:
    - This assumes the model maps STATE -> DERIVATIVE for the rollout term.
      If your model needs feature transforms (Theta, normalization, etc.),
      pass a custom dynamics_fn(state) that does that and returns dstate/dt.
    """

    assert k >= 1, "k must be >= 1"

    # -------------------------------------------------------------------------
    # Original k-step forward (kept for backward compatibility)
    # -------------------------------------------------------------------------
    def k_step_forward(inputs):
        state = inputs
        for _ in range(k):
            state = model.forward(state, singularity_avoiding=singularity_avoiding, y_th=y_th)
        return state

    # -------------------------------------------------------------------------
    # NEW: dynamics function for ODE rollout
    # -------------------------------------------------------------------------
    if dynamics_fn is None:
        def dynamics_fn(state):
            # default: treat model.forward(state) as f_theta(state)
            return model.forward(state, singularity_avoiding=singularity_avoiding, y_th=y_th)

    # Differentiable integrators (batch-safe)
    def euler_step(s, dt):
        return s + dt * dynamics_fn(s)

    def rk4_step(s, dt):
        k1 = dynamics_fn(s)
        k2 = dynamics_fn(s + 0.5 * dt * k1)
        k3 = dynamics_fn(s + 0.5 * dt * k2)
        k4 = dynamics_fn(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate_states(s0, t):
        """
        s0: (B, state_dim)
        t:  (B, T) or (T,)
        returns: pred_traj (B, T, state_dim)
        """
        if t.dim() == 1:
            # broadcast to (B,T)
            t = t.unsqueeze(0).expand(s0.shape[0], -1)
        B, T = t.shape
        state = s0
        out = [state]

        H = T - 1 if rollout_horizon is None else min(rollout_horizon, T - 1)

        for i in range(H):
            dt = (t[:, i+1] - t[:, i]).unsqueeze(1)  # (B,1)
            if integrator.lower() == "euler":
                state = euler_step(state, dt)
            elif integrator.lower() == "rk4":
                state = rk4_step(state, dt)
            else:
                raise ValueError(f"Unknown integrator: {integrator}")
            out.append(state)

        # If horizon < T-1, we only return the integrated prefix
        pred = torch.stack(out, dim=1)  # (B, H+1, state_dim)
        return pred

    # -------------------------------------------------------------------------
    # Symbolics & activation saving logic (unchanged)
    # -------------------------------------------------------------------------
    if lamb > 0. and not model.save_act:
        print("setting lamb=0. If you want lamb>0, set model.save_act=True")
    old_save_act, old_symbolic_enabled = model.disable_symbolic_in_fit(lamb)

    # -------------------------------------------------------------------------
    # Loss fn
    # -------------------------------------------------------------------------
    if loss_fn is None:
        loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
    else:
        loss_fn = loss_fn_eval = loss_fn

    # -------------------------------------------------------------------------
    # Grid update frequency
    # -------------------------------------------------------------------------
    grid_update_freq = max(1, int(stop_grid_update_step / grid_update_num))

    # -------------------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------------------
    if opt == "Adam":
        optimizer = torch.optim.Adam(model.get_params(), lr=lr)
    else:
        optimizer = LBFGS(
            model.get_params(), lr=lr, history_size=10,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32
        )

    # -------------------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------------------
    results = dict(train_loss=[], test_loss=[], reg=[], rollout_train_loss=[], rollout_test_loss=[])
    if metrics is not None:
        for m in metrics:
            results[m.__name__] = []

    # -------------------------------------------------------------------------
    # Batching for derivative supervision (unchanged)
    # -------------------------------------------------------------------------
    n_train = dataset['train_input'].shape[0]
    n_test  = dataset['test_input'].shape[0]
    batch_size      = n_train if batch == -1 or batch > n_train else batch
    batch_size_test = n_test  if batch == -1 or batch > n_test  else batch

    # -------------------------------------------------------------------------
    # Batching for trajectories (new)
    # -------------------------------------------------------------------------
    has_train_traj = ('train_traj' in dataset) and ('train_t' in dataset)
    has_test_traj  = ('test_traj' in dataset) and ('test_t' in dataset)

    if has_train_traj:
        n_traj_train = dataset['train_traj'].shape[0]
        traj_batch_size = n_traj_train if traj_batch == -1 or traj_batch > n_traj_train else traj_batch
    else:
        n_traj_train = 0
        traj_batch_size = 0

    if has_test_traj:
        n_traj_test = dataset['test_traj'].shape[0]
        traj_batch_size_test = n_traj_test if traj_batch == -1 or traj_batch > n_traj_test else traj_batch
    else:
        n_traj_test = 0
        traj_batch_size_test = 0

    global train_loss, reg_, rollout_train_loss

    # -------------------------------------------------------------------------
    # Helper to compute rollout loss on a batch of trajectories
    # -------------------------------------------------------------------------
    def rollout_loss_on_batch(traj, t):
        """
        traj: (B, T, state_dim) true
        t:    (T,) or (B, T)
        """
        s0 = traj[:, 0, :]                  # (B, state_dim)
        pred = integrate_states(s0, t)      # (B, H+1, state_dim)
        true = traj[:, :pred.shape[1], :]   # align horizons
        return loss_fn(pred, true)

    # -------------------------------------------------------------------------
    # LBFGS closure (updated to include rollout term)
    # -------------------------------------------------------------------------
    def closure():
        global train_loss, reg_, rollout_train_loss
        optimizer.zero_grad()

        # --- derivative supervision (your current behavior)
        pred = k_step_forward(dataset['train_input'][train_id])
        train_loss = loss_fn(pred, dataset['train_label'][train_id])

        # --- integrated rollout loss (new)
        if rollout_weight > 0.0 and has_train_traj:
            traj = dataset['train_traj'][traj_id]
            tt   = dataset['train_t']
            rollout_train_loss = rollout_loss_on_batch(traj, tt)
        else:
            rollout_train_loss = torch.tensor(0.0, device=pred.device)

        # --- reg (unchanged)
        if model.save_act:
            if reg_metric == "edge_backward": model.attribute()
            if reg_metric == "node_backward": model.node_attribute()
            reg_ = model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
        else:
            reg_ = torch.tensor(0., device=pred.device)

        obj = train_loss + rollout_weight * rollout_train_loss + lamb * reg_
        obj.backward()
        return obj

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    train_ptr = 0
    test_ptr = 0
    traj_ptr = 0
    traj_test_ptr = 0

    for step in range(steps):
        # --- derivative batches (unchanged)
        train_id = np.arange(train_ptr, train_ptr + batch_size) % n_train
        train_ptr = (train_ptr + batch_size) % n_train

        test_id = np.arange(test_ptr, test_ptr + batch_size_test) % n_test
        test_ptr = (test_ptr + batch_size_test) % n_test

        # --- trajectory batches (new)
        if has_train_traj and rollout_weight > 0.0:
            traj_id = np.arange(traj_ptr, traj_ptr + traj_batch_size) % n_traj_train
            traj_ptr = (traj_ptr + traj_batch_size) % n_traj_train

        if has_test_traj and rollout_weight > 0.0:
            traj_test_id = np.arange(traj_test_ptr, traj_test_ptr + traj_batch_size_test) % n_traj_test
            traj_test_ptr = (traj_test_ptr + traj_batch_size_test) % n_traj_test

        # --- grid update (unchanged)
        if (step % grid_update_freq == 0 and
            step < stop_grid_update_step and
            step >= start_grid_update_step and
            update_grid):
            model.update_grid(dataset['train_input'][train_id])

        # --- optimizer step
        if opt == "LBFGS":
            optimizer.step(closure)
        else:
            pred = k_step_forward(dataset['train_input'][train_id])
            train_loss = loss_fn(pred, dataset['train_label'][train_id])

            if rollout_weight > 0.0 and has_train_traj:
                traj = dataset['train_traj'][traj_id]
                tt   = dataset['train_t']
                rollout_train_loss = rollout_loss_on_batch(traj, tt)
            else:
                rollout_train_loss = torch.tensor(0.0, device=pred.device)

            if model.save_act:
                if reg_metric == "edge_backward": model.attribute()
                if reg_metric == "node_backward": model.node_attribute()
                reg_ = model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0., device=pred.device)

            loss = train_loss + rollout_weight * rollout_train_loss + lamb * reg_
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- test evaluation
        with torch.no_grad():
            test_pred = k_step_forward(dataset['test_input'][test_id])
            test_loss = loss_fn_eval(test_pred, dataset['test_label'][test_id])

            if rollout_weight > 0.0 and has_test_traj:
                traj = dataset['test_traj'][traj_test_id]
                tt   = dataset['test_t']
                rollout_test_loss = rollout_loss_on_batch(traj, tt)
            else:
                rollout_test_loss = torch.tensor(0.0, device=test_pred.device)

        # --- metrics (unchanged)
        if metrics is not None:
            for m in metrics:
                results[m.__name__].append(m().item())

        # --- store
        results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
        results['rollout_train_loss'].append(torch.sqrt(rollout_train_loss).cpu().detach().numpy())
        results['rollout_test_loss'].append(torch.sqrt(rollout_test_loss).cpu().detach().numpy())
        results['reg'].append(reg_.cpu().detach().numpy())

        if step % log == 0:
            print(
                f"step {step:5d} | "
                f"train {train_loss:.10f} | test {test_loss:.10f} | "
                f"roll_train {rollout_train_loss:.10f} | roll_test {rollout_test_loss:.10f}"
            )

    model.log_history("fit")
    model.symbolic_enabled = old_symbolic_enabled
    return results

@torch.no_grad()
def get_edge_activation(model, l: int, i: int, j: int, X: torch.Tensor):
    """
    Returns x,y for edge (l,i,j) where:
      x = model.acts[l][:, i]
      y = model.spline_postacts[l][:, j, i]
    Requires a forward pass to populate activations.
    
    Usage: 
    x, y = get_edge_activation(model, 0, 1, 1, dataset['train_input'])
    plt.plot(x, y)
    """
    rank = torch.argsort(model.acts[l][:, i]).cpu().numpy()

    x = model.acts[l][:, i][rank].cpu().detach().numpy()
    y = model.spline_postacts[l][:, j, i][rank].cpu().detach().numpy()

    return x, y

def round_numbers(expr, places=3):
    repl = {}
    for a in expr.atoms(sp.Number):
        try:
            repl[a] = sp.Float(round(float(a), places))
        except Exception:
            pass
    return expr.xreplace(repl)


def flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for it in obj:
            out.extend(flatten(it))
        return out
    return [obj]

# -----------------------------
# Robust symbolic fitting helpers
# -----------------------------


def _iter_edges(model):
    """
    Iterate over all the activization layers:
    for l in range(len(width_in)-1):
      for i in range(width_in[l]):
        for j in range(width_out[l+1])
    Yields (l, i, j).
    """
    for l in range(len(model.width_in) - 1):
        for i in range(int(model.width_in[l])):
            for j in range(int(model.width_out[l + 1])):
                yield (l, i, j)

def _choose_lib_for_input(i, split_index, simple_lib, complex_lib):
    """
    
    """
    return simple_lib if i < split_index else complex_lib

def _edge_complexity_from_name(name: str) -> int:
    """
    Fallback complexity if suggest_symbolic doesn't provide c in a consistent way.
    Tries to map common tokens to rough complexity.
    """
    if name in ("0",):
        return 0
    if name in ("x",):
        return 1
    if name.startswith("x^"):
        try:
            p = int(name.split("^")[1])
            return p
        except Exception:
            return 2
    return 2

def _safe_float(x, default=-np.inf):
    try:
        return float(x)
    except Exception:
        return default

def _score_edge(r2, c, weight_simple=0.9, weight=0.05):
    """
    Higher is better.
    - r2 encourages fit quality
    - complexity penalty encourages simplicity
    weight_simple in [0,1]: higher => stronger simplicity pressure
    """
    r2 = _safe_float(r2, default=-1e9)
    c  = _safe_float(c,  default=np.nan)
    if not np.isfinite(c):
        c = 1.0
    # penalty: larger when weight_simple is large
    penalty = (weight_simple / (1e-8 + (1.0 - weight_simple + 1e-8))) * (c)
    return r2 - weight * penalty  # 0.05 is a reasonable default; change if needed

# -----------------------------
# Main: robust_auto_symbolic
# -----------------------------
def robust_auto_symbolic(
    model,
    *,
    # libraries
    simple_lib=None,
    complex_lib=None,
    input_split_index=4,          # i < 4 => simple, else complex
    # suggestion ranges
    a_range=(-10, 10),
    b_range=(-10, 10),
    # selection controls
    r2_threshold=0.90,            # min r2 to even be eligible
    weight_simple=0.90,           # favors simplicity in scoring
    keep="topk",                  # "topk" or "threshold"
    topk_edges=64,                # global budget if keep="topk"
    max_total_complexity=None,    # optional global budget; if set, overrides/works with topk
    # zeroing behavior
    set_others_to_zero=True,
    # logging
    verbose=1,
    # reproducibility / no mutation
    inplace=True,                 # if False, works on a deep-copied model and returns it
):
    """
    A more robust replacement for your auto_symbolic:

    What it changes:
    1) Collects best symbolic candidate for EVERY edge first.
    2) Applies a *global* sparsity/complexity selection:
       - keep="topk": keep top-K edges by score (score = r2 - complexity penalty)
       - keep="threshold": keep all edges meeting r2_threshold (your old behavior, but still pre-collected)
       - optionally enforce max_total_complexity budget.
    3) Fixes kept edges to their chosen symbolic; optionally zeros others.

    Returns:
      (model_out, report)
      report: dict with per-edge candidates and which ones were kept.
    """

    # --- defaults to match your libs if not provided ---
    if simple_lib is None:
        simple_lib = ['x', 'x^2']
    if complex_lib is None:
        complex_lib = ['x', 'x^2']

    if not inplace:
        model = copy.deepcopy(model)

    # 1) Gather candidates
    candidates = []
    for (l, i, j) in _iter_edges(model):
        lib_for_edge = _choose_lib_for_input(i, input_split_index, simple_lib, complex_lib)

        # KAN's suggest_symbolic accepts either a dict-lib or a list of names depending on version.
        # You used self.suggest_symbolic(..., lib=SYMBOLIC_LIB, ...) earlier.
        # Here we pass the *name list* via lib_for_edge, and rely on model's internal library mapping,
        # OR you can pass your dict directly if your KAN expects it.
        #
        # If your KAN expects a dict library, pass that dict as `simple_lib`/`complex_lib` instead.
        try:
            name, fun, r2, c = model.suggest_symbolic(
                l, i, j,
                a_range=a_range,
                b_range=b_range,
                lib=lib_for_edge,
                verbose=False,
                weight_simple=weight_simple
            )
        except TypeError:
            # some KAN versions don't take weight_simple in suggest_symbolic
            name, fun, r2, c = model.suggest_symbolic(
                l, i, j,
                a_range=a_range,
                b_range=b_range,
                lib=lib_for_edge,
                verbose=False
            )

        # normalize c if missing/weird
        if c is None or (isinstance(c, float) and not np.isfinite(c)):
            c = _edge_complexity_from_name(str(name))

        r2f = _safe_float(r2, default=-1e9)
        cf  = _safe_float(c, default=_edge_complexity_from_name(str(name)))
        score = _score_edge(r2f, cf, weight_simple=weight_simple)

        candidates.append({
            "edge": (l, i, j),
            "name": str(name),
            "r2": r2f,
            "c": cf,
            "score": score,
            "lib": lib_for_edge,
        })

    # 2) Decide which edges to keep
    # First filter by r2_threshold (eligibility)
    eligible = [c for c in candidates if c["r2"] >= float(r2_threshold) and c["name"] != "0"]

    kept = []
    if keep == "threshold":
        kept = eligible
        # optional complexity cap
        if max_total_complexity is not None:
            eligible_sorted = sorted(kept, key=lambda d: d["score"], reverse=True)
            total_c = 0.0
            kept = []
            for d in eligible_sorted:
                if total_c + d["c"] <= float(max_total_complexity):
                    kept.append(d)
                    total_c += d["c"]

    elif keep == "topk":
        eligible_sorted = sorted(eligible, key=lambda d: d["score"], reverse=True)

        if max_total_complexity is None:
            kept = eligible_sorted[:int(topk_edges)]
        else:
            # keep best-scoring edges until complexity budget is used
            total_c = 0.0
            for d in eligible_sorted:
                if len(kept) >= int(topk_edges):
                    break
                if total_c + d["c"] <= float(max_total_complexity):
                    kept.append(d)
                    total_c += d["c"]
    else:
        raise ValueError(f"keep must be 'topk' or 'threshold', got: {keep}")

    kept_set = set([tuple(d["edge"]) for d in kept])

    # 3) Apply fixes
    n_fixed = 0
    n_zeroed = 0

    for d in candidates:
        l, i, j = d["edge"]
        if (l, i, j) in kept_set:
            # fix to chosen name
            model.fix_symbolic(l, i, j, d["name"], verbose=(verbose > 1), log_history=False)
            n_fixed += 1
            if verbose >= 2:
                print(f"[KEEP] ({l},{i},{j}) name={d['name']} r2={d['r2']:.4f} c={d['c']} score={d['score']:.4f}")
        else:
            if set_others_to_zero:
                model.fix_symbolic(l, i, j, '0', verbose=(verbose > 1), log_history=False)
                n_zeroed += 1
                if verbose >= 3:
                    print(f"[ZERO] ({l},{i},{j})")

    model.log_history('robust_auto_symbolic')

    if verbose >= 1:
        total = len(candidates)
        elig  = len(eligible)
        print(f"robust_auto_symbolic: total_edges={total}, eligible(r2>={r2_threshold})={elig}, kept={len(kept)}, fixed={n_fixed}, zeroed={n_zeroed}")

    report = {
        "params": {
            "a_range": a_range,
            "b_range": b_range,
            "r2_threshold": r2_threshold,
            "weight_simple": weight_simple,
            "keep": keep,
            "topk_edges": topk_edges,
            "max_total_complexity": max_total_complexity,
            "input_split_index": input_split_index,
        },
        "candidates": candidates,
        "kept": kept,
        "kept_edges": sorted(list(kept_set)),
        "counts": {
            "total_edges": len(candidates),
            "eligible": len(eligible),
            "kept": len(kept),
            "fixed": n_fixed,
            "zeroed": n_zeroed,
        }
    }
    return model, report


# In[3]:


import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torchdiffeq import odeint
from kan import KAN

import sympy as sp
from sympy import Eq
from sympy.printing.latex import latex
from IPython.display import display, Math

# --------- reproducibility / device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(0)
np.random.seed(0)

# -----------------------
# 1) KS PDE setup (periodic)
#    u_t = -u*u_x - u_xx - u_xxxx
#    x in [0, L), periodic
# -----------------------
L = 64 #32.0
Nx = 128
dx = L / Nx
x = np.arange(0.0, L, dx)

# training horizon (after spin-up)
T_spin  = 100.0
T_train = 20.0
dt = 0.1
t_all = np.linspace(0.0, T_spin + T_train, int(round((T_spin + T_train)/dt)) + 1)

# -----------------------
# 2) Periodic finite-difference derivative matrices
# -----------------------
def periodic_shift_mat(N, k):
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        M[i, (i + k) % N] = 1.0
    return M

I   = np.eye(Nx, dtype=np.float64)
Sp  = periodic_shift_mat(Nx, +1)
Sm  = periodic_shift_mat(Nx, -1)
Spp = periodic_shift_mat(Nx, +2)
Smm = periodic_shift_mat(Nx, -2)

Dx_mat     = (Sp - Sm) / (2.0 * dx)
Dxx_mat    = (Sp - 2.0*I + Sm) / (dx**2)
Dxxxx_mat  = (Smm - 4.0*Sm + 6.0*I - 4.0*Sp + Spp) / (dx**4)

def ks_rhs_np(_t, u):
    ux    = Dx_mat @ u
    uxx   = Dxx_mat @ u
    uxxxx = Dxxxx_mat @ u
    return -(u * ux) - uxx - uxxxx

# -----------------------
# 3) Generate training data (ground truth) using SciPy (BDF)
# -----------------------
print("Generating KS data with spin-up (SciPy BDF)...")
u0 = (np.cos(2*np.pi*x/L) + 0.1*np.random.randn(Nx)).astype(np.float64)

sol = solve_ivp(
    ks_rhs_np, (0.0, T_spin + T_train), u0,
    t_eval=t_all, method="BDF",
    rtol=1e-6, atol=1e-8
)
if not sol.success:
    raise RuntimeError(f"SciPy solve_ivp failed: {sol.message}")

U_all = sol.y.T  # (Nt_all, Nx)

spin_idx = int(round(T_spin/dt))
U_true = U_all[spin_idx:, :]       # (~T_train/dt + 1, Nx)
t = t_all[spin_idx:] - T_spin      # reset time to start at 0
u0 = U_true[0].copy()              # on-attractor IC

plt.figure(figsize=(8,3))
plt.contourf(t, x, U_true.T, levels=201, cmap="turbo")
plt.xlabel("t")
plt.ylabel("x")
plt.title("Ground truth KS field u(t,x)")
plt.show()
plt.close()

# -----------------------
# 4) Sparse supervision snapshots
# -----------------------
t_train = np.array([0.0, 2.0, 4.0, 6.0, 10.0, 14.0, 18.0, 20.0], dtype=np.float32)
t_train = np.arange(0.0, T_train + dt, dt).astype(np.float32)

# map times -> indices (robust to float issues using rounding to nearest dt index)
def time_to_index(tt, dt, t0=0.0):
    return int(round((float(tt) - float(t0)) / float(dt)))

train_indices = [time_to_index(tt, dt, t0=0.0) for tt in t_train]
if max(train_indices) >= len(t):
    raise ValueError("t_train contains times outside the generated time grid.")

X_train_np = U_true[train_indices, :]  # (K, Nx)


def build_library_features(u, ux, uxx, uxxxx):
    feats = []
    names = []

    #feats.append(np.ones_like(u)); names.append("1")
    feats += [u, ux, uxx, uxxxx]
    names += ["u", "u_x", "u_xx", "u_xxxx"]

    feats += [u**2, u**3]
    names += ["u^2", "u^3"]

    feats += [ux**2, uxx**2]
    names += ["u_x^2", "u_xx^2"]

    feats += [u*ux, u*uxx, u*uxxxx, ux*uxx]
    names += ["u*u_x", "u*u_xx", "u*u_xxxx", "u_x*u_xx"]

    Theta = np.stack(feats, axis=-1)          # (Nt, Nx, F)
    Theta = Theta.reshape(-1, Theta.shape[-1])# (Nt*Nx, F)
    return Theta, names

# sparse snapshots -> forward diff in time for u_t
X_snap = torch.tensor(X_train_np, dtype=torch.float32, device=device)  # (K, Nx)
t_snap = torch.tensor(t_train,    dtype=torch.float32, device=device)  # (K,)

dt_seg = (t_snap[1:] - t_snap[:-1]).unsqueeze(1)    # (K-1, 1)
U_k    = X_snap[:-1]                                # (K-1, Nx)
Ut_k   = (X_snap[1:] - X_snap[:-1]) / dt_seg        # (K-1, Nx)

U_k_np  = U_k.detach().cpu().numpy()
Ut_k_np = Ut_k.detach().cpu().numpy()

# spatial derivatives at same times
ux_np    = (U_k_np @ Dx_mat.T)
uxx_np   = (U_k_np @ Dxx_mat.T)
uxxxx_np = (U_k_np @ Dxxxx_mat.T)

Theta_np, feat_names = build_library_features(U_k_np, ux_np, uxx_np, uxxxx_np)
y_np = Ut_k_np.reshape(-1, 1)

X = torch.tensor(Theta_np, dtype=torch.float32, device=device)
y = torch.tensor(y_np,     dtype=torch.float32, device=device)

# normalize features
X_mean = X.mean(dim=0, keepdim=True)
X_std  = X.std(dim=0, keepdim=True) + 1e-8
Xn = (X - X_mean) / X_std

# train/test split
N = Xn.shape[0]
perm = torch.randperm(N, device=device)
test_frac = 0.2
N_test = max(1, int(test_frac * N))
test_idx = perm[:N_test]
train_idx = perm[N_test:]


dataset = {
    "train_input": Xn[train_idx],
    "train_label": y[train_idx],
    "test_input":  Xn[test_idx],
    "test_label":  y[test_idx],
}
print("dataset shapes:", {k: tuple(v.shape) for k, v in dataset.items()})
print("Library feature count:", Xn.shape[1])
print("Total samples:", Xn.shape[0])

# ------------------------------------------------------------
# Add this RIGHT AFTER you have U_true and t (after spin-up)
# ------------------------------------------------------------

rollout_horizon = 1
# 1) Torch FD matrices used inside dynamics_fn
Dx_t    = torch.tensor(Dx_mat,    dtype=torch.float32, device=device)
Dxx_t   = torch.tensor(Dxx_mat,   dtype=torch.float32, device=device)
Dxxxx_t = torch.tensor(Dxxxx_mat, dtype=torch.float32, device=device)

# 2) Build rollout trajectories for KAN's rollout loss
U_series = torch.tensor(U_true, dtype=torch.float32, device=device)   # (Nt, Nx)
t_series = torch.tensor(t,      dtype=torch.float32, device=device)   # (Nt,)

H = int(rollout_horizon)  # must match what you pass into fit(...)
Nt = U_series.shape[0]
if Nt < H + 1:
    raise ValueError(f"Need Nt >= rollout_horizon+1, got Nt={Nt}, H={H}")

def sample_windows(U, T, H, n_windows, seed=0):
    # sample random starting indices for windows of length H+1
    g = torch.Generator(device=U.device)
    g.manual_seed(seed)
    max_start = U.shape[0] - (H + 1)
    starts = torch.randint(0, max_start + 1, (n_windows,), generator=g, device=U.device)

    trj_u = torch.stack([U[s:s+H+1] for s in starts], dim=0)  # (n_windows, H+1, Nx)
    trj_t = torch.stack([T[s:s+H+1] for s in starts], dim=0)  # (n_windows, H+1)
    return trj_u, trj_t

n_train_trj = 128
n_test_trj  = 2

train_u_trj, train_t_trj = sample_windows(U_series, t_series, H, n_train_trj, seed=1)
test_u_trj,  test_t_trj  = sample_windows(U_series, t_series, H, n_test_trj,  seed=2)

# Put them into the dataset dict.
dataset["train_traj"] = train_u_trj          # (n_train_trj, H+1, Nx)
#dataset["test_traj"]  = test_u_trj           # (n_test_trj,  H+1, Nx)

# If you want per-trajectory time grids (works with integrate_states):
dataset["train_t"] = train_t_trj             # (n_train_trj, H+1)
#dataset["test_t"]  = test_t_trj              # (n_test_trj,  H+1)

# KAN: (n_features -> 1)
n_features = Xn.shape[1]
rbf = lambda x: torch.exp(-(3*x**2))
rbf = lambda x: torch.exp(-(x**2))
kan_pde = KAN(width=[n_features, 1], grid=10, k=5, base_fun=rbf, seed=0).to(device)

def dynamics_fn(state):
    """
    KS dynamics from learned library model:
    state: torch.Tensor of shape (B, Nx) or (Nx,)
    returns: torch.Tensor of shape (B, Nx) representing u_t
    Requires:
    - Dx_t, Dxx_t, Dxxxx_t: torch tensors (Nx, Nx) on same device/dtype as state
    - X_mean, X_std: saved normalization stats for Theta (shape (1, F) or (F,))
    - kan_pde (or rename to your model): maps normalized Theta -> u_t (per-point)
    - feature order must match training
    """
    # Ensure batched
    if state.dim() == 1:
        state = state.unsqueeze(0)  # (1, Nx)

    u = state  # (B, Nx)

    # Spatial derivatives (periodic FD matrices)
    # (B, Nx) @ (Nx, Nx)^T -> (B, Nx)
    ux    = u @ Dx_t.T
    uxx   = u @ Dxx_t.T
    uxxxx = u @ Dxxxx_t.T

    # Build feature library per gridpoint: (B, Nx, F)
    # Feature order below must match the training library you used.
    feats = [
        #torch.ones_like(u),   # 1
        u,                    # u
        ux,                   # u_x
        uxx,                  # u_xx
        uxxxx,                # u_xxxx
        u**2,                 # u^2
        u**3,                 # u^3
        ux**2,                # u_x^2
        uxx**2,               # u_xx^2
        u*ux,                 # u*u_x
        u*uxx,                # u*u_xx
        u*uxxxx,              # u*u_xxxx
        ux*uxx,               # u_x*u_xx
    ]
    Theta = torch.stack(feats, dim=-1)  # (B, Nx, F)

    # Flatten to per-point samples for the KAN: (B*Nx, F)
    B, Nx, F = Theta.shape
    Theta = Theta.reshape(B * Nx, F)

    # Normalize using saved stats
    mean = X_mean.reshape(1, -1) if X_mean.dim() == 1 else X_mean
    std  = X_std.reshape(1, -1)  if X_std.dim() == 1  else X_std
    Theta_n = (Theta - mean) / std

    # Predict u_t per gridpoint: (B*Nx, 1) or (B*Nx,) depending on your model
    ut = kan_pde(Theta_n)

    # Ensure shape (B*Nx,)
    if ut.dim() == 2 and ut.shape[1] == 1:
        ut = ut[:, 0]

    # Reshape back to field derivative: (B, Nx)
    ut = ut.reshape(B, Nx)
    return ut

# #kan_pde.fit(dataset, opt="LBFGS", steps=100,  lr=1e-3)#, lamb=1e-4)
# fit(kan_pde,
#     dataset,
#     opt="LBFGS",
#     steps=100,
#     lr=1e-3,
#     rollout_weight=1000.5,     # turn on integration loss
#     rollout_horizon=rollout_horizon,    # e.g. short horizon for stability
#     dynamics_fn=dynamics_fn,
#     integrator="rk4",
#     stop_grid_update_step=200
# )
fit(kan_pde,
    dataset,
    opt="LBFGS",
    steps=100,
    lr=1e-3,
    rollout_weight=0.9,     # turn on integration loss
    rollout_horizon=rollout_horizon,    # e.g. short horizon for stability
    dynamics_fn=dynamics_fn,
    integrator="rk4",
    stop_grid_update_step=200
)

with torch.no_grad():
    yhat = kan_pde(Xn).detach().cpu().numpy().reshape(-1)
    ytrue = y.detach().cpu().numpy().reshape(-1)
mse = np.mean((yhat - ytrue)**2)
print("Overall MSE (all samples):", mse)

print("\nFeature library used (order):")
for i, nm in enumerate(feat_names):
    print(f"{i:02d}: {nm}")

model = kan_pde
model.unfix_symbolic_all()
model.save_act = True
_ = model(dataset["train_input"])
model.prune()

model_sym, rep = robust_auto_symbolic(
    model,
    simple_lib=['x','x^2','x^3','0'],   # or pass your SYMBOLIC_LIB dict if your KAN expects dict
    complex_lib=['x','x^2','0'],
    r2_threshold=0.8,
    weight_simple=0.8,
    keep="topk",
    topk_edges=64,
    max_total_complexity=120,           # optional
    verbose=1,
    inplace=True
)

raw = model_sym.symbolic_formula()
print("Symbolic formula extracted.")
exprs_raw, vars_ = model.symbolic_formula()

n_in = int(model.width_in[0])

in_vars_latex_full = [
    #r"$1$",
    r"$u$",
    r"$u_x$",
    r"$u_{xx}$",
    r"$u_{xxxx}$",
    r"$u^2$",
    r"$u^3$",
    r"$u_x^2$",
    r"$u_{xx}^2$",
    r"$u\,u_x$",
    r"$u\,u_{xx}$",
    r"$u\,u_{xxxx}$",
    r"$u_x\,u_{xx}$",
]
out_vars_latex = [r"$u_t$"]

u_sym, u_x_sym, u_xx_sym, u_xxxx_sym = sp.symbols("u u_x u_xx u_xxxx")
feature_syms_full = [
    #sp.Integer(1),
    u_sym,
    u_x_sym,
    u_xx_sym,
    u_xxxx_sym,
    u_sym**2,
    u_sym**3,
    u_x_sym**2,
    u_xx_sym**2,
    u_sym*u_x_sym,
    u_sym*u_xx_sym,
    u_sym*u_xxxx_sym,
    u_x_sym*u_xx_sym,
]

if n_in > len(in_vars_latex_full):
    raise ValueError(
        f"Model expects {n_in} inputs but only {len(in_vars_latex_full)} labels were provided. "
        "Update in_vars_latex_full/feature_syms_full to match your dataset columns."
    )

in_vars_latex = in_vars_latex_full[:n_in]
feature_syms  = feature_syms_full[:n_in]

sub_map = {vars_[i]: feature_syms[i] for i in range(n_in)}

def round_numbers(expr, places=3):
    repl = {}
    for a in expr.atoms(sp.Number):
        try:
            repl[a] = sp.Float(round(float(a), places))
        except Exception:
            pass
    return expr.xreplace(repl)

# Ensure we have a flat list of 3 expressions
# Some KAN versions return nested lists; handle both.
def flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for it in obj:
            out.extend(flatten(it))
        return out
    return [obj]

exprs_list = flatten(exprs_raw)

# Substitute using vars_ (NOT free_symbols inference)
cleaned = []
for expr in exprs_list:
    if not hasattr(expr, "free_symbols"):
        continue
    expr_sub = expr.subs(sub_map)                 # <-- the key fix
    expr_sub = sp.together(sp.expand(expr_sub))
    expr_sub = round_numbers(expr_sub, 3)
    cleaned.append(expr_sub)

# Display
xd, yd, zd = sp.symbols(r"\dot{x} \dot{y} \dot{z}")
if len(cleaned) >= 3:
    display(Math(latex(Eq(xd, cleaned[0]))))
    display(Math(latex(Eq(yd, cleaned[1]))))
    display(Math(latex(Eq(zd, cleaned[2]))))
else:
    lines = []
    for k, ex in enumerate(cleaned):
        lines.append(latex(Eq(sp.Symbol(f"u_t"), ex)))
    display(Math(r"\begin{cases}" + r"\\ ".join(lines) + r"\end{cases}"))

print("\nKuramoto–Sivashinsky equation:")
print("u_t + u u_x + u_xx + u_xxxx = 0")
print("="*60)


# In[5]:


kan_pde.save_act = True
_ = kan_pde(dataset['train_input'])
#kan_pde = kan_pde.prune(node_th=0.185, edge_th=0.185)
#kan_pde = kan_pde.prune(node_th=0.185, edge_th=0.29)
'u u_x u_xx u_xxxx'
kan_pde.plot(in_vars=in_vars_latex, out_vars=[r"$u_t$"])
plt.savefig(
    "figs/ks_model.svg",
    format="svg",
    dpi=300,
    bbox_inches="tight"
)


# In[18]:


# ===========================
# Rollout + plot (robust cell)
# ===========================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

# --- assumes already defined in your notebook/script ---
# kan_pde, dynamics_fn, U_true, t, x, dx, dt, device

os.makedirs("figs", exist_ok=True)

# -----------------------
# 0) Config
# -----------------------
dt_save = float(dt)          # data spacing
n_sub   = 10                 # RK4 substeps per dt_save (increase if still unstable)
h       = dt_save / n_sub
Nt      = len(t)
Nx      = len(x)

use_double = False           # set True to rollout in float64 (often helps KS)
print_every = 50

# -----------------------
# 1) Initial condition
# -----------------------
u0 = torch.tensor(U_true[0], device=device)

if use_double:
    u0 = u0.double()
else:
    u0 = u0.float()

state = u0.unsqueeze(0).clone()   # (1, Nx)

# If switching dtype, you should also switch model + matrices used inside dynamics_fn.
# Easiest: keep use_double=False unless you've already converted kan_pde and Dx_t, etc.

kan_pde.eval()
U_kan = np.zeros_like(U_true, dtype=np.float64)
U_kan[0] = U_true[0]

# -----------------------
# 2) RK4 integrator (with substeps)
# -----------------------
def rk4_step(f, y, h):
    k1 = f(y)
    k2 = f(y + 0.5*h*k1)
    k3 = f(y + 0.5*h*k2)
    k4 = f(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

with torch.no_grad():
    for i in range(1, Nt):
        # integrate from t[i-1] -> t[i] using n_sub substeps
        for _ in range(n_sub):
            state = rk4_step(dynamics_fn, state, h)

        U_kan[i] = state.squeeze(0).detach().cpu().numpy()

        if (i % print_every) == 0:
            # small diagnostics for blow-up
            s = state.squeeze(0)
            print(f"[{i:4d}/{Nt-1}]  "
                  f"u: mean={s.mean().item():+.3e}, std={s.std().item():.3e}, "
                  f"min={s.min().item():+.3e}, max={s.max().item():+.3e}")

# -----------------------
# 3) Plot (matches your premium style)
# -----------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,  'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.direction': 'in',   'ytick.direction': 'in',
    'xtick.major.pad': 4,      'ytick.major.pad': 4,
    'xtick.top': True,         'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

colors_custom = [
    '#0a0e27', '#0d1b4a', '#1b3a6b', '#1f6f8b', '#2d9e8f', '#6ec6a0',
    '#c8e6a0', '#f5e663', '#f5a623', '#e8573a', '#c22e3a', '#7a1a40',
]
cmap_ks = LinearSegmentedColormap.from_list('ks_premium', colors_custom, N=512)

# fair scaling vs truth
vmin, vmax = np.percentile(U_true, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(U_kan.T, cmap=cmap_ks, norm=norm, blend_mode='soft',
               vert_exag=0.08, dx=dx, dy=dt_save, fraction=1.2)

T_grid, X_grid = np.meshgrid(t, x)

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(rgb, extent=[t[0], t[-1], x[0], x[-1]],
          aspect='auto', origin='lower', interpolation='bilinear')
ax.contour(T_grid, X_grid, U_kan.T, levels=np.linspace(vmin, vmax, 18),
           colors='white', linewidths=0.15, alpha=0.25)

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)

ax.set_title(
    r'KAN Model Prediction Rollout (RK4 w/ substeps): $\,u_t=\hat{f}(\Theta(u))$',
    fontsize=12, pad=12, fontweight='medium'
)
ax.text(0.98, 0.96, rf'$L={float(x[-1]+dx):.0f},\;\; N_x={Nx},\;\; n_{{sub}}={n_sub}$',
        transform=ax.transAxes,
        fontsize=8, color='white', ha='right', va='top', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none'))

sm = cm.ScalarMappable(cmap=cmap_ks, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('figs/ks_kan_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/ks_kan_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

print("Done! Saved figs/ks_kan_prediction.png and figs/ks_kan_prediction.pdf")


# In[14]:


n_in = int(model.width_in[0])

in_vars_latex_full = [
    #r"$1$",
    r"$u$",
    r"$u_x$",
    r"$u_{xx}$",
    r"$u_{xxxx}$",
    r"$u^2$",
    r"$u^3$",
    r"$u_x^2$",
    r"$u_{xx}^2$",
    r"$u\,u_x$",
    r"$u\,u_{xx}$",
    r"$u\,u_{xxxx}$",
    r"$u_x\,u_{xx}$",
]
out_vars_latex = [r"$u_t$"]

u_sym, u_x_sym, u_xx_sym, u_xxxx_sym = sp.symbols("u u_x u_xx u_xxxx")
feature_syms_full = [
    #sp.Integer(1),
    u_sym,
    u_x_sym,
    u_xx_sym,
    u_xxxx_sym,
    u_sym**2,
    u_sym**3,
    u_x_sym**2,
    u_xx_sym**2,
    u_sym*u_x_sym,
    u_sym*u_xx_sym,
    u_sym*u_xxxx_sym,
    u_x_sym*u_xx_sym,
]

if n_in > len(in_vars_latex_full):
    raise ValueError(
        f"Model expects {n_in} inputs but only {len(in_vars_latex_full)} labels were provided. "
        "Update in_vars_latex_full/feature_syms_full to match your dataset columns."
    )

in_vars_latex = in_vars_latex_full[:n_in]
feature_syms  = feature_syms_full[:n_in]

# plot (KAN internal)
model.save_act = True
beta_plot = 0.1
_ = model(dataset["train_input"])
model.plot(beta=beta_plot, in_vars=in_vars_latex, out_vars=out_vars_latex)
plt.show()

# symbolic extraction
_ = model(dataset["train_input"])
model.prune()

SYMBOLIC_LIB = {'x': (lambda x: x, lambda x: x, 1, lambda x, y_th: ((), x)),
                 'x^2': (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
                 'x^3': (lambda x: x**3, lambda x: x**3, 3, lambda x, y_th: ((), x**3)),
                 'x^4': (lambda x: x**4, lambda x: x**4, 3, lambda x, y_th: ((), x**4)),
                 'x^5': (lambda x: x**5, lambda x: x**5, 3, lambda x, y_th: ((), x**5)),
                 #'abs': (lambda x: torch.abs(x), lambda x: sympy.Abs(x), 1, lambda x, y_th: ((), torch.abs(x))),
                 #'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x), 2, lambda x, y_th: ((), torch.sin(x))),
                 #'cos': (lambda x: torch.cos(x), lambda x: sympy.cos(x), 2, lambda x, y_th: ((), torch.cos(x))),
                 '0': (lambda x: x*0, lambda x: x*0, 0, lambda x, y_th: ((), x*0)),
}
COMPLEX_SYMBOLIC_LIB = {
    'x': (lambda x: x, lambda x: x, 2, lambda x, y_th: ((), x)),
                 'x^2': (lambda x: x**2, lambda x: x**2, 2, lambda x, y_th: ((), x**2)),
                 #'sin': (lambda x: torch.sin(x), lambda x: sympy.sin(x), 3, lambda x, y_th: ((), torch.sin(x))),
                 #'cos': (lambda x: torch.cos(x), lambda x: sympy.cos(x), 3, lambda x, y_th: ((), torch.cos(x))),
                 '0': (lambda x: x*0, lambda x: x*0, 0, lambda x, y_th: ((), x*0)),
}
def auto_symbolic(self, a_range=(-10, 10), b_range=(-10, 10), lib=None, verbose=1, weight_simple = 0.8, r2_threshold=0.0):
        for l in range(len(self.width_in) - 1):
            for i in range(self.width_in[l]):
                for j in range(self.width_out[l + 1]):
                    if i <= 3:
                        name, fun, r2, c = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=SYMBOLIC_LIB, verbose=False, weight_simple=weight_simple)
                    else:
                        #print(f"wrong code path {l, j, i}")
                        name, fun, r2, c = self.suggest_symbolic(l, i, j, a_range=a_range, b_range=b_range, lib=COMPLEX_SYMBOLIC_LIB, verbose=False, weight_simple=weight_simple)
                        print(f"Node: {l, j, i} {name}, r2={r2}, c={c}")
                    if r2 >= r2_threshold:
                        self.fix_symbolic(l, i, j, name, verbose=verbose > 1, log_history=False)
                        if verbose >= 1:
                            print(f'fixing ({l},{i},{j}) with {name}, r2={r2}, c={c}')
                    else:
                        self.fix_symbolic(l, i, j, '0', verbose=verbose > 1, log_history=False)
                        print(f'For ({l},{i},{j}) the best fit was {name}, but r^2 = {r2} and this is lower than {r2_threshold}. This edge was omitted, keep training or try a different threshold. setting to zero.')

        self.log_history('auto_symbolic')

auto_symbolic(
    model,
    lib=['x','x^2'],
    r2_threshold=0.9,
    weight_simple=0.9
    )
raw = model.symbolic_formula()

def flatten_exprs(obj):
    out = []
    if isinstance(obj, (list, tuple)):
        for it in obj:
            out.extend(flatten_exprs(it))
    else:
        out.append(obj)
    return out

exprs = flatten_exprs(raw)
exprs = [e for e in exprs if hasattr(e, "free_symbols")]

def prune_small_terms(obj, tol=1e-3):
    # handle lists/tuples of expressions
    if isinstance(obj, (list, tuple)):
        return [prune_small_terms(e, tol=tol) for e in obj]

    expr = sp.expand(obj)

    kept = []
    for term in expr.as_ordered_terms():
        coeff, rest = term.as_coeff_Mul()
        try:
            if abs(float(coeff)) >= tol:
                kept.append(term)
        except Exception:
            # non-numeric coefficient; keep the term
            kept.append(term)

    return sp.Add(*kept)


exprs = prune_small_terms(exprs, tol=1e-2)

if len(exprs) == 0:
    raise ValueError(
        "symbolic_formula() did not return SymPy expressions. "
        f"Got type={type(raw)} with content={raw}"
    )

def infer_model_input_symbols(expr, n_inputs):
    syms = list(expr.free_symbols)

    def sort_key(s):
        name = str(s)
        num = ""
        for ch in reversed(name):
            if ch.isdigit():
                num = ch + num
            else:
                break
        base = name.rstrip("0123456789")
        return (base, int(num) if num else 10**9, name)

    syms_sorted = sorted(syms, key=sort_key)
    return syms_sorted[:n_inputs]

def round_numbers(expr, places=1):
    repl = {}
    for a in expr.atoms(sp.Number):
        try:
            repl[a] = sp.Float(round(float(a), places))
        except Exception:
            pass
    return expr.xreplace(repl)


cleaned = []
for expr in exprs:
    model_in_syms = infer_model_input_symbols(expr, n_in)
    sub_map = {s: v for s, v in zip(model_in_syms, feature_syms)}

    expr_pde = expr.xreplace(sub_map)
    expr_pde = sp.together(sp.expand(expr_pde))
    expr_pde = round_numbers(expr_pde, 3)
    cleaned.append(expr_pde)

u_t_sym = sp.Symbol("u_t")

if len(cleaned) == 1:
    display(Math(latex(Eq(u_t_sym, cleaned[0]))))
else:
    lines = []
    for k, ex in enumerate(cleaned):
        lines.append(latex(Eq(sp.Symbol(f"y_{k}"), ex)))
    display(Math(r"\begin{cases}" + r"\\ ".join(lines) + r"\end{cases}"))

print("\nDone.")


# In[4]:


model = kan_pde
model.unfix_symbolic_all()
model.save_act = True
_ = model(dataset["train_input"])
model.prune()

model_sym, rep = robust_auto_symbolic(
    model,
    simple_lib=['x','x^2','x^3','0'],   # or pass your SYMBOLIC_LIB dict if your KAN expects dict
    complex_lib=['x','x^2','0'],
    r2_threshold=0.8,
    weight_simple=0.8,
    keep="topk",
    topk_edges=64,
    max_total_complexity=120,           # optional
    verbose=1,
    inplace=True
)

raw = model_sym.symbolic_formula()
print("Symbolic formula extracted.")
exprs_raw, vars_ = model.symbolic_formula()

# vars_ might be a list like [x_1, x_2, ..., x_n]; its order matches model inputs.
# Build a deterministic substitution map from those vars_ to your semantic feature symbols.
if len(vars_) < n_in:
    raise ValueError(f"symbolic_formula returned only {len(vars_)} vars, but model has n_in={n_in}")

sub_map = {vars_[i]: feature_syms[i] for i in range(n_in)}

def round_numbers(expr, places=3):
    repl = {}
    for a in expr.atoms(sp.Number):
        try:
            repl[a] = sp.Float(round(float(a), places))
        except Exception:
            pass
    return expr.xreplace(repl)

# Ensure we have a flat list of 3 expressions
# Some KAN versions return nested lists; handle both.
def flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for it in obj:
            out.extend(flatten(it))
        return out
    return [obj]

exprs_list = flatten(exprs_raw)

# Substitute using vars_ (NOT free_symbols inference)
cleaned = []
for expr in exprs_list:
    if not hasattr(expr, "free_symbols"):
        continue
    expr_sub = expr.subs(sub_map)                 # <-- the key fix
    expr_sub = sp.together(sp.expand(expr_sub))
    expr_sub = round_numbers(expr_sub, 3)
    cleaned.append(expr_sub)

# Display
xd, yd, zd = sp.symbols(r"\dot{x} \dot{y} \dot{z}")
if len(cleaned) >= 3:
    display(Math(latex(Eq(xd, cleaned[0]))))
    display(Math(latex(Eq(yd, cleaned[1]))))
    display(Math(latex(Eq(zd, cleaned[2]))))
else:
    lines = []
    for k, ex in enumerate(cleaned):
        lines.append(latex(Eq(sp.Symbol(f"u_t"), ex)))
    display(Math(r"\begin{cases}" + r"\\ ".join(lines) + r"\end{cases}"))

print("\nKuramoto–Sivashinsky equation:")
print("u_t + u u_x + u_xx + u_xxxx = 0")
print("="*60)


# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource
from scipy.integrate import solve_ivp

np.random.seed(0)

# --- KS PDE setup (periodic) ---
L, Nx = 64, 128
dx = L / Nx
x = np.arange(0.0, L, dx)

T_spin, T_train, dt = 100.0, 20.0, 0.1
t_all = np.linspace(0.0, T_spin + T_train, int(round((T_spin + T_train) / dt)) + 1)

def periodic_shift_mat(N, k):
    M = np.zeros((N, N))
    for i in range(N):
        M[i, (i + k) % N] = 1.0
    return M

I   = np.eye(Nx)
Sp, Sm   = periodic_shift_mat(Nx, +1), periodic_shift_mat(Nx, -1)
Spp, Smm = periodic_shift_mat(Nx, +2), periodic_shift_mat(Nx, -2)

Dx    = (Sp - Sm) / (2 * dx)
Dxx   = (Sp - 2 * I + Sm) / dx**2
Dxxxx = (Smm - 4*Sm + 6*I - 4*Sp + Spp) / dx**4

def ks_rhs(_t, u):
    return -(u * (Dx @ u)) - Dxx @ u - Dxxxx @ u

# --- Solve ---
print("Generating KS data...")
u0 = np.cos(2 * np.pi * x / L) + 0.1 * np.random.randn(Nx)
sol = solve_ivp(ks_rhs, (0, T_spin + T_train), u0, t_eval=t_all, method="BDF",
                rtol=1e-6, atol=1e-8)
if not sol.success:
    raise RuntimeError(sol.message)

spin_idx = int(round(T_spin / dt))
U_true = sol.y.T[spin_idx:]
t = t_all[spin_idx:] - T_spin

# --- Plot ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,  'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.direction': 'in',   'ytick.direction': 'in',
    'xtick.major.pad': 4,      'ytick.major.pad': 4,
    'xtick.top': True,         'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

colors_custom = [
    '#0a0e27', '#0d1b4a', '#1b3a6b', '#1f6f8b', '#2d9e8f', '#6ec6a0',
    '#c8e6a0', '#f5e663', '#f5a623', '#e8573a', '#c22e3a', '#7a1a40',
]
cmap_ks = LinearSegmentedColormap.from_list('ks_premium', colors_custom, N=512)

vmin, vmax = np.percentile(U_true, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(U_true.T, cmap=cmap_ks, norm=norm, blend_mode='soft',
               vert_exag=0.08, dx=dx, dy=dt, fraction=1.2)

T_grid, X_grid = np.meshgrid(t, x)

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(rgb, extent=[t[0], t[-1], x[0], x[-1]],
          aspect='auto', origin='lower', interpolation='bilinear')
ax.contour(T_grid, X_grid, U_true.T, levels=np.linspace(vmin, vmax, 18),
           colors='white', linewidths=0.15, alpha=0.25)

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)
ax.set_title(r'Kuramoto–Sivashinsky Equation: $\,u_t = -u\,u_x - u_{xx} - u_{xxxx}$',
             fontsize=12, pad=12, fontweight='medium')
ax.text(0.98, 0.96, r'$L=64,\;\; N_x=128$', transform=ax.transAxes,
        fontsize=8, color='white', ha='right', va='top', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none'))

sm = cm.ScalarMappable(cmap=cmap_ks, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('ks_contour.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('ks_contour.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()
print("Done!")


# In[20]:


# ===========================
# Rollout + plot (robust cell)
# ===========================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

# --- assumes already defined in your notebook/script ---
# kan_pde, dynamics_fn, U_true, t, x, dx, dt, device

os.makedirs("figs", exist_ok=True)

# -----------------------
# 0) Config
# -----------------------
dt_save = float(dt)          # data spacing
n_sub   = 10                 # RK4 substeps per dt_save (increase if still unstable)
h       = dt_save / n_sub
Nt      = len(t)
Nx      = len(x)

use_double = False           # set True to rollout in float64 (often helps KS)
print_every = 50

# -----------------------
# 1) Initial condition
# -----------------------
u0 = torch.tensor(U_true[0], device=device)

if use_double:
    u0 = u0.double()
else:
    u0 = u0.float()

state = u0.unsqueeze(0).clone()   # (1, Nx)

# If switching dtype, you should also switch model + matrices used inside dynamics_fn.
# Easiest: keep use_double=False unless you've already converted kan_pde and Dx_t, etc.

kan_pde.eval()
U_kan = np.zeros_like(U_true, dtype=np.float64)
U_kan[0] = U_true[0]

# -----------------------
# 2) RK4 integrator (with substeps)
# -----------------------
def rk4_step(f, y, h):
    k1 = f(y)
    k2 = f(y + 0.5*h*k1)
    k3 = f(y + 0.5*h*k2)
    k4 = f(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

with torch.no_grad():
    for i in range(1, Nt):
        # integrate from t[i-1] -> t[i] using n_sub substeps
        for _ in range(n_sub):
            state = rk4_step(dynamics_fn, state, h)

        U_kan[i] = state.squeeze(0).detach().cpu().numpy()

        if (i % print_every) == 0:
            # small diagnostics for blow-up
            s = state.squeeze(0)
            print(f"[{i:4d}/{Nt-1}]  "
                  f"u: mean={s.mean().item():+.3e}, std={s.std().item():.3e}, "
                  f"min={s.min().item():+.3e}, max={s.max().item():+.3e}")

# -----------------------
# 3) Plot (matches your premium style)
# -----------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,  'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.direction': 'in',   'ytick.direction': 'in',
    'xtick.major.pad': 4,      'ytick.major.pad': 4,
    'xtick.top': True,         'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

colors_custom = [
    '#0a0e27', '#0d1b4a', '#1b3a6b', '#1f6f8b', '#2d9e8f', '#6ec6a0',
    '#c8e6a0', '#f5e663', '#f5a623', '#e8573a', '#c22e3a', '#7a1a40',
]
cmap_ks = LinearSegmentedColormap.from_list('ks_premium', colors_custom, N=512)

# fair scaling vs truth
vmin, vmax = np.percentile(U_true, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(U_kan.T, cmap=cmap_ks, norm=norm, blend_mode='soft',
               vert_exag=0.08, dx=dx, dy=dt_save, fraction=1.2)

T_grid, X_grid = np.meshgrid(t, x)

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(rgb, extent=[t[0], t[-1], x[0], x[-1]],
          aspect='auto', origin='lower', interpolation='bilinear')
ax.contour(T_grid, X_grid, U_kan.T, levels=np.linspace(vmin, vmax, 18),
           colors='white', linewidths=0.15, alpha=0.25)

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)

ax.set_title(
    r'KAN Model Prediction Rollout (RK4 w/ substeps): $\,u_t=\hat{f}(\Theta(u))$',
    fontsize=12, pad=12, fontweight='medium'
)
ax.text(0.98, 0.96, rf'$L={float(x[-1]+dx):.0f},\;\; N_x={Nx},\;\; n_{{sub}}={n_sub}$',
        transform=ax.transAxes,
        fontsize=8, color='white', ha='right', va='top', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none'))

sm = cm.ScalarMappable(cmap=cmap_ks, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('figs/ks_kan_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/ks_kan_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

print("Done! Saved figs/ks_kan_prediction.png and figs/ks_kan_prediction.pdf")


# In[22]:


# ============================================================
# Rollout + premium plot for DISCOVERED KS equation
#   u_t = -1.274*u*u_x - 1.052*u_xx - 1.92*u_xxxx
# ============================================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

os.makedirs("figs", exist_ok=True)

# -----------------------------
# 1) Discovered PDE RHS
# -----------------------------
a = -1.274
b = -1.052
c = -1.92
d = 0.0

@torch.no_grad()
def ks_discovered_rhs(u):
    """
    u: (B, Nx) torch tensor
    returns: (B, Nx)
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)

    ux    = u @ Dx_t.T
    uxx   = u @ Dxx_t.T
    uxxxx = u @ Dxxxx_t.T

    return a * (u * ux) + b * uxx + c * uxxxx + d

# -----------------------------
# 2) RK4 rollout (with substeps)
# -----------------------------
@torch.no_grad()
def rollout_rk4_substeps(u0, t_grid, rhs_fn, n_sub=60, stop_on_nan=True):
    """
    u0: (Nx,) or (1,Nx)
    t_grid: (Nt,) torch tensor
    rhs_fn: function(u)->u_t
    """
    if u0.dim() == 1:
        u = u0[None, :].clone()
    else:
        u = u0.clone()

    Nt = t_grid.numel()
    Nx = u.shape[-1]
    out = torch.zeros((Nt, Nx), device=u.device, dtype=u.dtype)
    out[0] = u[0]

    for n in range(Nt - 1):
        dt_out = (t_grid[n+1] - t_grid[n]).item()
        h = dt_out / n_sub

        for _ in range(n_sub):
            k1 = rhs_fn(u)
            k2 = rhs_fn(u + 0.5 * h * k1)
            k3 = rhs_fn(u + 0.5 * h * k2)
            k4 = rhs_fn(u + h * k3)
            u = u + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            if stop_on_nan and (torch.isnan(u).any() or torch.isinf(u).any()):
                print(f"NaN/Inf encountered at outer step n={n}, substep h={h:.3e}, t={t_grid[n].item():.3f}")
                return out[:n+1]  # return partial

        out[n+1] = u[0]

        if (n+1) % 50 == 0:
            s = u[0]
            print(f"[{n+1:4d}/{Nt-1}] u: mean={s.mean().item():+.3e}, std={s.std().item():.3e}, "
                  f"min={s.min().item():+.3e}, max={s.max().item():+.3e}")

    return out

# -----------------------------
# 3) Run simulation on the same grid/time as your data
# -----------------------------
t_grid = torch.tensor(t, dtype=torch.float32, device=device)      # (Nt,)
u0_sim = torch.tensor(U_true[0], dtype=torch.float32, device=device)  # (Nx,)

# Stability: with dx=0.5 and |c|=1.92, n_sub ~ 40+ is typically needed for dt_out=0.1
n_sub = 60
U_sim = rollout_rk4_substeps(u0_sim, t_grid, ks_discovered_rhs, n_sub=n_sub)

U_sim_np  = U_sim.detach().cpu().numpy()
U_true_np = np.asarray(U_true)

# If rollout stopped early due to NaNs, adjust t used for plotting
t_plot = t[:U_sim_np.shape[0]]

# -----------------------------
# 4) Premium plot (same style)
# -----------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,  'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.direction': 'in',   'ytick.direction': 'in',
    'xtick.major.pad': 4,      'ytick.major.pad': 4,
    'xtick.top': True,         'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

colors_custom = [
    '#0a0e27', '#0d1b4a', '#1b3a6b', '#1f6f8b', '#2d9e8f', '#6ec6a0',
    '#c8e6a0', '#f5e663', '#f5a623', '#e8573a', '#c22e3a', '#7a1a40',
]
cmap_ks = LinearSegmentedColormap.from_list('ks_premium', colors_custom, N=512)

# Fair scaling: same as truth
vmin, vmax = np.percentile(U_true_np, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(U_sim_np.T, cmap=cmap_ks, norm=norm, blend_mode='soft',
               vert_exag=0.08, dx=dx, dy=float(dt), fraction=1.2)

T_grid, X_grid = np.meshgrid(t_plot, x)

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(rgb, extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
          aspect='auto', origin='lower', interpolation='bilinear')
ax.contour(T_grid, X_grid, U_sim_np.T, levels=np.linspace(vmin, vmax, 18),
           colors='white', linewidths=0.15, alpha=0.25)

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)

ax.set_title(
    r'Discovered Model Rollout: $u_t=-1.274\,u\,u_x-1.052\,u_{xx}-1.92\,u_{xxxx}$',
    fontsize=12, pad=12, fontweight='medium'
)

ax.text(0.98, 0.96, rf'$L=64,\;\; N_x=128,\;\; n_{{sub}}={n_sub}$',
        transform=ax.transAxes,
        fontsize=8, color='white', ha='right', va='top', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none'))

sm = cm.ScalarMappable(cmap=cmap_ks, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('figs/ks_discovered_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/ks_discovered_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

print("Done! Saved figs/ks_discovered_prediction.png/pdf")


# In[23]:


# ===========================
# Premium error-field plots
#   (red-white-blue, white=0)
# ===========================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

os.makedirs("figs", exist_ok=True)

# ---------- style (same as your plots) ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,  'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.direction': 'in',   'ytick.direction': 'in',
    'xtick.major.pad': 4,      'ytick.major.pad': 4,
    'xtick.top': True,         'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

# Diverging colormap: blue -> white -> red (white=0)
cmap_err = LinearSegmentedColormap.from_list(
    "rbw",
    ["#223b8f", "#f7f7f7", "#b2182b"],  # deep blue, near-white, deep red
    N=512
)

def plot_error_field(U_pred, U_true, t, x, dx, dt,
                     title, out_png, out_pdf=None,
                     clip_pct=99.0, n_contours=18):
    """
    U_pred, U_true: (Nt, Nx)
    """
    U_pred = np.asarray(U_pred)
    U_true = np.asarray(U_true)

    Nt = min(U_pred.shape[0], U_true.shape[0], len(t))
    U_pred = U_pred[:Nt]
    U_true = U_true[:Nt]
    t_plot = t[:Nt]

    E = U_pred - U_true  # (Nt, Nx)

    # robust symmetric limits around 0
    vmax = np.nanpercentile(np.abs(E), clip_pct) + 1e-12
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # premium shaded look
    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(E.T, cmap=cmap_err, norm=norm, blend_mode='soft',
                   vert_exag=0.08, dx=dx, dy=float(dt), fraction=1.2)

    T_grid, X_grid = np.meshgrid(t_plot, x)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.imshow(rgb, extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
              aspect='auto', origin='lower', interpolation='bilinear')

    # subtle contours (symmetric about 0)
    levels = np.linspace(-vmax, vmax, n_contours)
    ax.contour(T_grid, X_grid, E.T, levels=levels,
               colors='white', linewidths=0.12, alpha=0.20)

    ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
    ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
    ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=12, pad=12, fontweight='medium')

    ax.text(0.98, 0.96, rf'clip={clip_pct:.1f}\%,\;\; |e|_{{max}}\approx {vmax:.3g}',
            transform=ax.transAxes,
            fontsize=8, color='black', ha='right', va='top', alpha=0.75,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.6, edgecolor='none'))

    sm = cm.ScalarMappable(cmap=cmap_err, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
    cbar.ax.tick_params(labelsize=9, width=0.5)
    cbar.set_label(r'error $e(x,t)=u_{\mathrm{pred}}-u_{\mathrm{true}}$', fontsize=12, labelpad=8)
    cbar.outline.set_linewidth(0.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    if out_pdf is not None:
        plt.savefig(out_pdf, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

    return E, vmax

# ------------------------------
# Example 1: KAN error field
# ------------------------------
# Requires: U_kan already computed by your KAN rollout cell
E_kan, vmax_kan = plot_error_field(
    U_pred=U_kan,
    U_true=U_true,
    t=t,
    x=x,
    dx=dx,
    dt=dt,
    title=r'KAN Rollout Error Field: $e(x,t)=u_{\mathrm{KAN}}-u_{\mathrm{true}}$',
    out_png='figs/ks_kan_error.png',
    out_pdf='figs/ks_kan_error.pdf',
    clip_pct=99.0
)

# ------------------------------
# Example 2: Discovered-PDE error field
# ------------------------------
# Requires: U_sim_np (or whatever you named it) from your discovered rollout
E_disc, vmax_disc = plot_error_field(
    U_pred=U_sim_np,
    U_true=U_true,
    t=t,
    x=x,
    dx=dx,
    dt=dt,
    title=r'Discovered PDE Error Field: $e(x,t)=u_{\mathrm{disc}}-u_{\mathrm{true}}$',
    out_png='figs/ks_discovered_error.png',
    out_pdf='figs/ks_discovered_error.pdf',
    clip_pct=99.0
)

print("Saved:")
print("  figs/ks_kan_error.png/.pdf")
print("  figs/ks_discovered_error.png/.pdf")


# In[5]:


import numpy as np
import torch
from kan import KAN

# ---------------------------
# 0) Small utilities
# ---------------------------
def set_all_seeds(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def time_to_index(tt, dt, t0=0.0):
    return int(round((float(tt) - float(t0)) / float(dt)))

def split_train_test(Xn, y, test_frac=0.2, seed=0):
    g = torch.Generator(device=Xn.device)
    g.manual_seed(seed)
    N = Xn.shape[0]
    perm = torch.randperm(N, generator=g, device=Xn.device)
    N_test = max(1, int(test_frac * N))
    test_idx = perm[:N_test]
    train_idx = perm[N_test:]
    return {
        "train_input": Xn[train_idx],
        "train_label": y[train_idx],
        "test_input":  Xn[test_idx],
        "test_label":  y[test_idx],
    }

def sample_windows(U, T, H, n_windows, seed=0):
    # U: (Nt, Nx), T: (Nt,)
    g = torch.Generator(device=U.device)
    g.manual_seed(seed)
    max_start = U.shape[0] - (H + 1)
    starts = torch.randint(0, max_start + 1, (n_windows,), generator=g, device=U.device)
    trj_u = torch.stack([U[s:s+H+1] for s in starts], dim=0)  # (n_windows, H+1, Nx)
    trj_t = torch.stack([T[s:s+H+1] for s in starts], dim=0)  # (n_windows, H+1)
    return trj_u, trj_t

# ---------------------------
# 1) Feature libraries
# ---------------------------
def build_library_features(u, ux, uxx, uxxxx):
    """
    Matches your current KS feature library. Returns:
      Theta_flat: (Nt*Nx, F)
      names: list[str] length F
    """
    feats = []
    names = []

    feats += [u, ux, uxx, uxxxx]
    names += ["u", "u_x", "u_xx", "u_xxxx"]

    feats += [u**2, u**3]
    names += ["u^2", "u^3"]

    feats += [ux**2, uxx**2]
    names += ["u_x^2", "u_xx^2"]

    feats += [u*ux, u*uxx, u*uxxxx, ux*uxx]
    names += ["u*u_x", "u*u_xx", "u*u_xxxx", "u_x*u_xx"]

    Theta = np.stack(feats, axis=-1)           # (Nt, Nx, F)
    Theta = Theta.reshape(-1, Theta.shape[-1]) # (Nt*Nx, F)
    return Theta, names

# ---------------------------
# 2) Build dataset for "features" and "no-features"
# ---------------------------
def make_ks_dataset(
    U_true_np: np.ndarray,   # (Nt, Nx)
    t_np: np.ndarray,        # (Nt,)
    dt: float,
    Dx_mat: np.ndarray,
    Dxx_mat: np.ndarray,
    Dxxxx_mat: np.ndarray,
    *,
    use_features: bool,
    test_frac: float = 0.2,
    seed: int = 0,
):
    """
    Builds per-gridpoint regression:
      X -> u_t at each spatial point (flattened).

    If use_features=False: X = [u] only (1D local model).
    If use_features=True:  X = Theta(u, ux, uxx, uxxxx, ...) (your full library).
    """
    # snapshots for finite-diff in time
    t_train = np.arange(0.0, float(t_np[-1]) + dt, dt).astype(np.float32)
    idx = [time_to_index(tt, dt, t0=0.0) for tt in t_train]
    idx = [i for i in idx if i < len(t_np)]
    X_train_np = U_true_np[idx, :]  # (K, Nx)

    # forward diff in time for u_t
    U_k = X_train_np[:-1, :]             # (K-1, Nx)
    Ut  = (X_train_np[1:, :] - X_train_np[:-1, :]) / dt  # (K-1, Nx)

    # flatten target to (samples, 1)
    y_np = Ut.reshape(-1, 1).astype(np.float32)

    if not use_features:
        # X = u only (flatten)
        X_np = U_k.reshape(-1, 1).astype(np.float32)  # ( (K-1)*Nx, 1 )
        feat_names = ["u"]
    else:
        # compute spatial derivatives at same times
        ux    = (U_k @ Dx_mat.T)
        uxx   = (U_k @ Dxx_mat.T)
        uxxxx = (U_k @ Dxxxx_mat.T)
        Theta_np, feat_names = build_library_features(U_k, ux, uxx, uxxxx)
        X_np = Theta_np.astype(np.float32)

    # torch tensors
    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    # normalize features
    X_mean = X.mean(dim=0, keepdim=True)
    X_std  = X.std(dim=0, keepdim=True) + 1e-8
    Xn = (X - X_mean) / X_std

    dataset = split_train_test(Xn, y, test_frac=test_frac, seed=seed)
    dataset["X_mean"] = X_mean
    dataset["X_std"]  = X_std
    dataset["feat_names"] = feat_names
    dataset["use_features"] = use_features
    dataset["dt"] = dt
    return dataset

# ---------------------------
# 3) Rollout dynamics_fn factories
# ---------------------------
def make_dynamics_fn_no_features(model, X_mean, X_std):
    """
    Pointwise local dynamics: u_t(x) = model( normalize([u(x)]) )
    NOTE: this ignores spatial coupling, intentionally "no features".
    """
    def f(state):
        if state.dim() == 1:
            state_b = state.unsqueeze(0)  # (1, Nx)
        else:
            state_b = state               # (B, Nx)

        B, Nx = state_b.shape
        u_flat = state_b.reshape(B * Nx, 1)  # (B*Nx, 1)
        u_n = (u_flat - X_mean) / X_std
        ut_flat = model(u_n)
        if ut_flat.dim() == 2 and ut_flat.shape[1] == 1:
            ut_flat = ut_flat[:, 0]
        ut = ut_flat.reshape(B, Nx)
        return ut
    return f

def make_dynamics_fn_features(model, X_mean, X_std, Dx_t, Dxx_t, Dxxxx_t):
    """
    Your physics-library dynamics:
      build Theta(u, ux, uxx, uxxxx, ...) per gridpoint -> model -> u_t
    """
    def f(state):
        if state.dim() == 1:
            state_b = state.unsqueeze(0)  # (1, Nx)
        else:
            state_b = state               # (B, Nx)

        u = state_b
        ux    = u @ Dx_t.T
        uxx   = u @ Dxx_t.T
        uxxxx = u @ Dxxxx_t.T

        feats = [
            u, ux, uxx, uxxxx,
            u**2, u**3,
            ux**2, uxx**2,
            u*ux, u*uxx, u*uxxxx, ux*uxx
        ]
        Theta = torch.stack(feats, dim=-1)  # (B, Nx, F)
        B, Nx, F = Theta.shape
        Theta = Theta.reshape(B * Nx, F)

        Theta_n = (Theta - X_mean) / X_std
        ut_flat = model(Theta_n)
        if ut_flat.dim() == 2 and ut_flat.shape[1] == 1:
            ut_flat = ut_flat[:, 0]
        ut = ut_flat.reshape(B, Nx)
        return ut
    return f

# ---------------------------
# 4) Model builders (deep vs zero-depth)
# ---------------------------
def make_kan_model(n_in: int, *, depth: str, seed: int, grid=10, k=5):
    rbf = lambda x: torch.exp(-(x**2))
    if depth == "zero":
        width = [n_in, 1]
    elif depth == "deep":
        # reasonable default; adjust widths if you want param-budget matching
        width = [n_in, 16, 16, 1]
    else:
        raise ValueError("depth must be 'zero' or 'deep'")
    return KAN(width=width, grid=grid, k=k, base_fun=rbf, seed=seed).to(device)

# ---------------------------
# 5) One runner for a single condition
# ---------------------------
def train_one_condition(
    *,
    condition_name: str,
    dataset,
    U_series, t_series,
    rollout_horizon: int,
    n_train_trj: int,
    seed: int,
    depth: str,
    opt="LBFGS",
    steps=100,
    lr=1e-3,
    rollout_weight=0.9,
    integrator="rk4",
    stop_grid_update_step=200,
):
    set_all_seeds(seed)

    use_features = dataset["use_features"]
    X_mean = dataset["X_mean"]
    X_std  = dataset["X_std"]

    # Trajectory windows for rollout loss
    H = int(rollout_horizon)
    train_u_trj, train_t_trj = sample_windows(U_series, t_series, H, n_train_trj, seed=seed+10)

    dataset_local = dict(dataset)  # shallow copy OK
    dataset_local["train_traj"] = train_u_trj
    dataset_local["train_t"]    = train_t_trj

    # Derivative matrices in torch (needed only for features dynamics)
    Dx_t    = torch.tensor(Dx_mat,    dtype=torch.float32, device=device)
    Dxx_t   = torch.tensor(Dxx_mat,   dtype=torch.float32, device=device)
    Dxxxx_t = torch.tensor(Dxxxx_mat, dtype=torch.float32, device=device)

    # Model
    n_in = dataset_local["train_input"].shape[1]
    model = make_kan_model(n_in, depth=depth, seed=seed, grid=10, k=5)

    # dynamics_fn for rollout term
    if use_features:
        dynamics_fn = make_dynamics_fn_features(model, X_mean, X_std, Dx_t, Dxx_t, Dxxxx_t)
    else:
        dynamics_fn = make_dynamics_fn_no_features(model, X_mean, X_std)

    # Train
    hist = fit(
        model,
        dataset_local,
        opt=opt,
        steps=steps,
        lr=lr,
        rollout_weight=rollout_weight,
        rollout_horizon=rollout_horizon,
        dynamics_fn=dynamics_fn,
        integrator=integrator,
        stop_grid_update_step=stop_grid_update_step,
    )

    # Evaluate derivative MSE on *heldout* (test_input/test_label)
    with torch.no_grad():
        yhat = model(dataset_local["test_input"])
        ytrue = dataset_local["test_label"]
        mse_test = torch.mean((yhat - ytrue) ** 2).item()

    out = {
        "condition": condition_name,
        "depth": depth,
        "use_features": use_features,
        "n_in": n_in,
        "test_mse": mse_test,
        "history": hist,
        "model": model,
        "dynamics_fn": dynamics_fn,
    }
    return out

# ---------------------------
# 6) Run the full 2x2 ablation
# ---------------------------
def run_full_ablation(
    U_true_np, t_np, dt,
    Dx_mat, Dxx_mat, Dxxxx_mat,
    *,
    rollout_horizon=1,
    n_train_trj=128,
    seed=0,
    steps=100,
    lr=1e-3,
    rollout_weight=0.9,
):
    # series tensors for rollout sampling
    U_series = torch.tensor(U_true_np, dtype=torch.float32, device=device)  # (Nt, Nx)
    t_series = torch.tensor(t_np,      dtype=torch.float32, device=device)  # (Nt,)

    # build datasets
    ds_feat = make_ks_dataset(
        U_true_np, t_np, dt, Dx_mat, Dxx_mat, Dxxxx_mat,
        use_features=True, test_frac=0.2, seed=seed
    )
    ds_nof  = make_ks_dataset(
        U_true_np, t_np, dt, Dx_mat, Dxx_mat, Dxxxx_mat,
        use_features=False, test_frac=0.2, seed=seed
    )

    # conditions in requested order
    conditions = [
        ("Deep KAN (no features)",     "deep", ds_nof),
        ("Zero-depth KAN (no features)","zero", ds_nof),
        ("Deep KAN + features",        "deep", ds_feat),
        ("Zero-depth KAN + features",  "zero", ds_feat),
    ]

    results = {}
    for name, depth, ds in conditions:
        print("\n" + "="*80)
        print(f"RUN: {name}")
        print(f"  depth={depth} | use_features={ds['use_features']} | n_in={ds['train_input'].shape[1]}")
        print("="*80)

        res = train_one_condition(
            condition_name=name,
            dataset=ds,
            U_series=U_series,
            t_series=t_series,
            rollout_horizon=rollout_horizon,
            n_train_trj=n_train_trj,
            seed=seed,
            depth=depth,
            steps=steps,
            lr=lr,
            rollout_weight=rollout_weight,
        )
        results[name] = res
        print(f"[DONE] {name} | test_mse={res['test_mse']:.6e}")

    # quick summary
    print("\n" + "#"*80)
    print("ABLATION SUMMARY (lower is better)")
    for k, v in results.items():
        print(f"{k:30s} | test_mse = {v['test_mse']:.6e} | n_in={v['n_in']}")
    print("#"*80)

    return results

# ---------------------------
# 7) Call it (uses your existing U_true, t, dt, Dx_mat, Dxx_mat, Dxxxx_mat)
# ---------------------------
# NOTE: U_true should be your post-spin-up array (Nt, Nx)
#       t should be your post-spin-up time vector length Nt
ablation_results = run_full_ablation(
    U_true_np=U_true,
    t_np=t,
    dt=dt,
    Dx_mat=Dx_mat,
    Dxx_mat=Dxx_mat,
    Dxxxx_mat=Dxxxx_mat,
    rollout_horizon=1,
    n_train_trj=10,
    seed=0,
    steps=100,
    lr=1e-3,
    rollout_weight=0.9,
)


# In[6]:


import numpy as np
import torch

# ============================================================
# 1) Generic rollout integrators (torch)
# ============================================================

@torch.no_grad()
def rollout_torch(dynamics_fn, u0, dt, n_steps, method="rk4"):
    """
    dynamics_fn: f(u) -> u_t, supports u shape (B,Nx) or (Nx,)
    u0: torch (Nx,) or (B,Nx)
    returns: traj (n_steps+1, Nx) if u0 is (Nx,) else (n_steps+1,B,Nx)
    """
    if u0.dim() == 1:
        u = u0.unsqueeze(0)  # (1,Nx)
        squeeze = True
    else:
        u = u0
        squeeze = False

    out = [u.clone()]

    def euler_step(u):
        return u + dt * dynamics_fn(u)

    def rk4_step(u):
        k1 = dynamics_fn(u)
        k2 = dynamics_fn(u + 0.5 * dt * k1)
        k3 = dynamics_fn(u + 0.5 * dt * k2)
        k4 = dynamics_fn(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    stepper = rk4_step if method.lower() == "rk4" else euler_step

    for _ in range(n_steps):
        u = stepper(u)
        # If it blows up, stop early
        if not torch.isfinite(u).all():
            break
        out.append(u.clone())

    traj = torch.cat(out, dim=0)  # (T, B, Nx)
    if squeeze:
        traj = traj[:, 0, :]      # (T, Nx)
    return traj

# ============================================================
# 2) Ground-truth KS integrator (numpy RK4, periodic FD matrices)
# ============================================================

def ks_rhs_np(u, Dx_mat, Dxx_mat, Dxxxx_mat):
    ux    = Dx_mat @ u
    uxx   = Dxx_mat @ u
    uxxxx = Dxxxx_mat @ u
    return -(u * ux) - uxx - uxxxx

def rollout_truth_np(u0, dt, n_steps, Dx_mat, Dxx_mat, Dxxxx_mat, method="rk4"):
    u = u0.copy()
    out = [u.copy()]

    for _ in range(n_steps):
        if method.lower() == "euler":
            u = u + dt * ks_rhs_np(u, Dx_mat, Dxx_mat, Dxxxx_mat)
        else:
            k1 = ks_rhs_np(u, Dx_mat, Dxx_mat, Dxxxx_mat)
            k2 = ks_rhs_np(u + 0.5*dt*k1, Dx_mat, Dxx_mat, Dxxxx_mat)
            k3 = ks_rhs_np(u + 0.5*dt*k2, Dx_mat, Dxx_mat, Dxxxx_mat)
            k4 = ks_rhs_np(u + dt*k3, Dx_mat, Dxx_mat, Dxxxx_mat)
            u  = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        if not np.isfinite(u).all():
            break
        out.append(u.copy())

    return np.stack(out, axis=0)  # (T, Nx)

# ============================================================
# 3) Diagnostics: power spectrum, energy PDF, KL
# ============================================================

def spatial_power_spectrum(traj, L):
    """
    traj: (T, Nx) numpy
    returns: k (Nx//2+1,), S (Nx//2+1,)
    """
    T, Nx = traj.shape
    # rfft over space axis
    Uhat = np.fft.rfft(traj, axis=1)  # (T, Nk)
    S = np.mean(np.abs(Uhat)**2, axis=0) / Nx  # average over time
    k = 2*np.pi*np.fft.rfftfreq(Nx, d=L/Nx)
    return k, S

def energy_time_series(traj):
    """E(t)=<u^2>_x"""
    return np.mean(traj**2, axis=1)

def kl_divergence_hist(p_samples, q_samples, bins=100, eps=1e-12):
    """
    KL(P||Q) estimated via shared histogram support.
    """
    lo = min(np.min(p_samples), np.min(q_samples))
    hi = max(np.max(p_samples), np.max(q_samples))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.inf

    edges = np.linspace(lo, hi, bins+1)
    p, _ = np.histogram(p_samples, bins=edges, density=True)
    q, _ = np.histogram(q_samples, bins=edges, density=True)
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))

# ============================================================
# 4) Largest Lyapunov exponent (Benettin: two trajectories)
# ============================================================

def largest_lyapunov_exponent_torch(
    dynamics_fn, u0, dt, n_steps_total,
    reorthonorm_every=10,
    delta0=1e-7,
    method="rk4",
):
    """
    Estimates largest Lyapunov exponent from learned dynamics.
    Uses two trajectories u and v=u+perturb, periodically renormalizing.
    """
    device = u0.device
    Nx = u0.numel()

    u = u0.clone()
    # random perturbation
    d = torch.randn(Nx, device=device)
    d = d / (torch.norm(d) + 1e-12)
    v = u + delta0 * d

    log_sum = 0.0
    count = 0
    steps = 0

    while steps < n_steps_total:
        # integrate both for m steps
        m = min(reorthonorm_every, n_steps_total - steps)

        traj_u = rollout_torch(dynamics_fn, u, dt, m, method=method)  # (m+1,Nx)
        traj_v = rollout_torch(dynamics_fn, v, dt, m, method=method)

        # if blew up, abort
        if traj_u.shape[0] < m+1 or traj_v.shape[0] < m+1:
            return np.nan

        u = traj_u[-1]
        v = traj_v[-1]

        diff = v - u
        dist = torch.norm(diff)
        if not torch.isfinite(dist) or dist.item() == 0.0:
            return np.nan

        log_sum += torch.log(dist / delta0).item()
        count += 1

        # renormalize
        diff = diff / dist
        v = u + delta0 * diff

        steps += m

    # lambda = (1/(total_time)) * sum log(dist/delta0)
    total_time = (count * reorthonorm_every) * dt
    if total_time <= 0:
        return np.nan
    return float(log_sum / total_time)

# ============================================================
# 5) Correlation dimension D2 (Grassberger–Procaccia on PCA)
# ============================================================

def pca_fit_transform(X, n_components=6):
    """
    X: (T, Nx) numpy
    returns: Z: (T, n_components) PCA coords
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD on (T,Nx) is fine for moderate sizes
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :n_components] * S[:n_components]
    return Z

def correlation_dimension_gp(Z, n_r=25, r_min=None, r_max=None, sample=2000, seed=0):
    """
    Z: (T, d) numpy (PCA coordinates)
    Estimates D2 from slope of log C(r) vs log r over a mid-range.
    """
    rng = np.random.default_rng(seed)
    T = Z.shape[0]
    if T < 10:
        return np.nan

    # subsample points for O(N^2) distance cost
    idx = rng.choice(T, size=min(sample, T), replace=False)
    Y = Z[idx]  # (M,d)
    M = Y.shape[0]

    # pairwise distances (upper triangle)
    # (M,M,d) might be large; do blockwise
    dists = []
    block = 500
    for i0 in range(0, M, block):
        i1 = min(M, i0+block)
        A = Y[i0:i1]
        # compute distances to all (including self) then take upper-tri later
        D = np.sqrt(np.sum((A[:, None, :] - Y[None, :, :])**2, axis=-1))
        for i in range(i0, i1):
            dists.append(D[i - i0, i+1:])  # only j>i
    dists = np.concatenate(dists)
    dists = dists[np.isfinite(dists)]
    if dists.size == 0:
        return np.nan

    if r_min is None:
        r_min = np.quantile(dists, 0.01)
    if r_max is None:
        r_max = np.quantile(dists, 0.50)  # avoid saturation

    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min <= 0 or r_max <= r_min:
        return np.nan

    rs = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    # correlation integral C(r) = fraction of pairs with dist < r
    C = np.array([np.mean(dists < r) for r in rs], dtype=np.float64)

    # pick a fitting region where 0 < C < 1 and away from extremes
    mask = (C > 1e-4) & (C < 1e-1)
    if np.sum(mask) < 5:
        # fallback to a broader region
        mask = (C > 1e-5) & (C < 1e-0)

    if np.sum(mask) < 5:
        return np.nan

    x = np.log(rs[mask])
    y = np.log(C[mask])
    # linear fit slope
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)

# ============================================================
# 6) One evaluation function per model vs truth
# ============================================================

def evaluate_attractor_metrics(
    *,
    model_dynamics_fn_torch,
    u0_np,
    dt,
    n_steps,
    burn_in_steps,
    L,
    Dx_mat, Dxx_mat, Dxxxx_mat,
    method="rk4",
    pca_components=6,
    hist_bins=100,
    seed=0,
):
    """
    Returns metrics for model and truth, plus comparisons (KL, spectrum error).
    """
    # --- truth rollout
    truth_traj = rollout_truth_np(u0_np, dt, n_steps, Dx_mat, Dxx_mat, Dxxxx_mat, method=method)
    if truth_traj.shape[0] <= burn_in_steps + 10:
        return {"ok": False, "reason": "truth rollout too short"}

    truth_clim = truth_traj[burn_in_steps:, :]  # (Tclim,Nx)

    # --- model rollout
    u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device)
    model_traj_t = rollout_torch(model_dynamics_fn_torch, u0_t, dt, n_steps, method=method)
    model_traj = model_traj_t.detach().cpu().numpy()
    if model_traj.shape[0] <= burn_in_steps + 10:
        return {"ok": False, "reason": "model rollout blew up/too short"}

    model_clim = model_traj[burn_in_steps:, :]

    # --- power spectrum
    k, S_truth = spatial_power_spectrum(truth_clim, L)
    _, S_model = spatial_power_spectrum(model_clim, L)

    # spectrum distance (log-space L2)
    eps = 1e-12
    spec_err = float(np.sqrt(np.mean((np.log(S_model + eps) - np.log(S_truth + eps))**2)))

    # --- energy series + KL divergence on energy distribution
    E_truth = energy_time_series(truth_clim)
    E_model = energy_time_series(model_clim)
    kl_E = kl_divergence_hist(E_truth, E_model, bins=hist_bins)

    # --- correlation dimension on PCA embedding
    Z_truth = pca_fit_transform(truth_clim, n_components=pca_components)
    Z_model = pca_fit_transform(model_clim, n_components=pca_components)
    D2_truth = correlation_dimension_gp(Z_truth, seed=seed)
    D2_model = correlation_dimension_gp(Z_model, seed=seed+1)
    D2_abs_err = float(np.abs(D2_model - D2_truth)) if np.isfinite(D2_truth) and np.isfinite(D2_model) else np.inf

    return {
        "ok": True,
        "truth": {
            "E_mean": float(np.mean(E_truth)),
            "E_std":  float(np.std(E_truth)),
            "D2":     float(D2_truth),
            "k":      k,
            "S":      S_truth,
        },
        "model": {
            "E_mean": float(np.mean(E_model)),
            "E_std":  float(np.std(E_model)),
            "D2":     float(D2_model),
            "k":      k,
            "S":      S_model,
        },
        "compare": {
            "spec_logrmse": spec_err,
            "kl_energy":    float(kl_E),
            "D2_abs_err":   D2_abs_err,
        }
    }

# ============================================================
# 7) Evaluate all 4 conditions (adds LLE too)
# ============================================================

def evaluate_ablation_longrun(
    ablation_results,
    *,
    u0_np,
    dt,
    n_steps=5000,
    burn_in_steps=1000,
    L=64.0,
    Dx_mat=None, Dxx_mat=None, Dxxxx_mat=None,
    method="rk4",
    lle_steps=4000,
    lle_reorth=10,
    lle_delta0=1e-7,
    seed=0,
):
    summary = {}

    for name, res in ablation_results.items():
        print("\n" + "="*80)
        print(f"Evaluating attractor metrics: {name}")
        print("="*80)

        model = res["model"]

        # IMPORTANT: use the SAME dynamics_fn used during training rollout loss
        # We stored only the model; we rebuild a consistent dynamics_fn here:
        ds = None
        # If you kept the dataset or X_mean/X_std somewhere, use that.
        # We'll assume you can reconstruct it from res or saved objects.
        # Practical approach: attach `res["dynamics_fn"]` in your training runner.
        if "dynamics_fn" not in res:
            raise ValueError(
                "Please modify your training runner to store res['dynamics_fn'].\n"
                "Easiest fix: in train_one_condition(), include 'dynamics_fn' in out dict."
            )

        dynamics_fn = res["dynamics_fn"]

        # --- LLE (largest)
        try:
            u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device)
            lle = largest_lyapunov_exponent_torch(
                dynamics_fn, u0_t, dt,
                n_steps_total=lle_steps,
                reorthonorm_every=lle_reorth,
                delta0=lle_delta0,
                method=method,
            )
        except Exception as e:
            lle = np.nan
            print("LLE failed:", e)

        # --- Attractor stats (spectrum, D2, KL)
        stats = evaluate_attractor_metrics(
            model_dynamics_fn_torch=dynamics_fn,
            u0_np=u0_np,
            dt=dt,
            n_steps=n_steps,
            burn_in_steps=burn_in_steps,
            L=L,
            Dx_mat=Dx_mat, Dxx_mat=Dxx_mat, Dxxxx_mat=Dxxxx_mat,
            method=method,
            seed=seed,
        )

        summary[name] = {
            "lle": float(lle) if np.isfinite(lle) else np.nan,
            "stats": stats,
        }

        if stats["ok"]:
            print(f"LLE: {summary[name]['lle']:.5f}")
            print(f"spec_logrmse: {stats['compare']['spec_logrmse']:.5e}")
            print(f"KL(E):        {stats['compare']['kl_energy']:.5e}")
            print(f"D2 truth/model: {stats['truth']['D2']:.3f} / {stats['model']['D2']:.3f}")
        else:
            print("Stats failed:", stats["reason"])

    print("\n" + "#"*80)
    print("FINAL SUMMARY (lower is better for spec_logrmse, KL, D2_err; LLE should match truth scale)")
    for name, out in summary.items():
        st = out["stats"]
        if st["ok"]:
            print(
                f"{name:30s} | LLE={out['lle']:.4f} | "
                f"spec_logrmse={st['compare']['spec_logrmse']:.3e} | "
                f"KL(E)={st['compare']['kl_energy']:.3e} | "
                f"D2_err={st['compare']['D2_abs_err']:.3e}"
            )
        else:
            print(f"{name:30s} | FAILED ({st['reason']})")
    print("#"*80)

    return summary

u0_np = U_true[0].copy()   # or pick another time index on attractor

eval_summary = evaluate_ablation_longrun(
    ablation_results,
    u0_np=u0_np,
    dt=dt,
    n_steps=2000,          # longer = better stats
    burn_in_steps=200,
    L=L,
    Dx_mat=Dx_mat,
    Dxx_mat=Dxx_mat,
    Dxxxx_mat=Dxxxx_mat,
    method="rk4",
    lle_steps=1000,
    lle_reorth=10,
    lle_delta0=1e-7,
    seed=0,
)


# In[8]:


@torch.no_grad()
def rollout_torch(dynamics_fn, u0, dt, n_steps, method="rk4"):
    """
    dynamics_fn: f(u)->u_t, supports u shape (B,Nx) or (Nx,)
    returns:
      if u0 is (Nx,)  -> (T, Nx)
      if u0 is (B,Nx) -> (T, B, Nx)
    """
    if u0.dim() == 1:
        u = u0.unsqueeze(0)  # (1,Nx)
        squeeze = True
    else:
        u = u0
        squeeze = False

    def euler_step(u):
        return u + dt * dynamics_fn(u)

    def rk4_step(u):
        k1 = dynamics_fn(u)
        k2 = dynamics_fn(u + 0.5 * dt * k1)
        k3 = dynamics_fn(u + 0.5 * dt * k2)
        k4 = dynamics_fn(u + dt * k3)
        return u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    stepper = rk4_step if method.lower() == "rk4" else euler_step

    out = [u.clone()]  # each is (B,Nx)
    for _ in range(n_steps):
        u = stepper(u)
        if not torch.isfinite(u).all():
            break
        out.append(u.clone())

    traj = torch.stack(out, dim=0)  # (T, B, Nx)

    if squeeze:
        traj = traj[:, 0, :]        # (T, Nx)
    return traj

import numpy as np
import torch

# ----------------------------
# Spectrum, energy, KL, PCA-D2
# ----------------------------
def spatial_power_spectrum(traj, L):
    """
    traj: (T, Nx) numpy
    """
    T, Nx = traj.shape
    Uhat = np.fft.rfft(traj, axis=1)
    S = np.mean(np.abs(Uhat)**2, axis=0) / Nx
    k = 2*np.pi*np.fft.rfftfreq(Nx, d=L/Nx)
    return k, S

def energy_time_series(traj):
    return np.mean(traj**2, axis=1)

def kl_divergence_hist(p_samples, q_samples, bins=100, eps=1e-12):
    lo = min(np.min(p_samples), np.min(q_samples))
    hi = max(np.max(p_samples), np.max(q_samples))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return np.inf
    edges = np.linspace(lo, hi, bins+1)
    p, _ = np.histogram(p_samples, bins=edges, density=True)
    q, _ = np.histogram(q_samples, bins=edges, density=True)
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return float(np.sum(p * np.log(p / q)))

def pca_fit_transform(X, n_components=6):
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = U[:, :n_components] * S[:n_components]
    return Z

def correlation_dimension_gp(Z, n_r=25, r_min=None, r_max=None, sample=2000, seed=0):
    rng = np.random.default_rng(seed)
    T = Z.shape[0]
    if T < 50:
        return np.nan
    idx = rng.choice(T, size=min(sample, T), replace=False)
    Y = Z[idx]
    M = Y.shape[0]

    # pairwise distances (upper triangle), blockwise
    dists = []
    block = 500
    for i0 in range(0, M, block):
        i1 = min(M, i0+block)
        A = Y[i0:i1]
        D = np.sqrt(np.sum((A[:, None, :] - Y[None, :, :])**2, axis=-1))
        for i in range(i0, i1):
            dists.append(D[i - i0, i+1:])
    dists = np.concatenate(dists)
    dists = dists[np.isfinite(dists)]
    if dists.size == 0:
        return np.nan

    if r_min is None:
        r_min = np.quantile(dists, 0.01)
    if r_max is None:
        r_max = np.quantile(dists, 0.50)

    if not np.isfinite(r_min) or not np.isfinite(r_max) or r_min <= 0 or r_max <= r_min:
        return np.nan

    rs = np.logspace(np.log10(r_min), np.log10(r_max), n_r)
    C = np.array([np.mean(dists < r) for r in rs], dtype=np.float64)

    # fit region (avoid saturation/extremes)
    mask = (C > 1e-4) & (C < 1e-1)
    if np.sum(mask) < 5:
        mask = (C > 1e-5) & (C < 1e-0)
    if np.sum(mask) < 5:
        return np.nan

    x = np.log(rs[mask])
    y = np.log(C[mask])
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)

# ----------------------------
# Largest Lyapunov (Benettin)
# ----------------------------
def largest_lyapunov_exponent_torch(
    dynamics_fn, u0, dt, n_steps_total,
    reorthonorm_every=10,
    delta0=1e-7,
    method="rk4",
):
    device = u0.device
    Nx = u0.numel()

    u = u0.clone()
    d = torch.randn(Nx, device=device)
    d = d / (torch.norm(d) + 1e-12)
    v = u + delta0 * d

    log_sum = 0.0
    count = 0
    steps = 0

    while steps < n_steps_total:
        m = min(reorthonorm_every, n_steps_total - steps)

        traj_u = rollout_torch(dynamics_fn, u, dt, m, method=method)  # (m+1,Nx)
        traj_v = rollout_torch(dynamics_fn, v, dt, m, method=method)

        if traj_u.shape[0] < m+1 or traj_v.shape[0] < m+1:
            return np.nan

        u = traj_u[-1]
        v = traj_v[-1]

        diff = v - u
        dist = torch.norm(diff)
        if (not torch.isfinite(dist)) or dist.item() == 0.0:
            return np.nan

        log_sum += torch.log(dist / delta0).item()
        count += 1

        diff = diff / dist
        v = u + delta0 * diff

        steps += m

    total_time = (count * reorthonorm_every) * dt
    return float(log_sum / total_time) if total_time > 0 else np.nan

# ----------------------------
# Evaluate 1 model vs truth using U_true
# ----------------------------
def evaluate_model_vs_truth(
    dynamics_fn,
    *,
    U_true_np,    # (Nt,Nx) truth from SciPy
    dt,
    L,
    u0_np,
    n_steps_model=8000,
    burn_in_steps=2000,
    truth_start_idx=0,
    hist_bins=100,
    pca_components=6,
    seed=0,
    method="rk4",
    lle_steps=6000,
    lle_reorth=10,
    lle_delta0=1e-7,
):
    # ---- truth window from U_true (no integration)
    end = truth_start_idx + burn_in_steps + (n_steps_model - burn_in_steps) + 1
    if end > U_true_np.shape[0]:
        return {"ok": False, "reason": "Not enough truth samples in U_true for requested window."}

    truth_traj = U_true_np[truth_start_idx:end, :]            # (T,Nx)
    truth_clim = truth_traj[burn_in_steps:, :]

    # ---- model rollout
    u0_t = torch.tensor(u0_np, dtype=torch.float32, device=device)
    model_traj_t = rollout_torch(dynamics_fn, u0_t, dt, n_steps_model, method=method)
    model_traj = model_traj_t.detach().cpu().numpy()
    if model_traj.shape[0] <= burn_in_steps + 50:
        return {"ok": False, "reason": "Model rollout blew up/too short."}
    model_clim = model_traj[burn_in_steps:, :]

    # ---- power spectrum
    k, S_truth = spatial_power_spectrum(truth_clim, L)
    _, S_model = spatial_power_spectrum(model_clim, L)
    eps = 1e-12
    spec_logrmse = float(np.sqrt(np.mean((np.log(S_model+eps) - np.log(S_truth+eps))**2)))

    # ---- KL on energy distribution
    E_truth = energy_time_series(truth_clim)
    E_model = energy_time_series(model_clim)
    kl_E = kl_divergence_hist(E_truth, E_model, bins=hist_bins)

    # ---- correlation dimension on PCA embedding
    Z_truth = pca_fit_transform(truth_clim, n_components=pca_components)
    Z_model = pca_fit_transform(model_clim, n_components=pca_components)
    D2_truth = correlation_dimension_gp(Z_truth, seed=seed)
    D2_model = correlation_dimension_gp(Z_model, seed=seed+1)
    D2_err = float(np.abs(D2_model - D2_truth)) if np.isfinite(D2_truth) and np.isfinite(D2_model) else np.inf

    # ---- LLE (largest)
    lle = largest_lyapunov_exponent_torch(
        dynamics_fn,
        u0_t,
        dt,
        n_steps_total=lle_steps,
        reorthonorm_every=lle_reorth,
        delta0=lle_delta0,
        method=method,
    )

    return {
        "ok": True,
        "lle": float(lle) if np.isfinite(lle) else np.nan,
        "compare": {
            "spec_logrmse": spec_logrmse,
            "kl_energy": kl_E,
            "D2_truth": float(D2_truth),
            "D2_model": float(D2_model),
            "D2_abs_err": D2_err,
        },
        "spectrum": {"k": k, "S_truth": S_truth, "S_model": S_model},
        "energy": {"E_truth": E_truth, "E_model": E_model},
    }

def evaluate_ablation_longrun_with_truth(
    ablation_results,
    *,
    U_true_np,
    dt,
    L,
    u0_np=None,
    n_steps_model=8000,
    burn_in_steps=2000,
    truth_start_idx=0,
    method="rk4",
):
    if u0_np is None:
        u0_np = U_true_np[truth_start_idx].copy()

    out = {}
    for name, res in ablation_results.items():
        print("\n" + "="*80)
        print("Evaluating:", name)
        print("="*80)

        if "dynamics_fn" not in res:
            raise ValueError("Missing res['dynamics_fn']. Store it during training.")

        metrics = evaluate_model_vs_truth(
            res["dynamics_fn"],
            U_true_np=U_true_np,
            dt=dt,
            L=L,
            u0_np=u0_np,
            n_steps_model=n_steps_model,
            burn_in_steps=burn_in_steps,
            truth_start_idx=truth_start_idx,
            method=method,
        )
        out[name] = metrics

        if metrics["ok"]:
            c = metrics["compare"]
            print(f"LLE:          {metrics['lle']:.5f}")
            print(f"spec_logrmse: {c['spec_logrmse']:.3e}")
            print(f"KL(E):        {c['kl_energy']:.3e}")
            print(f"D2 truth/model: {c['D2_truth']:.3f} / {c['D2_model']:.3f}  (abs err {c['D2_abs_err']:.3e})")
        else:
            print("FAILED:", metrics["reason"])

    print("\n" + "#"*80)
    print("SUMMARY")
    for name, m in out.items():
        if not m["ok"]:
            print(f"{name:30s} | FAILED")
            continue
        c = m["compare"]
        print(
            f"{name:30s} | LLE={m['lle']:.4f} | "
            f"spec_logrmse={c['spec_logrmse']:.3e} | "
            f"KL(E)={c['kl_energy']:.3e} | "
            f"D2_err={c['D2_abs_err']:.3e}"
        )
    print("#"*80)

    return out

# run:
eval_summary = evaluate_ablation_longrun_with_truth(
    ablation_results,
    U_true_np=U_true,  # from SciPy solve_ivp
    dt=dt,
    L=L,
    n_steps_model=3000,
    burn_in_steps=200,
    truth_start_idx=0,
    method="rk4",
)


# In[9]:


eval_summary = evaluate_ablation_longrun_with_truth(
    ablation_results,
    U_true_np=U_true,
    dt=dt,
    L=L,
    n_steps_model=1000,     # <= len(U_true)-1
    burn_in_steps=100,      # keep small
    truth_start_idx=0,
    method="rk4",
)


# In[ ]:





# In[ ]:





# In[19]:


# ============================================================
# Rollout + plot for DISCOVERED KS equation (same premium style)
#   u_t = -1.274*u*u_x - 1.052*u_xx - 1.92*u_xxxx
# ============================================================
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

# --- assumes already in scope from your earlier cells ---
# U_true, t, x, dx, dt, device
# and the FD matrices you already built:
# Dx_t, Dxx_t, Dxxxx_t  (torch tensors on device)

os.makedirs("figs", exist_ok=True)

# -----------------------
# 0) Config
# -----------------------
dt_save = float(dt)
n_sub   = 10               # increase (e.g. 20-50) if you see instability
h       = dt_save / n_sub
Nt      = len(t)
Nx      = len(x)
print_every = 50

# Discovered coefficients
c_adv  = 1.274   # multiplies u*u_x with a minus sign
c_xx   = 1.052   # multiplies u_xx with a minus sign
c_xxxx = 1.92    # multiplies u_xxxx with a minus sign

# -----------------------
# 1) Discovered dynamics (uses your same periodic FD operators)
# -----------------------
def dynamics_fn_discovered(state: torch.Tensor) -> torch.Tensor:
    """
    state: (B, Nx) or (Nx,)
    returns: (B, Nx)
    """
    if state.dim() == 1:
        state = state.unsqueeze(0)

    u = state
    ux    = u @ Dx_t.T
    uxx   = u @ Dxx_t.T
    uxxxx = u @ Dxxxx_t.T

    ut = -(c_adv * (u * ux) + c_xx * uxx + c_xxxx * uxxxx)
    return ut

# -----------------------
# 2) Rollout (RK4 with substeps)
# -----------------------
def rk4_step(f, y, h):
    k1 = f(y)
    k2 = f(y + 0.5*h*k1)
    k3 = f(y + 0.5*h*k2)
    k4 = f(y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

u0_torch = torch.tensor(U_true[0], dtype=torch.float32, device=device).unsqueeze(0)  # (1, Nx)
state = u0_torch.clone()

U_disc = np.zeros_like(U_true, dtype=np.float64)
U_disc[0] = U_true[0]

with torch.no_grad():
    for i in range(1, Nt):
        for _ in range(n_sub):
            state = rk4_step(dynamics_fn_discovered, state, h)

        U_disc[i] = state.squeeze(0).cpu().numpy()

        if i % print_every == 0:
            s = state.squeeze(0)
            print(f"[{i:4d}/{Nt-1}]  u: mean={s.mean().item():+.3e}, std={s.std().item():.3e}, "
                  f"min={s.min().item():+.3e}, max={s.max().item():+.3e}")

# -----------------------
# 3) Plot (same premium style)
# -----------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,  'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.direction': 'in',   'ytick.direction': 'in',
    'xtick.major.pad': 4,      'ytick.major.pad': 4,
    'xtick.top': True,         'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

colors_custom = [
    '#0a0e27', '#0d1b4a', '#1b3a6b', '#1f6f8b', '#2d9e8f', '#6ec6a0',
    '#c8e6a0', '#f5e663', '#f5a623', '#e8573a', '#c22e3a', '#7a1a40',
]
cmap_ks = LinearSegmentedColormap.from_list('ks_premium', colors_custom, N=512)

# Use SAME color scale as ground truth (fair comparison)
vmin, vmax = np.percentile(U_true, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(U_disc.T, cmap=cmap_ks, norm=norm, blend_mode='soft',
               vert_exag=0.08, dx=dx, dy=dt_save, fraction=1.2)

T_grid, X_grid = np.meshgrid(t, x)

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(rgb, extent=[t[0], t[-1], x[0], x[-1]],
          aspect='auto', origin='lower', interpolation='bilinear')
ax.contour(T_grid, X_grid, U_disc.T, levels=np.linspace(vmin, vmax, 18),
           colors='white', linewidths=0.15, alpha=0.25)

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)

ax.set_title(
    r'Discovered Model Rollout: $u_t=-1.274\,u\,u_x-1.052\,u_{xx}-1.92\,u_{xxxx}$',
    fontsize=12, pad=12, fontweight='medium'
)

ax.text(0.98, 0.96, rf'$L=64,\;\; N_x={Nx},\;\; n_{{sub}}={n_sub}$',
        transform=ax.transAxes,
        fontsize=8, color='white', ha='right', va='top', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none'))

sm = cm.ScalarMappable(cmap=cmap_ks, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('figs/ks_discovered_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/ks_discovered_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

print("Done! Saved figs/ks_discovered_prediction.png and figs/ks_discovered_prediction.pdf")


# In[16]:


import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource
from scipy.integrate import solve_ivp

# ============================================================
# Assumes everything from your model script is already in scope:
#   kan_pde, dynamics_fn, Dx_t, Dxx_t, Dxxxx_t,
#   X_mean, X_std, U_true, t, x, dx, dt, device
# ============================================================

# --- Roll out the KAN model from the same IC ---
u0_torch = torch.tensor(U_true[0], dtype=torch.float32, device=device).unsqueeze(0)  # (1, Nx)
Nt = len(t)
dt_rollout = float(dt)

U_kan = np.zeros_like(U_true)  # (Nt, Nx)
U_kan[0] = U_true[0]

state = u0_torch.clone()
kan_pde.eval()

with torch.no_grad():
    for i in range(1, Nt):
        # RK4 step
        k1 = dynamics_fn(state)
        k2 = dynamics_fn(state + 0.5 * dt_rollout * k1)
        k3 = dynamics_fn(state + 0.5 * dt_rollout * k2)
        k4 = dynamics_fn(state + dt_rollout * k3)
        state = state + (dt_rollout / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        U_kan[i] = state.squeeze(0).cpu().numpy()

        if i % 50 == 0:
            print(f"Rollout step {i}/{Nt-1}")

# --- Plot (identical style to your ground-truth plot) ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['CMU Serif', 'Computer Modern Roman', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,  'ytick.major.width': 0.5,
    'xtick.minor.width': 0.35, 'ytick.minor.width': 0.35,
    'xtick.direction': 'in',   'ytick.direction': 'in',
    'xtick.major.pad': 4,      'ytick.major.pad': 4,
    'xtick.top': True,         'ytick.right': True,
    'xtick.minor.visible': True, 'ytick.minor.visible': True,
})

colors_custom = [
    '#0a0e27', '#0d1b4a', '#1b3a6b', '#1f6f8b', '#2d9e8f', '#6ec6a0',
    '#c8e6a0', '#f5e663', '#f5a623', '#e8573a', '#c22e3a', '#7a1a40',
]
cmap_ks = LinearSegmentedColormap.from_list('ks_premium', colors_custom, N=512)

# Use the SAME color scale as ground truth for fair comparison
vmin, vmax = np.percentile(U_true, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(U_kan.T, cmap=cmap_ks, norm=norm, blend_mode='soft',
               vert_exag=0.08, dx=dx, dy=dt, fraction=1.2)

T_grid, X_grid = np.meshgrid(t, x)

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(rgb, extent=[t[0], t[-1], x[0], x[-1]],
          aspect='auto', origin='lower', interpolation='bilinear')
ax.contour(T_grid, X_grid, U_kan.T, levels=np.linspace(vmin, vmax, 18),
           colors='white', linewidths=0.15, alpha=0.25)

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)
ax.set_title(r'KAN Model Prediction: $\,u_t = -u\,u_x - u_{xx} - u_{xxxx}$',
             fontsize=12, pad=12, fontweight='medium')
ax.text(0.98, 0.96, r'$L=64,\;\; N_x=128$', transform=ax.transAxes,
        fontsize=8, color='white', ha='right', va='top', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none'))

sm = cm.ScalarMappable(cmap=cmap_ks, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('figs/ks_kan_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/ks_kan_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()
print("Done! Saved ks_kan_prediction.png/pdf")


# In[ ]:


import numpy as np
import torch
import matplotlib.pyplot as plt

# -----------------------------
# 1) Define the discovered PDE RHS (from your symbolic equation)
#    u_t = a * u*u_x + b * u_xx + c * u_xxxx + d
# -----------------------------
a = -0.839
b = -0.458
c = -1.199
d = -0.001

@torch.no_grad()
def ks_discovered_rhs(u):
    """
    u: (B, Nx) torch tensor
    returns: (B, Nx) = u_t
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)

    ux    = u @ Dx_t.T
    uxx   = u @ Dxx_t.T
    uxxxx = u @ Dxxxx_t.T

    ut = a * (u * ux) + b * uxx + c * uxxxx + d
    return ut

# -----------------------------
# 2) RK4 rollout (with substeps)
# -----------------------------
@torch.no_grad()
def rollout_rk4_substeps(u0, t_grid, rhs_fn, n_sub=20):
    """
    u0: (Nx,) or (1,Nx)
    t_grid: (Nt,) torch tensor
    rhs_fn: function(u)->u_t
    """
    if u0.dim() == 1:
        u = u0[None, :].clone()
    else:
        u = u0.clone()

    Nt = t_grid.numel()
    Nx = u.shape[-1]
    out = torch.zeros((Nt, Nx), device=u.device, dtype=u.dtype)
    out[0] = u[0]

    for n in range(Nt - 1):
        dt_out = (t_grid[n+1] - t_grid[n]).item()
        dt = dt_out / n_sub

        for _ in range(n_sub):
            k1 = rhs_fn(u)
            k2 = rhs_fn(u + 0.5 * dt * k1)
            k3 = rhs_fn(u + 0.5 * dt * k2)
            k4 = rhs_fn(u + dt * k3)
            u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        out[n+1] = u[0]

    return out  # (Nt, Nx)

# -----------------------------
# 3) Run simulation on the same grid/time as your data
# -----------------------------
t_grid = torch.tensor(t, dtype=torch.float32, device=device)  # (Nt,)
u0_sim = torch.tensor(U_true[0], dtype=torch.float32, device=device)  # (Nx,)

U_sim = rollout_rk4_substeps(u0_sim, t_grid, ks_discovered_rhs, n_sub=20)  # (Nt, Nx)

U_sim_np  = U_sim.detach().cpu().numpy()
U_true_np = np.asarray(U_true)

# -----------------------------
# 4) Plot space-time for discovered PDE
# -----------------------------
plt.figure(figsize=(10, 3))
plt.contourf(t, x, U_sim_np.T, levels=201, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x")
plt.title("Space-time rollout from discovered PDE")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()

# -----------------------------
# 5) (Optional) Compare to ground truth + plot error
# -----------------------------
# Use matched color scale for fair visual comparison
v = np.nanpercentile(np.abs(U_true_np), 99.5) + 1e-12

plt.figure(figsize=(10, 3))
plt.contourf(t, x, U_true_np.T, levels=201, vmin=-v, vmax=v, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x")
plt.title("Ground truth space-time")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10, 3))
plt.contourf(t, x, U_sim_np.T, levels=201, vmin=-v, vmax=v, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x")
plt.title("Discovered PDE space-time (matched scale)")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()

err = U_sim_np - U_true_np
v_err = np.nanpercentile(np.abs(err), 99) + 1e-12
plt.figure(figsize=(10, 3))
plt.contourf(t, x, err.T, levels=201, vmin=-v_err, vmax=v_err, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x")
plt.title("Error: discovered - true (clipped at 99th pct)")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()

# Normalized RMSE vs time (forecast metric)
rmse_t = np.sqrt(np.mean(err**2, axis=1))
rms_true_t = np.sqrt(np.mean(U_true_np**2, axis=1)) + 1e-12
nrmse_t = rmse_t / rms_true_t

plt.figure(figsize=(8, 3))
plt.plot(t, nrmse_t)
plt.yscale("log")
plt.xlabel("t"); plt.ylabel("nRMSE = RMSE/RMS(true)")
plt.title("Forecast nRMSE vs time (discovered PDE)")
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


# -----------------------
# Rollout + plotting
# -----------------------
import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def rollout_rk4(u0, t_grid, dynamics_fn):
    """
    Fixed-step RK4 rollout on a provided time grid.
    u0:     (Nx,) or (1,Nx) torch tensor
    t_grid: (Nt,) torch tensor, increasing
    returns: (Nt, Nx) torch tensor
    """
    if u0.dim() == 1:
        u = u0.unsqueeze(0)  # (1, Nx)
    else:
        u = u0
    u = u.clone()

    Nt = t_grid.numel()
    Nx = u.shape[-1]
    out = torch.zeros((Nt, Nx), device=u.device, dtype=u.dtype)
    out[0] = u[0]

    for n in range(Nt - 1):
        dt = (t_grid[n + 1] - t_grid[n]).item()

        k1 = dynamics_fn(u)                       # (1, Nx)
        k2 = dynamics_fn(u + 0.5 * dt * k1)
        k3 = dynamics_fn(u + 0.5 * dt * k2)
        k4 = dynamics_fn(u + dt * k3)

        u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        out[n + 1] = u[0]

    return out  # (Nt, Nx)

# --- Make sure model is in eval mode for rollout ---
kan_pde.eval()

# --- Choose rollout IC/time grid (full training horizon) ---
t_grid = torch.tensor(t, dtype=torch.float32, device=device)          # (Nt,)
u0_roll = torch.tensor(U_true[0], dtype=torch.float32, device=device) # (Nx,)

# --- Rollout ---
U_pred = rollout_rk4(u0_roll, t_grid, dynamics_fn)                    # (Nt, Nx)

# --- Ground truth tensor for comparison ---
U_true_t = torch.tensor(U_true, dtype=torch.float32, device=device)   # (Nt, Nx)

# --- Convert to numpy for plotting ---
U_pred_np = U_pred.detach().cpu().numpy()
U_true_np = U_true_t.detach().cpu().numpy()

# -----------------------
# 1) Space-time contour plots
# -----------------------
plt.figure(figsize=(10, 3))
plt.contourf(t, x, U_true_np.T, levels=201, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x"); plt.title("Ground truth u(t,x)")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10, 3))
plt.contourf(t, x, U_pred_np.T, levels=201, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x"); plt.title("KAN rollout prediction û(t,x)")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()

# -----------------------
# 2) Error contour + time error curve
# -----------------------
err = U_pred_np - U_true_np
plt.figure(figsize=(10, 3))
plt.contourf(t, x, err.T, levels=201, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x"); plt.title("Rollout error: û - u")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()

rmse_t = np.sqrt(np.mean(err**2, axis=1))  # (Nt,)
plt.figure(figsize=(8, 3))
plt.plot(t, rmse_t)
plt.xlabel("t"); plt.ylabel("RMSE over x"); plt.title("Rollout RMSE vs time")
plt.tight_layout()
plt.show()
plt.close()

# -----------------------
# 3) Optional: compare a few spatial snapshots
# -----------------------
snap_ts = [0.0, 2.0, 6.0, 10.0, 20.0]
snap_idx = [int(round(tt / dt)) for tt in snap_ts]

plt.figure(figsize=(10, 6))
for i, idx in enumerate(snap_idx, 1):
    plt.subplot(len(snap_idx), 1, i)
    plt.plot(x, U_true_np[idx], label="true")
    plt.plot(x, U_pred_np[idx], label="pred", linestyle="--")
    plt.ylabel(f"t={t[idx]:.1f}")
    if i == 1:
        plt.legend(loc="upper right")
plt.xlabel("x")
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

# U_true_np: (Nt, Nx)
# U_pred_np: (Nt, Nx)
# t:         (Nt,)

err = U_pred_np - U_true_np  # (Nt, Nx)

# --- RMSE over space at each time ---
rmse_t = np.sqrt(np.mean(err**2, axis=1))  # (Nt,)

# --- Normalizers (choose one) ---
# 1) Per-time RMS of true signal (very common)
rms_true_t = np.sqrt(np.mean(U_true_np**2, axis=1)) + 1e-12
nrmse_rms_t = rmse_t / rms_true_t

# 2) Per-time std of true signal
std_true_t = np.std(U_true_np, axis=1) + 1e-12
nrmse_std_t = rmse_t / std_true_t

# 3) Global RMS over the whole rollout (constant denominator)
rms_true_global = np.sqrt(np.mean(U_true_np**2)) + 1e-12
nrmse_rms_global = rmse_t / rms_true_global

# --- Plot: raw RMSE ---
plt.figure(figsize=(8,3))
plt.plot(t, rmse_t)
plt.xlabel("t")
plt.ylabel("RMSE over x")
plt.title("Forecast RMSE vs time")
plt.tight_layout()
plt.show()
plt.close()

# --- Plot: normalized RMSE variants ---
plt.figure(figsize=(8,3))
plt.plot(t, nrmse_rms_t, label="nRMSE(t) / RMS(true(t))")
plt.plot(t, nrmse_std_t, label="nRMSE(t) / std(true(t))")
plt.plot(t, nrmse_rms_global, label="nRMSE(t) / RMS(true) global", linestyle="--")
plt.xlabel("t")
plt.ylabel("normalized RMSE")
plt.title("Forecast normalized RMSE vs time")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

# --- Optional: print summary metrics (single numbers) ---
print("RMSE(t) mean:", rmse_t.mean())
print("nRMSE/RMS(t) mean:", nrmse_rms_t.mean())
print("nRMSE/std(t) mean:", nrmse_std_t.mean())
print("nRMSE/RMS(global) mean:", nrmse_rms_global.mean())

# --- Optional: final-time metrics ---
print("Final RMSE:", rmse_t[-1])
print("Final nRMSE/RMS(t):", nrmse_rms_t[-1])
print("Final nRMSE/std(t):", nrmse_std_t[-1])
print("Final nRMSE/RMS(global):", nrmse_rms_global[-1])


# In[ ]:


@torch.no_grad()
def rollout_rk4_substeps(u0, t_grid, dynamics_fn, n_sub=10):
    """
    RK4 with n_sub internal substeps per output step.
    """
    if u0.dim() == 1:
        u = u0[None, :].clone()
    else:
        u = u0.clone()

    Nt = t_grid.numel()
    Nx = u.shape[-1]
    out = torch.zeros((Nt, Nx), device=u.device, dtype=u.dtype)
    out[0] = u[0]

    for n in range(Nt - 1):
        dt_out = (t_grid[n+1] - t_grid[n]).item()
        dt = dt_out / n_sub

        for _ in range(n_sub):
            k1 = dynamics_fn(u)
            k2 = dynamics_fn(u + 0.5 * dt * k1)
            k3 = dynamics_fn(u + 0.5 * dt * k2)
            k4 = dynamics_fn(u + dt * k3)
            u = u + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

            # stop early if NaNs/Infs appear
            if not torch.isfinite(u).all():
                return out, n

        out[n+1] = u[0]

    return out, None

kan_pde.eval()
t_grid = torch.tensor(t, dtype=torch.float32, device=device)
u0_roll = torch.tensor(U_true[0], dtype=torch.float32, device=device)

U_pred, broke_at = rollout_rk4_substeps(u0_roll, t_grid, dynamics_fn, n_sub=20)
print("broke_at:", broke_at)


# In[ ]:


U_pred_np = U_pred.detach().cpu().numpy()
U_true_np = np.array(U_true, copy=False)

err = U_pred_np - U_true_np
err_masked = np.ma.masked_invalid(err)

plt.figure(figsize=(10,3))
plt.contourf(t, x, err_masked.T, levels=201, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x"); plt.title("Rollout error (masked NaNs/Infs)")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


err = U_pred_np - U_true_np
rmse_t = np.sqrt(np.mean(err**2, axis=1))

rms_true_t = np.sqrt(np.mean(U_true_np**2, axis=1)) + 1e-12
nrmse_t = rmse_t / rms_true_t

plt.figure(figsize=(8,3))
plt.plot(t, nrmse_t)
plt.xlabel("t")
plt.ylabel("nRMSE = RMSE / RMS(true)")
plt.title("Normalized forecast error vs time")
plt.yscale("log")  # optional but very helpful
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


rel = err / (np.abs(U_true_np) + 1e-12)
v = np.nanpercentile(np.abs(rel), 99)
plt.figure(figsize=(10,3))
plt.contourf(t, x, rel.T, levels=201, vmin=-v, vmax=v, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x")
plt.title("Relative error (û-u)/(|u|+eps), clipped at 99th pct")
plt.colorbar()
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


threshold = 0.5
idx = np.argmax(nrmse_t > threshold) if np.any(nrmse_t > threshold) else None
vpt = t[idx] if idx is not None else t[-1]

print(f"VPT @ nRMSE>{threshold}: {vpt:.3f}")

plt.figure(figsize=(8,3))
plt.plot(t, nrmse_t)
plt.axhline(threshold, linestyle="--")
if idx is not None:
    plt.axvline(vpt, linestyle="--")
plt.xlabel("t"); plt.ylabel("nRMSE")
plt.title(f"nRMSE and VPT (threshold={threshold})")
plt.yscale("log")
plt.tight_layout()
plt.show()
plt.close()


# In[ ]:


plt.semilogy(t, nrmse_t)
plt.xlabel("t")
plt.ylabel("nRMSE")
plt.title("KS forecast error growth")
plt.tight_layout()
plt.show()


# In[ ]:


plt.figure(figsize=(8,3))
plt.plot(t, np.max(np.abs(U_true_np), axis=1), label="true")
plt.plot(t, np.max(np.abs(U_pred_np), axis=1), "--", label="pred")
plt.xlabel("t")
plt.ylabel("max |u|")
plt.legend()
plt.title("Amplitude stability check")
plt.tight_layout()
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def plot_ks_rollout_diagnostics(t, x, U_true_np, U_pred_np, clip_pct=99, eps=1e-12):
    """
    t: (Nt,)
    x: (Nx,)
    U_true_np, U_pred_np: (Nt, Nx)
    """

    assert U_true_np.shape == U_pred_np.shape
    Nt, Nx = U_true_np.shape

    err = U_pred_np - U_true_np

    # --- normalization for forecast metrics ---
    rmse_t = np.sqrt(np.mean(err**2, axis=1))                        # (Nt,)
    rms_true_t = np.sqrt(np.mean(U_true_np**2, axis=1)) + eps        # (Nt,)
    nrmse_t = rmse_t / rms_true_t

    # --- robust clipping for spacetime error (avoid "blank" plots) ---
    v_err = np.nanpercentile(np.abs(err), clip_pct) + eps

    # --- shared color scale for u(t,x) plots (so you can visually compare) ---
    v_u = np.nanpercentile(np.abs(U_true_np), 99.5) + eps

    # --- relative error field (useful for attractor drift vs phase error) ---
    rel = err / (np.abs(U_true_np) + eps)
    v_rel = np.nanpercentile(np.abs(rel), clip_pct) + eps

    # --- amplitude and energy-like summaries ---
    max_true = np.max(np.abs(U_true_np), axis=1)
    max_pred = np.max(np.abs(U_pred_np), axis=1)
    rms_true = np.sqrt(np.mean(U_true_np**2, axis=1))
    rms_pred = np.sqrt(np.mean(U_pred_np**2, axis=1))

    # -----------------------
    # Figure 1: True vs Pred spacetime with matched color limits
    # -----------------------
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.contourf(t, x, U_true_np.T, levels=201, vmin=-v_u, vmax=v_u, cmap="turbo")
    ax1.set_title("Ground truth $u(t,x)$ (matched scale)")
    ax1.set_xlabel("t"); ax1.set_ylabel("x")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.contourf(t, x, U_pred_np.T, levels=201, vmin=-v_u, vmax=v_u, cmap="turbo")
    ax2.set_title("Rollout prediction $\hat{u}(t,x)$ (matched scale)")
    ax2.set_xlabel("t"); ax2.set_ylabel("x")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(2, 2, 3)
    im3 = ax3.contourf(t, x, err.T, levels=201, vmin=-v_err, vmax=v_err, cmap="turbo")
    ax3.set_title(f"Error field $(\\hat{{u}}-u)$ (clipped ±{clip_pct}th pct)")
    ax3.set_xlabel("t"); ax3.set_ylabel("x")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t, nrmse_t, label="nRMSE = RMSE / RMS(true)")
    ax4.set_yscale("log")
    ax4.set_title("Normalized forecast error vs time (log scale)")
    ax4.set_xlabel("t"); ax4.set_ylabel("nRMSE")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # -----------------------
    # Figure 2: Relative error spacetime + amplitude/energy summaries
    # -----------------------
    fig = plt.figure(figsize=(12, 6))

    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.contourf(t, x, rel.T, levels=201, vmin=-v_rel, vmax=v_rel, cmap="turbo")
    ax1.set_title(f"Relative error $(\\hat{{u}}-u)/( |u|+\\epsilon)$ (clipped ±{clip_pct}th pct)")
    ax1.set_xlabel("t"); ax1.set_ylabel("x")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(t, max_true, label="max|u| true")
    ax2.plot(t, max_pred, "--", label="max|u| pred")
    ax2.set_title("Amplitude stability (max |u|)")
    ax2.set_xlabel("t"); ax2.set_ylabel("max|u|")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(t, rms_true, label="RMS(u) true")
    ax3.plot(t, rms_pred, "--", label="RMS(u) pred")
    ax3.set_title("Energy proxy (RMS)")
    ax3.set_xlabel("t"); ax3.set_ylabel("RMS(u)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(t, rmse_t, label="RMSE")
    ax4.set_yscale("log")
    ax4.set_title("RMSE vs time (log scale)")
    ax4.set_xlabel("t"); ax4.set_ylabel("RMSE")
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # Return arrays in case you want to compute VPT etc.
    return rmse_t, nrmse_t

# Usage:
rmse_t, nrmse_t = plot_ks_rollout_diagnostics(t, x, U_true_np, U_pred_np, clip_pct=99)


# In[ ]:


import numpy as np

def stlsq(Theta, y, lam=0.05, max_iter=20, ridge=1e-8, normalize=True):
    """
    Sequential Thresholded Least Squares (STLSQ) for sparse regression.

    Theta: (N, F) library
    y:     (N, 1) target
    lam: threshold for sparsity
    ridge: ridge term for stability
    normalize: column-normalize Theta internally (recommended)

    Returns:
      xi: (F, 1) coefficients in ORIGINAL (unnormalized) Theta basis
      mask: (F,) active terms
    """
    Theta = np.asarray(Theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

    # Column scaling
    if normalize:
        col_norms = np.linalg.norm(Theta, axis=0) + 1e-12
        Theta_n = Theta / col_norms
    else:
        col_norms = np.ones(Theta.shape[1])
        Theta_n = Theta

    F = Theta_n.shape[1]
    I = np.eye(F)

    # initial ridge solve
    xi = np.linalg.solve(Theta_n.T @ Theta_n + ridge * I, Theta_n.T @ y)

    for _ in range(max_iter):
        small = (np.abs(xi[:, 0]) < lam)
        xi[small] = 0.0
        big = ~small
        if big.sum() == 0:
            break

        # refit only active terms
        Theta_b = Theta_n[:, big]
        Fb = Theta_b.shape[1]
        xi_b = np.linalg.solve(Theta_b.T @ Theta_b + ridge * np.eye(Fb), Theta_b.T @ y)

        xi[:] = 0.0
        xi[big, 0] = xi_b[:, 0]

    # unscale back to original Theta basis
    xi_unscaled = xi / col_norms.reshape(-1, 1)

    return xi_unscaled, (~(np.abs(xi_unscaled[:, 0]) < 1e-15))

def print_equation(feat_names, xi, label="u_t"):
    terms = []
    for name, c in zip(feat_names, xi[:, 0]):
        if abs(c) > 0:
            terms.append(f"({c:+.6f})*{name}")
    rhs = " ".join(terms) if terms else "0"
    print(f"{label} = {rhs}")

lam = 0.1  # try 0.005, 0.01, 0.02, 0.05, 0.1
xi, active = stlsq(Theta_np, y_np, lam=lam, max_iter=30, ridge=1e-10, normalize=True)

print("Active terms:", [feat_names[i] for i in np.where(active)[0]])
print_equation(feat_names, xi, label="u_t")


# In[1]:


X_std


# In[ ]:




