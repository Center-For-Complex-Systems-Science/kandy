#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pykan')


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

def _fit(
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
from kan.LBFGS import LBFGS


def fit(
    model, dataset, opt="LBFGS", steps=100, log=1,
    lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0.,
    update_grid=True, grid_update_num=10, loss_fn=None, lr=1.,
    start_grid_update_step=-1, stop_grid_update_step=50, batch=-1, metrics=None,
    in_vars=None, out_vars=None, beta=3, singularity_avoiding=False, y_th=1000.,
    reg_metric='edge_forward_spline_n', display_metrics=None, k=1,

    # ---- Integrated rollout loss -----------------------------------------------
    rollout_weight=0.0,
    rollout_horizon=None,
    traj_batch=-1,
    dynamics_fn=None,
    integrator="rk4",

    # ---- NEW: Rusanov-specific parameters --------------------------------------
    dx=None,                     # spatial grid spacing (REQUIRED for rusanov)
    flux_physical=None,          # physical flux f(u), default: 0.5*u^2 (Burgers)
    max_wavespeed_fn=None,       # a(uL,uR) for Rusanov dissipation, default: max(|uL|,|uR|)
    nu_art=0.0,                  # additional explicit artificial viscosity coefficient
    kan_correction_weight=0.0,   # how much of KAN's output to blend into Rusanov RHS
                                 # 0.0 = pure Rusanov rollout (KAN only supervised on u_t)
                                 # 1.0 = full KAN correction added to Rusanov base
):
    """
    Extended KAN fit with optional Rusanov-based integrator for shock-robust rollout.

    The Rusanov integrator performs conservative updates:
        u^{n+1} = u^n - dt/dx * (F_{i+1/2} - F_{i-1/2})  [+ optional KAN correction + art. viscosity]

    where F_{i+1/2} = 0.5*(f(uL) + f(uR)) - 0.5*alpha*(uR - uL)  is the Rusanov numerical flux.

    Three rollout modes:
    1. integrator="euler"/"rk4"  — original behavior, KAN predicts full u_t (blows up at shocks)
    2. integrator="rusanov"      — Rusanov flux handles the physics, KAN correction is optional
    3. integrator="rusanov_kan"  — Rusanov provides dissipation backbone, KAN correction is blended in

    The derivative supervision loss (train_input -> train_label) is ALWAYS active and trains
    the KAN on pointwise u_t. The Rusanov integrator is only used for the rollout loss term.
    """

    assert k >= 1, "k must be >= 1"

    # -------------------------------------------------------------------------
    # k-step forward for derivative supervision (unchanged)
    # -------------------------------------------------------------------------
    def k_step_forward(inputs):
        state = inputs
        for _ in range(k):
            state = model.forward(state, singularity_avoiding=singularity_avoiding, y_th=y_th)
        return state

    # -------------------------------------------------------------------------
    # Dynamics function for ODE-based rollout (euler/rk4)
    # -------------------------------------------------------------------------
    if dynamics_fn is None:
        def dynamics_fn(state):
            return model.forward(state, singularity_avoiding=singularity_avoiding, y_th=y_th)

    # -------------------------------------------------------------------------
    # Rusanov flux machinery
    # -------------------------------------------------------------------------
    if flux_physical is None:
        def flux_physical(u):
            return 0.5 * u ** 2  # Burgers flux

    if max_wavespeed_fn is None:
        def max_wavespeed_fn(uL, uR):
            return torch.maximum(torch.abs(uL), torch.abs(uR))

    def rusanov_rhs(u):
        """
        Conservative Rusanov RHS: du/dt = -1/dx * (F_{i+1/2} - F_{i-1/2}) [+ viscosity]
        u: (B, Nx)   — batch of spatial fields
        returns: (B, Nx)
        """
        assert dx is not None, "dx must be provided for Rusanov integrator"

        uL = u
        uR = torch.roll(u, shifts=-1, dims=1)  # periodic BC

        fL = flux_physical(uL)
        fR = flux_physical(uR)

        alpha = max_wavespeed_fn(uL, uR)

        # Rusanov numerical flux at i+1/2
        F_iphalf = 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
        F_imhalf = torch.roll(F_iphalf, shifts=1, dims=1)

        rhs = -(F_iphalf - F_imhalf) / dx

        # Optional: explicit artificial viscosity (Laplacian dissipation)
        if nu_art > 0.0:
            u_xx = (torch.roll(u, -1, dims=1) - 2.0 * u + torch.roll(u, 1, dims=1)) / (dx ** 2)
            rhs = rhs + nu_art * u_xx

        return rhs

    def rusanov_kan_rhs(u):
        """
        Rusanov base + blended KAN correction.
        The KAN correction is scaled by kan_correction_weight.
        """
        base_rhs = rusanov_rhs(u)
        kan_rhs = dynamics_fn(u)  # KAN predicts full u_t
        return base_rhs + kan_correction_weight * (kan_rhs - base_rhs)
        # When weight=0 -> pure Rusanov
        # When weight=1 -> pure KAN (but Rusanov dissipation still present implicitly
        #                   because base_rhs cancels out)
        # Intermediate values blend: robust backbone + learned correction

    # -------------------------------------------------------------------------
    # Integrator dispatch
    # -------------------------------------------------------------------------
    def euler_step(s, dt):
        return s + dt * dynamics_fn(s)

    def rk4_step(s, dt):
        k1 = dynamics_fn(s)
        k2 = dynamics_fn(s + 0.5 * dt * k1)
        k3 = dynamics_fn(s + 0.5 * dt * k2)
        k4 = dynamics_fn(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def rusanov_euler_step(s, dt):
        """Simple forward Euler with Rusanov RHS — stable for CFL < 1."""
        return s + dt * rusanov_rhs(s)

    def rusanov_kan_euler_step(s, dt):
        """Forward Euler with blended Rusanov+KAN RHS."""
        return s + dt * rusanov_kan_rhs(s)

    def ssprk3_step(s, dt, rhs_fn):
        """
        Strong Stability Preserving RK3 (Shu-Osher).
        This is TVD — it won't create new extrema, critical for shocks.
        """
        s1 = s + dt * rhs_fn(s)
        s2 = 0.75 * s + 0.25 * (s1 + dt * rhs_fn(s1))
        s3 = (1.0 / 3.0) * s + (2.0 / 3.0) * (s2 + dt * rhs_fn(s2))
        return s3

    def step_fn(s, dt):
        """Unified step function based on integrator choice."""
        integ = integrator.lower()
        if integ == "euler":
            return euler_step(s, dt)
        elif integ == "rk4":
            return rk4_step(s, dt)
        elif integ == "rusanov":
            # Use SSP-RK3 for TVD property (better than plain Euler)
            return ssprk3_step(s, dt, rusanov_rhs)
        elif integ == "rusanov_kan":
            return ssprk3_step(s, dt, rusanov_kan_rhs)
        elif integ == "rusanov_euler":
            return rusanov_euler_step(s, dt)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

    # -------------------------------------------------------------------------
    # Integrate states (updated to use unified step_fn)
    # -------------------------------------------------------------------------
    def integrate_states(s0, t):
        """
        s0: (B, state_dim)
        t:  (B, T) or (T,)
        returns: pred_traj (B, H+1, state_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(s0.shape[0], -1)
        B, T = t.shape
        state = s0
        out = [state]

        H = T - 1 if rollout_horizon is None else min(rollout_horizon, T - 1)

        for i in range(H):
            dt_i = (t[:, i + 1] - t[:, i]).unsqueeze(1)  # (B, 1)
            state = step_fn(state, dt_i)
            out.append(state)

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
    results = dict(
        train_loss=[], test_loss=[], reg=[],
        rollout_train_loss=[], rollout_test_loss=[]
    )
    if metrics is not None:
        for m in metrics:
            results[m.__name__] = []

    # -------------------------------------------------------------------------
    # Batching for derivative supervision
    # -------------------------------------------------------------------------
    n_train = dataset['train_input'].shape[0]
    n_test  = dataset['test_input'].shape[0]
    batch_size      = n_train if batch == -1 or batch > n_train else batch
    batch_size_test = n_test  if batch == -1 or batch > n_test  else batch

    # -------------------------------------------------------------------------
    # Batching for trajectories
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
    # Rollout loss helper
    # -------------------------------------------------------------------------
    def rollout_loss_on_batch(traj, t):
        s0 = traj[:, 0, :]
        pred = integrate_states(s0, t)
        true = traj[:, :pred.shape[1], :]
        return loss_fn(pred, true)

    # -------------------------------------------------------------------------
    # LBFGS closure
    # -------------------------------------------------------------------------
    def closure():
        global train_loss, reg_, rollout_train_loss
        optimizer.zero_grad()

        pred = k_step_forward(dataset['train_input'][train_id])
        train_loss = loss_fn(pred, dataset['train_label'][train_id])

        if rollout_weight > 0.0 and has_train_traj:
            traj = dataset['train_traj'][traj_id]
            tt   = dataset['train_t']
            rollout_train_loss = rollout_loss_on_batch(traj, tt)
        else:
            rollout_train_loss = torch.tensor(0.0, device=pred.device)

        if model.save_act:
            if reg_metric == "edge_backward":
                model.attribute()
            if reg_metric == "node_backward":
                model.node_attribute()
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
        # --- derivative batches
        train_id = np.arange(train_ptr, train_ptr + batch_size) % n_train
        train_ptr = (train_ptr + batch_size) % n_train

        test_id = np.arange(test_ptr, test_ptr + batch_size_test) % n_test
        test_ptr = (test_ptr + batch_size_test) % n_test

        # --- trajectory batches
        if has_train_traj and rollout_weight > 0.0:
            traj_id = np.arange(traj_ptr, traj_ptr + traj_batch_size) % n_traj_train
            traj_ptr = (traj_ptr + traj_batch_size) % n_traj_train

        if has_test_traj and rollout_weight > 0.0:
            traj_test_id = np.arange(traj_test_ptr, traj_test_ptr + traj_batch_size_test) % n_traj_test
            traj_test_ptr = (traj_test_ptr + traj_batch_size_test) % n_traj_test

        # --- grid update
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
                if reg_metric == "edge_backward":
                    model.attribute()
                if reg_metric == "node_backward":
                    model.node_attribute()
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

        if metrics is not None:
            for m in metrics:
                results[m.__name__].append(m().item())

        results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
        results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
        results['rollout_train_loss'].append(
            torch.sqrt(rollout_weight * rollout_train_loss).cpu().detach().numpy()
        )
        results['rollout_test_loss'].append(
            torch.sqrt(rollout_weight * rollout_test_loss).cpu().detach().numpy()
        )
        results['reg'].append(reg_.cpu().detach().numpy())

        if step % log == 0:
            print(
                f"step {step:5d} | "
                f"train {train_loss:.6e} | test {test_loss:.6e} | "
                f"roll_tr {rollout_weight * rollout_train_loss:.6e} | "
                f"roll_te {rollout_test_loss:.6e}"
            )

    model.log_history("fit")
    model.symbolic_enabled = old_symbolic_enabled
    return results


# In[4]:


"""
KANDy-based PDE discovery for inviscid Burgers equation with shocks.

Key changes from original to prevent blowup:
1. Feature library includes u_xx (so KAN can learn viscosity)
2. Rollout uses Rusanov flux integrator (conservative, dissipative)
3. SSP-RK3 time stepping during rollout (TVD — won't create new oscillations)
4. Explicit artificial viscosity as safety net during final evaluation rollout
5. CFL-limited time stepping
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from kan import KAN
#from fit_rusanov import fit

# -----------------------
# Reproducibility / device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
print("Device:", device)

# ============================================================
# 1) Ground truth: inviscid Burgers with Rusanov flux
# ============================================================
x_min, x_max = -np.pi, np.pi
dx = 0.02
x = np.arange(x_min, x_max + dx, dx)
Nx = len(x)

t0, t1 = 0.0, 3.0
dt_data = 0.002
t = np.linspace(t0, t1, int(round((t1 - t0) / dt_data)) + 1)

# Random Fourier initial condition
K_fourier = 20
p = 1.5
a_coeff = np.random.randn(K_fourier) * (np.arange(1, K_fourier + 1) ** (-p))
phi = 2 * np.pi * np.random.rand(K_fourier)
u0_np = sum(
    a_coeff[kk] * np.sin((kk + 1) * x + phi[kk]) for kk in range(K_fourier)
).astype(np.float64)


def flux_np(u):
    return 0.5 * u ** 2


def burgers_rhs_np(_t, u):
    uL = u
    uR = np.roll(u, -1)
    fL, fR = flux_np(uL), flux_np(uR)
    alpha = np.maximum(np.abs(uL), np.abs(uR))
    F_iphalf = 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
    F_imhalf = np.roll(F_iphalf, 1)
    return -(F_iphalf - F_imhalf) / dx


print("Generating Burgers ground truth (Rusanov + RK45)...")
sol = solve_ivp(
    burgers_rhs_np, (t0, t1), u0_np,
    t_eval=t, method="RK45", rtol=1e-7, atol=1e-9
)
if not sol.success:
    raise RuntimeError(sol.message)

U_true = sol.y.T.astype(np.float32)  # (Nt, Nx)
Nt = U_true.shape[0]
print(f"Ground truth shape: {U_true.shape}  (Nt={Nt}, Nx={Nx})")

plt.figure(figsize=(8, 3))
plt.contourf(t, x, U_true.T, levels=201, cmap="turbo")
plt.xlabel("t"); plt.ylabel("x")
plt.title("Ground truth Burgers (includes shocks)")
plt.tight_layout()
plt.savefig("figs/ground_truth.png", dpi=150)
plt.show()
plt.close()

# ============================================================
# 2) Spatial derivative operators (shock-robust)
# ============================================================

def minmod(a, b):
    return 0.5 * (torch.sign(a) + torch.sign(b)) * torch.minimum(torch.abs(a), torch.abs(b))


def tvd_ux(u, dx):
    """TVD minmod-limited first derivative, periodic BC."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    up = torch.roll(u, shifts=-1, dims=1)
    um = torch.roll(u, shifts=+1, dims=1)
    du_f = (up - u) / dx
    du_b = (u - um) / dx
    return minmod(du_b, du_f)


def laplacian(u, dx):
    """Second derivative (for viscosity term), periodic BC."""
    if u.dim() == 1:
        u = u.unsqueeze(0)
    return (torch.roll(u, -1, dims=1) - 2.0 * u + torch.roll(u, 1, dims=1)) / (dx ** 2)


# ============================================================
# 3) Training data: sparse snapshots
# ============================================================
#t_train = np.array([0.0, 0.4, 0.8, 1.0, 1.2, 1.4, 1.8, 2.2, 2.6, 3.0], dtype=np.float32)
#t_train = np.arange(0.0, T_train + dt, dt).astype(np.float32)
t_train = t.astype(np.float32)

def time_to_index(tt):
    return int(round((float(tt) - t0) / dt_data))


train_indices = [time_to_index(tt) for tt in t_train]
X_train_np = U_true[train_indices, :]  # (K, Nx)
K_snap = X_train_np.shape[0]

X_snap = torch.tensor(X_train_np, dtype=torch.float32, device=device)
t_snap = torch.tensor(t_train, dtype=torch.float32, device=device)

dt_seg = (t_snap[1:] - t_snap[:-1]).unsqueeze(1)
U_k = X_snap[:-1]
Ut_k = (X_snap[1:] - X_snap[:-1]) / dt_seg

# Compute features for each snapshot
ux_k = tvd_ux(U_k, dx)
uxx_k = laplacian(U_k, dx)

# ============================================================
# 4) Feature library: [u, u_x, u*u_x, u_xx]
#    True Burgers: u_t = -u*u_x  (+ 0*u_xx)
#    Including u_xx lets the KAN learn a small viscosity if needed.
# ============================================================

def build_library(u, ux=None):
    """
    u: (B, Nx)
    Returns Theta: (B*Nx, F) with F=4 features
    """
    if ux is None:
        ux = tvd_ux(u, dx)
    uxx = laplacian(u, dx)
    feats = [
        u,          # feature 0: u
        ux,         # feature 1: u_x
        u * ux,     # feature 2: u*u_x  (the actual Burgers nonlinearity)
        uxx,        # feature 3: u_xx   (viscosity — KAN can learn its coefficient)
    ]
    Theta = torch.stack(feats, dim=-1)  # (B, Nx, F)
    B, N, F = Theta.shape
    return Theta.reshape(B * N, F)


Theta = build_library(U_k, ux_k)
y = Ut_k.reshape(-1, 1)

# Normalize features
X_mean = Theta.mean(dim=0, keepdim=True)
X_std = Theta.std(dim=0, keepdim=True) + 1e-8
Theta_n = (Theta - X_mean) / X_std

# Train/test split
N_pts = Theta_n.shape[0]
perm = torch.randperm(N_pts, device=device)
N_test = max(1, int(0.2 * N_pts))
test_idx = perm[:N_test]
train_idx = perm[N_test:]

dataset = {
    "train_input": Theta_n[train_idx],
    "train_label": y[train_idx],
    "test_input":  Theta_n[test_idx],
    "test_label":  y[test_idx],
}

print(f"Dataset: train={dataset['train_input'].shape}, test={dataset['test_input'].shape}")
print(f"Features: F={Theta_n.shape[1]}  [u, u_x, u*u_x, u_xx]")

# ============================================================
# 5) Rollout windows
# ============================================================
rollout_horizon = 5
H = rollout_horizon

U_series = torch.tensor(U_true, dtype=torch.float32, device=device)
t_series = torch.tensor(t, dtype=torch.float32, device=device)


def sample_windows(U, T, H, n_windows, seed=0):
    g = torch.Generator(device=U.device)
    g.manual_seed(seed)
    max_start = U.shape[0] - (H + 1)
    starts = torch.randint(0, max_start + 1, (n_windows,), generator=g, device=U.device)
    trj_u = torch.stack([U[s:s + H + 1] for s in starts], dim=0)
    trj_t = torch.stack([T[s:s + H + 1] for s in starts], dim=0)
    return trj_u, trj_t


n_train_trj = 12
n_test_trj = 1
train_u_trj, train_t_trj = sample_windows(U_series, t_series, H, n_train_trj, seed=1)
test_u_trj, test_t_trj = sample_windows(U_series, t_series, H, n_test_trj, seed=2)

dataset["train_traj"] = train_u_trj
dataset["train_t"] = train_t_trj[0]  # (H+1,) shared time

# ============================================================
# 6) KAN model
# ============================================================
n_features = int(Theta_n.shape[1])
rbf = lambda x: torch.exp(-(3*x ** 2))
base_fun = lambda x: 0.01*x # Slope is the viscocity
kan_pde = KAN(width=[n_features, 1], grid=5, k=3, base_fun=rbf, seed=0).to(device)

# ============================================================
# 7) Dynamics function: u -> u_t via KAN
#    Used for BOTH derivative supervision rollout AND evaluation.
# ============================================================

def dynamics_fn(u):
    """
    u: (B, Nx) -> u_t: (B, Nx)
    Computes features, normalizes, runs KAN, reshapes.
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)
    Theta = build_library(u)
    Theta_normed = (Theta - X_mean) / X_std
    ut = kan_pde(Theta_normed)
    if ut.dim() == 2 and ut.shape[1] == 1:
        ut = ut[:, 0]
    return ut.reshape(u.shape[0], u.shape[1])


# ============================================================
# 8) Training
# ============================================================

# Phase 1: Pure derivative supervision (no rollout) to get a good initial KAN
# print("\n=== Phase 1: Derivative supervision only ===")
# _ = kan_pde.fit(
#     dataset,
#     singularity_avoiding=True,
#     steps=200,
# )

# Phase 2: Add Rusanov-based rollout loss
print("\n=== Phase 2: Derivative supervision + Rusanov rollout ===")
results = fit(
    kan_pde, dataset,
    steps=50,
    lamb=0.0,
    lr=10,
    rollout_weight=10.1,
    rollout_horizon=H, #10,
    traj_batch=1,
    singularity_avoiding=False,
    dynamics_fn=dynamics_fn,

    # --- Rusanov integrator for shock-safe rollout ---
    integrator="euler",#"rusanov",          # uses SSP-RK3 + Rusanov flux
    dx=dx,                         # spatial grid spacing
    flux_physical=lambda u: 0.5 * u ** 2,  # Burgers flux
    nu_art=0.001,                  # small artificial viscosity
    kan_correction_weight=0.5,     # start with pure Rusanov rollout
    stop_grid_update_step=250
)

# ============================================================
# 9) Evaluation rollout: Rusanov-based (won't blow up)
# ============================================================

def rusanov_step_eval(u, dt_val):
    """
    Single Rusanov + artificial viscosity step for evaluation.
    Uses SSP-RK3 for TVD property.
    """
    nu_eval = 0.5 * dx  # O(dx) viscosity, matches Rusanov's implicit dissipation

    def rhs(u_):
        uL = u_
        uR = torch.roll(u_, shifts=-1, dims=1)
        fL = 0.5 * uL ** 2
        fR = 0.5 * uR ** 2
        alpha = torch.maximum(torch.abs(uL), torch.abs(uR))
        F_iphalf = 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
        F_imhalf = torch.roll(F_iphalf, shifts=1, dims=1)
        r = -(F_iphalf - F_imhalf) / dx
        # Add artificial viscosity
        u_xx = (torch.roll(u_, -1, dims=1) - 2.0 * u_ + torch.roll(u_, 1, dims=1)) / (dx ** 2)
        r = r + nu_eval * u_xx
        return r

    # SSP-RK3
    s1 = u + dt_val * rhs(u)
    s2 = 0.75 * u + 0.25 * (s1 + dt_val * rhs(s1))
    s3 = (1.0 / 3.0) * u + (2.0 / 3.0) * (s2 + dt_val * rhs(s2))
    return s3


def kan_rusanov_step_eval(u, dt_val, blend=0.3):
    """
    Blended: Rusanov backbone + KAN learned correction.
    blend=0 -> pure Rusanov, blend=1 -> pure KAN
    """
    nu_eval = 0.5 * dx

    def rhs(u_):
        # Rusanov base
        uL = u_
        uR = torch.roll(u_, shifts=-1, dims=1)
        fL = 0.5 * uL ** 2
        fR = 0.5 * uR ** 2
        alpha = torch.maximum(torch.abs(uL), torch.abs(uR))
        F_iphalf = 0.5 * (fL + fR) - 0.5 * alpha * (uR - uL)
        F_imhalf = torch.roll(F_iphalf, shifts=1, dims=1)
        base = -(F_iphalf - F_imhalf) / dx
        u_xx = (torch.roll(u_, -1, dims=1) - 2.0 * u_ + torch.roll(u_, 1, dims=1)) / (dx ** 2)
        base = base + nu_eval * u_xx

        # KAN correction
        kan_ut = dynamics_fn(u_)

        return base + blend * (kan_ut - base)

    # SSP-RK3
    s1 = u + dt_val * rhs(u)
    s2 = 0.75 * u + 0.25 * (s1 + dt_val * rhs(s1))
    s3 = (1.0 / 3.0) * u + (2.0 / 3.0) * (s2 + dt_val * rhs(s2))
    return s3


print("\n=== Evaluation rollout ===")
with torch.no_grad():
    u_pred = torch.tensor(u0_np.astype(np.float32), device=device).unsqueeze(0)
    U_pred = [u_pred.squeeze(0).cpu().numpy()]

    for kk in range(Nt - 1):
        dt_k = float(t[kk + 1] - t[kk])

        # Option A: Pure Rusanov (guaranteed stable, but no KAN learning used in rollout)
        # u_pred = rusanov_step_eval(u_pred, dt_k)

        # Option B: Blended — Rusanov backbone + KAN correction (best of both worlds)
        u_pred = kan_rusanov_step_eval(u_pred, dt_k, blend=0.3)

        U_pred.append(u_pred.squeeze(0).cpu().numpy())

        if (kk + 1) % 200 == 0:
            max_val = np.abs(U_pred[-1]).max()
            print(f"  step {kk+1}/{Nt-1}, max|u|={max_val:.4f}")
            if max_val > 100:
                print("  WARNING: blowup detected, stopping early")
                break

U_pred = np.stack(U_pred, axis=0)
print(f"Rollout shape: {U_pred.shape}, max|u|={np.abs(U_pred).max():.4f}")

# ============================================================
# 10) Plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Ground truth
ax = axes[0]
cf = ax.contourf(t, x, U_true.T, levels=201, cmap="turbo")
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("Ground truth")

# KAN rollout
t_pred = t[:U_pred.shape[0]]
ax = axes[1]
vmin, vmax = U_true.min(), U_true.max()
cf = ax.contourf(t_pred, x, U_pred.T, levels=201, cmap="turbo",
                 vmin=vmin, vmax=vmax)
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("KAN + Rusanov rollout")

# Error
ax = axes[2]
err = np.abs(U_pred[:len(t)] - U_true[:U_pred.shape[0]])
cf = ax.contourf(t_pred, x, err.T, levels=51, cmap="hot")
ax.set_xlabel("t"); ax.set_ylabel("x")
ax.set_title("Absolute error")
plt.colorbar(cf, ax=ax)

plt.tight_layout()
plt.savefig("figs/burgers_results.png", dpi=150)
plt.show()
plt.close()
print("Saved: figs/burgers_results.png")

# Final time comparison
fig, ax = plt.subplots(figsize=(8, 3))
idx_end = min(U_pred.shape[0], Nt) - 1
ax.plot(x, U_true[idx_end], 'k-', lw=2, label="true")
ax.plot(x, U_pred[idx_end], 'r--', lw=1.5, label="predicted")
ax.set_xlabel("x"); ax.set_ylabel("u")
ax.set_title(f"Snapshot at t={t[idx_end]:.2f}")
ax.legend()
plt.tight_layout()
plt.savefig("figs/burgers_final_snapshot.png", dpi=150)
plt.show()
plt.close()
print("Saved: figs/burgers_final_snapshot.png")

# Loss curves
if results:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.semilogy(results['train_loss'], label='train (deriv)')
    ax.semilogy(results['test_loss'], label='test (deriv)')
    ax.semilogy(results['rollout_train_loss'], label='rollout train')
    ax.set_xlabel("step"); ax.set_ylabel("RMSE")
    ax.legend(); ax.set_title("Training losses")
    plt.tight_layout()
    plt.savefig("figs/burgers_losses.png", dpi=150)
    plt.show()
    plt.close()
    print("Saved: figs/burgers_losses.png")


# In[10]:


import sympy as sp
import sympy as sp
from sympy import Eq
from sympy.printing.latex import latex
from IPython.display import display, Math

model = kan_pde
model.unfix_symbolic_all()
model.save_act = True
_ = model(dataset["train_input"])
model.prune()

model_sym, rep = robust_auto_symbolic(
    model,
    # simple_lib=['x','x^2','x^3','0'],   # or pass your SYMBOLIC_LIB dict if your KAN expects dict
    # complex_lib=['x','x^2','0'],
    simple_lib=['x','x^2', '0'],   # or pass your SYMBOLIC_LIB dict if your KAN expects dict
    complex_lib=['x',"x^2",'0'],
    input_split_index=2,
    # r2_threshold=0.31,
    # weight_simple=0.2,
    r2_threshold=0.8,
    weight_simple=0.8,    
    keep="topk",
    topk_edges=64,
    max_total_complexity=10,           # optional
    verbose=4,
    inplace=True
)

print(rep)
raw = model_sym.symbolic_formula()
print("Symbolic formula extracted.")
exprs_raw, vars_ = model.symbolic_formula()

print(exprs_raw)
n_in = int(model.width_in[0])

in_vars_latex_full = [
    #r"$1$",
    r"$u$",
    r"$u_x$",
    r"$uu_x$",
    r"$u_{xx}$",
]
out_vars_latex = [r"$u_t$"]

u_sym, u_x_sym, u_xx_sym, u_xxxx_sym = sp.symbols("u u_x uu_x u_{xx}")
feature_syms_full = [
    u_sym,
    u_x_sym,
    u_xx_sym,
    u_xxxx_sym,
    # u_sym**2,
    # u_sym**3,
    # u_x_sym**2,
    # u_xx_sym**2,
    # u_sym*u_x_sym,
    # u_sym*u_xx_sym,
    # u_sym*u_xxxx_sym,
    # u_x_sym*u_xx_sym,
]

if n_in > len(in_vars_latex_full):
    raise ValueError(
        f"Model expects {n_in} inputs but only {len(in_vars_latex_full)} labels were provided. "
        "Update in_vars_latex_full/feature_syms_full to match your dataset columns."
    )

in_vars_latex = in_vars_latex_full[:n_in]
feature_syms  = feature_syms_full[:n_in]

sub_map = {vars_[i]: feature_syms[i] for i in range(n_in)}


# Substitute using vars_ (NOT free_symbols inference)
cleaned = []
for expr in exprs_raw:
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

print("\nViscous Burgers' equation:")
print("u_t + u u_x = ν u_xx")
print("="*60)


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
kan_pde.plot()


# In[ ]:





# In[20]:


kan_pde.unfix_symbolic_all()
kan_pde.save_act = True
_ = kan_pde(dataset['train_input'])
#kan_pde = kan_pde.prune(node_th=0.185, edge_th=0.185)
#kan_pde = kan_pde.prune(node_th=0.185, edge_th=0.29)
'u u_x u_xx u_xxxx'
kan_pde.plot(in_vars=[r"$u$", r"$u_{x}$", r"$uu_x$", r"$u_{xx}$"], out_vars=[r"$u_t$"])
plt.savefig(
    "figs/burgers_model.svg",
    format="svg",
    dpi=300,
    bbox_inches="tight"
)


# In[13]:


kan_pde.save_act = True
_ = kan_pde(dataset['train_input'])
#kan_pde = kan_pde.prune(node_th=0.185, edge_th=0.185)
#kan_pde = kan_pde.prune(node_th=0.185, edge_th=0.29)
"u u_x uu_xx u_{xx}"
kan_pde.plot(in_vars=[r"$u$", r"$u_{x}$", r"$uu_x$", r"$u_{xx}$"], out_vars=[r"$u_t$"])


# In[6]:


X_std


# In[15]:


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
dt_save = float(dt_data)          # data spacing
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
plt.savefig('figs/burgers_kan_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/burgers_kan_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/burgers_kan_prediction.svg', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

print("Done! Saved figs/ks_kan_prediction.png and figs/ks_kan_prediction.pdf")


# In[16]:


# ============================================================
# Premium rollout + plot for DISCOVERED Burgers model
#   u_t = -1.553*u*u_x - 0.279*u_x + 1.115*u_xx - 8.2747e-5
# ============================================================
import os
import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

os.makedirs("figs", exist_ok=True)

#𝑢𝑡=−0.279𝑢𝑥+1.115𝑢𝑥𝑥−1.553𝑢𝑢𝑥
#𝑢𝑡=−0.279𝑢𝑥+1.115𝑢𝑥𝑥−1.553𝑢𝑢𝑥
#1.159𝑢𝑥𝑥−1.564𝑢𝑢𝑥

# -----------------------------
# 1) Discovered PDE RHS (shock-robust derivatives)
# -----------------------------
a_adv = -1.564#-1.55331333414051     # coefficient on u*u_x
a_ux  = 0# -0.279082183138712    # coefficient on u_x
a_uxx =  0.0064 # Denormalizaed 0.159#1.159#1.11511753986415     # coefficient on u_xx
a0    = 0#-8.27470212101389e-5  # constant bias
# a_adv = -1.0     # coefficient on u*u_x
# a_ux  = 0    # coefficient on u_x
# a_uxx = 0     # coefficient on u_xx
# a0    = 0#-8.27470212101389e-5  # constant bias

@torch.no_grad()
def burgers_discovered_rhs(u, dx, nu_safety=0.0):
    """
    u: (B, Nx) torch tensor
    returns: (B, Nx)
    Uses TVD u_x and standard Laplacian u_xx.
    nu_safety: optional extra viscosity (adds nu_safety * u_xx)
    """
    if u.dim() == 1:
        u = u.unsqueeze(0)

    ux  = tvd_ux(u, dx)        # shock-robust first derivative
    uxx = laplacian(u, dx)     # second derivative

    return a_adv * (u * ux) + a_ux * ux + a_uxx * uxx + a0 + nu_safety * uxx


# -----------------------------
# 2) SSP-RK3 rollout with substeps (TVD-friendly)
# -----------------------------
@torch.no_grad()
def rollout_ssp_rk3_substeps(u0, t_grid, rhs_fn, dx, n_sub=1, nu_safety=0.0, stop_on_nan=True):
    """
    u0: (Nx,) or (1,Nx)
    t_grid: (Nt,) torch tensor
    rhs_fn: function(u, dx, nu_safety)->u_t
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
            # SSP-RK3
            k1 = rhs_fn(u, dx, nu_safety=nu_safety)
            u1 = u + h * k1

            k2 = rhs_fn(u1, dx, nu_safety=nu_safety)
            u2 = 0.75 * u + 0.25 * (u1 + h * k2)

            k3 = rhs_fn(u2, dx, nu_safety=nu_safety)
            u  = (1.0/3.0) * u + (2.0/3.0) * (u2 + h * k3)

            if stop_on_nan and (torch.isnan(u).any() or torch.isinf(u).any()):
                print(f"NaN/Inf encountered at outer step n={n}, substep h={h:.3e}, t={t_grid[n].item():.6f}")
                return out[:n+1]

        out[n+1] = u[0]

        if (n+1) % 500 == 0:
            s = u[0]
            print(f"[{n+1:6d}/{Nt-1}] u: mean={s.mean().item():+.3e}, std={s.std().item():+.3e}, "
                  f"min={s.min().item():+.3e}, max={s.max().item():+.3e}")

    return out

@torch.no_grad()
def rollout_ssp_rk3_cfl(u0, t_grid, rhs_fn, dx,
                        cfl=0.35, cfl_diff=0.25,
                        nu_safety=0.0,
                        max_substeps=5000,
                        stop_on_nan=True):
    """
    CFL-adaptive SSP-RK3 rollout.
    - Hyperbolic CFL: h <= cfl * dx / max_speed
      where max_speed ~ |a_adv|*max|u| + |a_ux| (computed inside rhs wrapper)
    - Diffusive CFL:  h <= cfl_diff * dx^2 / (|a_uxx| + nu_safety)
    """

    if u0.dim() == 1:
        u = u0[None, :].clone()
    else:
        u = u0.clone()

    Nt = t_grid.numel()
    Nx = u.shape[-1]
    out = torch.zeros((Nt, Nx), device=u.device, dtype=u.dtype)
    out[0] = u[0]

    # pull coefficients if they exist globally (as in the discovered block)
    # fall back to safe defaults if not found
    a_adv_local = float(globals().get("a_adv", -1.0))
    a_ux_local  = float(globals().get("a_ux",  0.0))
    a_uxx_local = float(globals().get("a_uxx", 0.0))

    for n in range(Nt - 1):
        dt_out = float((t_grid[n+1] - t_grid[n]).item())

        # estimate max wave speed for CFL
        umax = float(torch.max(torch.abs(u)).item()) + 1e-12
        max_speed = abs(a_adv_local) * umax + abs(a_ux_local)  # conservative
        h_hyp = cfl * dx / max_speed

        # diffusion stability bound (explicit)
        diff_coeff = abs(a_uxx_local) + float(nu_safety)
        if diff_coeff > 0:
            h_diff = cfl_diff * (dx * dx) / diff_coeff
        else:
            h_diff = float("inf")

        h_max = min(h_hyp, h_diff)
        n_sub = int(max(1, math.ceil(dt_out / h_max)))
        if n_sub > max_substeps:
            print(f"Step {n}: required n_sub={n_sub} > max_substeps={max_substeps}. Stopping.")
            return out[:n+1]

        h = dt_out / n_sub

        for _ in range(n_sub):
            # SSP-RK3
            k1 = rhs_fn(u, dx, nu_safety=nu_safety)
            u1 = u + h * k1

            k2 = rhs_fn(u1, dx, nu_safety=nu_safety)
            u2 = 0.75 * u + 0.25 * (u1 + h * k2)

            k3 = rhs_fn(u2, dx, nu_safety=nu_safety)
            u  = (1.0/3.0) * u + (2.0/3.0) * (u2 + h * k3)

            if stop_on_nan and (torch.isnan(u).any() or torch.isinf(u).any()):
                print(f"NaN/Inf encountered at outer step n={n}, substep h={h:.3e}, t={t_grid[n].item():.6f}")
                return out[:n+1]

        out[n+1] = u[0]

        if (n+1) % 500 == 0:
            s = u[0]
            print(f"[{n+1:6d}/{Nt-1}] n_sub={n_sub:4d}  max|u|={torch.max(torch.abs(s)).item():.3e}")

    return out



# -----------------------------
# 3) Run discovered-model rollout on your same grid/time
# -----------------------------
# Assumes you already have: x, t, dx, dt_data, U_true, device
t_grid = torch.tensor(t, dtype=torch.float32, device=device)             # (Nt,)
u0_sim = torch.tensor(U_true[0], dtype=torch.float32, device=device)     # (Nx,)

# Settings:
# - n_sub: increase if you see instability (start with 1–5 for dt_data=0.002)
# - nu_safety: small extra viscosity as a stability safety net (0 to start)
n_sub = 1
nu_safety = 0.0

# U_sim = rollout_ssp_rk3_cfl(
#     u0_sim, t_grid,
#     rhs_fn=burgers_discovered_rhs,
#     dx=dx,
#     n_sub=n_sub,
#     nu_safety=nu_safety,
# )

U_sim = rollout_ssp_rk3_cfl(
    u0_sim, t_grid,
    rhs_fn=burgers_discovered_rhs,
    dx=dx,
    cfl=0.25,        # try 0.25 if still unstable
    cfl_diff=0.25,
    nu_safety= 5e-4#0.0    # if still NaNs, try 1e-4 to 5e-4
)

U_sim_np  = U_sim.detach().cpu().numpy()
U_true_np = np.asarray(U_true)

# If rollout stopped early due to NaNs, adjust t used for plotting
t_plot = t[:U_sim_np.shape[0]]

# -----------------------------
# 4) Premium plot (KS style)
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
cmap_premium = LinearSegmentedColormap.from_list('burgers_premium', colors_custom, N=512)

# Fair scaling: match truth scaling (percentile robust)
vmin, vmax = np.percentile(U_true_np, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(
    U_sim_np.T,
    cmap=cmap_premium,
    norm=norm,
    blend_mode='soft',
    vert_exag=0.10,
    dx=dx,
    dy=float(t[1] - t[0]),
    fraction=1.2
)

T_grid, X_grid = np.meshgrid(t_plot, x)

# U_sim_np  = U_sim.detach().cpu().numpy()
# U_true_np = np.asarray(U_true)

# # if rollout stopped early
# t_plot = np.asarray(t[:U_sim_np.shape[0]], dtype=float)

# # Build meshgrid that MATCHES U_sim_np.T shape (Nx, Nt_plot)
# T_grid, X_grid = np.meshgrid(t_plot, x, indexing="xy")

# # If there are NaNs, don't contour them
# Z = U_sim_np.T  # (Nx, Nt_plot)
# finite_mask = np.isfinite(Z)
# do_contour = (Z.shape[1] >= 3) and finite_mask.all()   # require >=3 frames and no NaNs

t_plot_1d = np.asarray(t_plot).reshape(-1)   # (Nt_plot,)
x_1d      = np.asarray(x).reshape(-1)        # (Nx,)

T_grid, X_grid = np.meshgrid(t_plot_1d, x_1d, indexing="xy")  # both (Nx, Nt_plot)
Z = U_sim_np.T  # (Nx, Nt_plot)


fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(
    rgb,
    extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    interpolation='bilinear'
)
# ax.contour(
#     T_grid, X_grid, U_sim_np.T,
#     levels=np.linspace(vmin, vmax, 18),
#     colors='white',
#     linewidths=0.15,
#     alpha=0.25
# )

ax.contour(
    Z,
    levels=np.linspace(vmin, vmax, 18),
    colors='white',
    linewidths=0.15,
    alpha=0.25,
    extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
    origin='lower'
)

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)

ax.set_title(
    rf'Discovered Model Rollout: $u_t={a_adv:+.3f}\,u\,u_x{a_ux:+.3f}\,u_x{a_uxx:+.3f}\,u_{{xx}}{a0:+.1e}$',
    fontsize=12, pad=12, fontweight='medium'
)

L = float(x.max() - x.min())
ax.text(
    0.98, 0.96,
    rf'$L\approx{L:.3f},\;\; N_x={len(x)},\;\; n_{{sub}}={n_sub},\;\; \nu_{{safety}}={nu_safety:.1e}$',
    transform=ax.transAxes,
    fontsize=8, color='white', ha='right', va='top', alpha=0.7,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none')
)

sm = cm.ScalarMappable(cmap=cmap_premium, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('figs/burgers_discovered_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/burgers_discovered_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

print("Done! Saved figs/burgers_discovered_prediction.png/pdf")


# In[18]:


# ===========================
# Premium error-field plots (Burgers)
#   (red-white-blue, white=0)
# ===========================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

os.makedirs("figs", exist_ok=True)

# ---------- style (match your premium plots) ----------
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
    ["#223b8f", "#f7f7f7", "#b2182b"],
    N=512
)

def plot_error_field(U_pred, U_true, t, x, dx, dt,
                     title, out_png, out_pdf=None,
                     clip_pct=99.0, n_contours=18,
                     do_contours=True):
    """
    U_pred, U_true: (Nt, Nx)
    """
    U_pred = np.asarray(U_pred)
    U_true = np.asarray(U_true)

    Nt = min(U_pred.shape[0], U_true.shape[0], len(t))
    U_pred = U_pred[:Nt]
    U_true = U_true[:Nt]
    t_plot = np.asarray(t[:Nt]).reshape(-1)

    x_1d = np.asarray(x).reshape(-1)

    E = U_pred - U_true  # (Nt, Nx)

    # robust symmetric limits around 0
    vmax = np.nanpercentile(np.abs(E), clip_pct) + 1e-12
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # premium shaded look
    ls = LightSource(azdeg=315, altdeg=35)
    rgb = ls.shade(E.T, cmap=cmap_err, norm=norm, blend_mode='soft',
                   vert_exag=0.08, dx=float(dx), dy=float(dt), fraction=1.2)

    # Make sure meshgrid matches E.T shape (Nx, Nt)
    T_grid, X_grid = np.meshgrid(t_plot, x_1d, indexing="xy")
    Z = E.T  # (Nx, Nt)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.imshow(rgb, extent=[t_plot[0], t_plot[-1], x_1d[0], x_1d[-1]],
              aspect='auto', origin='lower', interpolation='bilinear')

    # subtle contours (symmetric about 0)
    if do_contours and (Z.shape[1] >= 3) and np.isfinite(Z).all():
        levels = np.linspace(-vmax, vmax, n_contours)
        ax.contour(T_grid, X_grid, Z, levels=levels,
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

def plot_error_field(U_pred, U_true, t, x, dx, dt,
                     title, out_png, out_pdf=None,
                     clip_pct=99.0, n_contours=18,
                     do_contours=True):
    """
    U_pred, U_true: (Nt, Nx)
    NOTE: does NOT create meshgrid (avoids x-shape bugs + kernel crashes)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import cm
    from matplotlib.colors import LightSource

    U_pred = np.asarray(U_pred)
    U_true = np.asarray(U_true)

    # Force 1D t and x for extents only
    t = np.asarray(t).reshape(-1)
    x = np.asarray(x).reshape(-1)

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
                   vert_exag=0.08, dx=float(dx), dy=float(dt), fraction=1.2)

    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
    ax.imshow(
        rgb,
        extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
        aspect='auto',
        origin='lower',
        interpolation='bilinear'
    )

    # Contours WITHOUT meshgrid
    if do_contours and np.isfinite(E).all() and (Nt >= 3):
        levels = np.linspace(-vmax, vmax, n_contours)
        ax.contour(
            E.T, levels=levels,
            colors='white', linewidths=0.12, alpha=0.20,
            extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
            origin='lower'
        )

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

# ============================================================
# Burgers error fields
#   Requires:
#     - U_true (Nt, Nx)
#     - t, x, dx
#     - dt_data (or dt = t[1]-t[0])
#     - U_sim_np from your discovered rollout cell (Nt_plot, Nx)
# Optional:
#     - U_pred from your KAN+Rusanov rollout cell
# ============================================================
dt = float(t[1] - t[0])  # dt_data

# --- Discovered PDE rollout error ---
E_disc, vmax_disc = plot_error_field(
    U_pred=U_sim_np,
    U_true=U_true,
    t=t,
    x=x,
    dx=dx,
    dt=dt,
    title=r'Discovered PDE Error Field: $e(x,t)=u_{\mathrm{disc}}-u_{\mathrm{true}}$',
    out_png='figs/burgers_discovered_error.png',
    out_pdf='figs/burgers_discovered_error.pdf',
    clip_pct=99.0
)

print("Saved:")
print("  figs/burgers_discovered_error.png/.pdf")

# --- If you also have a KAN rollout array (e.g. U_pred or U_kan), plot it too ---
# Uncomment and set the variable name you used for your KAN prediction:
E_kan, vmax_kan = plot_error_field(
    U_pred=U_pred,   # <-- change to your KAN rollout array name
    U_true=U_true,
    t=t,
    x=x,
    dx=dx,
    dt=dt,
    title=r'KAN Rollout Error Field: $e(x,t)=u_{\mathrm{KAN}}-u_{\mathrm{true}}$',
    out_png='figs/burgers_kan_error.png',
    out_pdf='figs/burgers_kan_error.pdf',
    clip_pct=99.0
)
# print("  figs/burgers_kan_error.png/.pdf")


# In[9]:


# ============================================================
# END-TO-END CELL: Premium rollout + plot for your DISCOVERED Burgers PDE
#   u_t = a_adv*u*u_x + a_ux*u_x + a_uxx*u_xx + a0
#
# Shock-safe rollout:
#   write advection in conservative form + Rusanov flux (entropy-stable)
#   diffuse term kept as explicit Laplacian
#   time stepping: SSP-RK3
# ============================================================

import os, math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, LightSource

os.makedirs("figs", exist_ok=True)

# -----------------------------
# 0) Assumes you already have:
#   x (numpy, shape Nx), t (numpy, shape Nt), dx (float),
#   U_true (numpy, shape Nt x Nx), device (torch.device)
# -----------------------------
assert "x" in globals() and "t" in globals() and "dx" in globals() and "U_true" in globals() and "device" in globals()

# -----------------------------
# 1) Your discovered coefficients (REPORT THESE AS-IS)
# -----------------------------
a_adv = -1.55331333414051     # coefficient on u*u_x
a_ux  = -0.279082183138712    # coefficient on u_x
a_uxx =  1.11511753986415     # coefficient on u_xx
a0    =  0.0                  # (you set it to 0)

# -----------------------------
# 2) Shock-safe step: conservative flux + Rusanov + diffusion (SSP-RK3)
# -----------------------------
@torch.no_grad()
def step_discovered_burgers_conservative(u, dt, dx, a_adv, a_ux, a_uxx, a0=0.0):
    """
    Solves u_t = a_adv*u*u_x + a_ux*u_x + a_uxx*u_xx + a0

    Conservative rewrite for advection:
      u_t + d/dx( -a_adv/2*u^2 - a_ux*u ) = a_uxx*u_xx + a0

    Rusanov for the flux derivative, Laplacian for diffusion.
    SSP-RK3 time stepping (TVD-ish for flux part).
    """
    if u.dim() == 1:
        u = u[None, :]

    # flux F(u) = c2*u^2 + c1*u so that -F_x = a_adv*u*u_x + a_ux*u_x
    c2 = -a_adv / 2.0
    c1 = -a_ux

    def flux(u_):
        return c2 * u_**2 + c1 * u_

    def rhs(u_):
        # Rusanov flux
        uL = u_
        uR = torch.roll(u_, -1, dims=1)
        fL = flux(uL)
        fR = flux(uR)

        # wavespeed bound alpha >= |F'(u)| = |2*c2*u + c1|
        alpha = torch.maximum(torch.abs(2*c2*uL + c1), torch.abs(2*c2*uR + c1))

        F_ip = 0.5*(fL + fR) - 0.5*alpha*(uR - uL)
        F_im = torch.roll(F_ip, 1, dims=1)
        adv = -(F_ip - F_im) / dx

        # diffusion (explicit Laplacian)
        uxx = (torch.roll(u_, -1, dims=1) - 2*u_ + torch.roll(u_, 1, dims=1)) / (dx**2)
        diff = a_uxx * uxx

        return adv + diff + a0

    # SSP-RK3
    s1 = u + dt * rhs(u)
    s2 = 0.75*u + 0.25*(s1 + dt*rhs(s1))
    s3 = (1.0/3.0)*u + (2.0/3.0)*(s2 + dt*rhs(s2))
    return s3

# -----------------------------
# 3) Rollout on same grid/time as your data
# -----------------------------
t_grid = torch.tensor(t, dtype=torch.float32, device=device)
u0_sim = torch.tensor(U_true[0], dtype=torch.float32, device=device)  # (Nx,)

with torch.no_grad():
    u = u0_sim[None, :].clone()
    U_sim_list = [u[0].detach().cpu().numpy()]

    for k in range(len(t) - 1):
        dt_k = float(t[k+1] - t[k])
        u = step_discovered_burgers_conservative(u, dt_k, dx, a_adv, a_ux, a_uxx, a0)

        if torch.isnan(u).any() or torch.isinf(u).any():
            print(f"NaN/Inf encountered at step {k}, t={t[k]:.6f}. Stopping early.")
            break

        U_sim_list.append(u[0].detach().cpu().numpy())

U_sim_np  = np.stack(U_sim_list, axis=0)              # (Nt_plot, Nx)
U_true_np = np.asarray(U_true)                        # (Nt, Nx)
t_plot    = np.asarray(t[:U_sim_np.shape[0]], float)  # (Nt_plot,)

print(f"Rollout done: U_sim_np shape={U_sim_np.shape}, max|u|={np.max(np.abs(U_sim_np)):.4f}")

# -----------------------------
# 4) Premium plot (KS style) + safe contouring
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
cmap_premium = LinearSegmentedColormap.from_list('burgers_premium', colors_custom, N=512)

# Fair scaling based on truth
vmin, vmax = np.percentile(U_true_np, [1, 99])
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Shaded relief
ls = LightSource(azdeg=315, altdeg=35)
rgb = ls.shade(
    U_sim_np.T,  # (Nx, Nt_plot)
    cmap=cmap_premium,
    norm=norm,
    blend_mode='soft',
    vert_exag=0.10,
    dx=float(dx),
    dy=float(t[1] - t[0]) if len(t) > 1 else 1.0,
    fraction=1.2
)

# Meshgrid matching Z shape
T_grid, X_grid = np.meshgrid(t_plot, x, indexing="xy")
Z = U_sim_np.T  # (Nx, Nt_plot)

fig, ax = plt.subplots(figsize=(10, 4.5), dpi=300)
ax.imshow(
    rgb,
    extent=[t_plot[0], t_plot[-1], x[0], x[-1]],
    aspect='auto',
    origin='lower',
    interpolation='bilinear'
)

# Safe contouring: only if enough frames and finite
if (Z.shape[1] >= 3) and np.isfinite(Z).all():
    ax.contour(
        T_grid, X_grid, Z,
        levels=np.linspace(vmin, vmax, 18),
        colors='white', linewidths=0.15, alpha=0.25
    )

ax.set_xlabel(r'$t$', fontsize=13, labelpad=6)
ax.set_ylabel(r'$x$', fontsize=13, labelpad=6)
ax.tick_params(labelsize=10)

ax.set_title(
    rf'Discovered Model Rollout: $u_t={a_adv:+.3f}\,u\,u_x{a_ux:+.3f}\,u_x{a_uxx:+.3f}\,u_{{xx}}{a0:+.1e}$',
    fontsize=12, pad=12, fontweight='medium'
)

L = float(np.max(x) - np.min(x))
ax.text(
    0.98, 0.96,
    rf'$L\approx{L:.3f},\;\; N_x={len(x)},\;\; \Delta x={dx:.3g}$',
    transform=ax.transAxes,
    fontsize=8, color='white', ha='right', va='top', alpha=0.7,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3, edgecolor='none')
)

sm = cm.ScalarMappable(cmap=cmap_premium, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
cbar.ax.tick_params(labelsize=9, width=0.5)
cbar.set_label(r'$u(x,t)$', fontsize=12, labelpad=8)
cbar.outline.set_linewidth(0.5)

plt.tight_layout()
plt.savefig('figs/burgers_discovered_prediction.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('figs/burgers_discovered_prediction.pdf', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
plt.close()

print("Done! Saved figs/burgers_discovered_prediction.png/pdf")


# In[13]:


kan_pde.unfix_symbolic_all()
kan_pde.save_act = True
kan_pde(dataset['train_input'])
x, y = get_edge_activation(kan_pde, 0, 3, 0, dataset['train_input'])

x = np.asarray(x).ravel()
y = np.asarray(y).ravel()

# ----- Linear regression (degree 1) -----
m1, b1 = np.polyfit(x, y, 1)
y_lin = m1 * x + b1

ss_res_lin = np.sum((y - y_lin)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2_lin = 1 - ss_res_lin / ss_tot

# ----- Quadratic regression (degree 2) -----
a2, b2, c2 = np.polyfit(x, y, 2)
y_quad = a2 * x**2 + b2 * x + c2

ss_res_quad = np.sum((y - y_quad)**2)
r2_quad = 1 - ss_res_quad / ss_tot

# ----- Plot -----
order = np.argsort(x)

plt.figure(figsize=(7, 5), dpi=150)

plt.scatter(x, y, s=18, alpha=0.7, label="data")

# Linear fit
plt.plot(x[order], y_lin[order],
         linewidth=2.5,
         label=f"Linear: y = {m1:.3f}x + {b1:.3f}\n$R^2$ = {r2_lin:.3f}")

# Quadratic fit
plt.plot(x[order], y_quad[order],
         linewidth=2.5,
         linestyle='--',
         label=f"Quadratic: y = {a2:.3f}x² + {b2:.3f}x + {c2:.3f}\n$R^2$ = {r2_quad:.3f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Edge activation with linear and quadratic regression fits")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print("----- Linear Fit -----")
print(f"slope m = {m1:.6f}")
print(f"intercept b = {b1:.6f}")
print(f"R^2 = {r2_lin:.6f}")

print("\n----- Quadratic Fit -----")
print(f"a = {a2:.6f}")
print(f"b = {b2:.6f}")
print(f"c = {c2:.6f}")
print(f"R^2 = {r2_quad:.6f}")


# In[14]:


X_std


# In[87]:


kan_pde.save_act = True
kan_pde(dataset['train_input'])
x, y = get_edge_activation(kan_pde, 0, 1, 0, dataset['train_input'])

x = np.asarray(x).ravel()
y = np.asarray(y).ravel()

# ----- linear regression y = m x + b -----
m, b = np.polyfit(x, y, 1)
y_hat = m * x + b

# standard R^2
ss_res = np.sum((y - y_hat)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res / ss_tot

# ----- plot -----
order = np.argsort(x)
plt.figure(figsize=(7, 5), dpi=150)

plt.scatter(x, y, s=18, alpha=0.7, label="data")
plt.plot(x[order], y_hat[order],
         linewidth=2.5,
         label=f"linear fit: y = {m:.3f}x + {b:.3f}\n$R^2$ = {r2:.3f}")

plt.xlabel("x")
plt.ylabel("y")
plt.title("Edge activation with linear regression fit")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

print(f"slope m = {m:.6f}")
print(f"intercept b = {b:.6f}")
print(f"R^2 = {r2:.6f}")


# In[30]:


x, y = get_edge_activation(kan_pde, 0, 0, 0, dataset['train_input'])

x = np.asarray(x).ravel()
y = np.asarray(y).ravel()
plt.plot(x,y)
plt.show()


# In[31]:


x, y = get_edge_activation(kan_pde, 0, 1, 0, dataset['train_input'])

x = np.asarray(x).ravel()
y = np.asarray(y).ravel()
plt.plot(x,y)
plt.show()


# In[32]:


x, y = get_edge_activation(kan_pde, 0, 2, 0, dataset['train_input'])

x = np.asarray(x).ravel()
y = np.asarray(y).ravel()
plt.plot(x,y)
plt.show()


# In[33]:


x, y = get_edge_activation(kan_pde, 0, 3, 0, dataset['train_input'])

x = np.asarray(x).ravel()
y = np.asarray(y).ravel()
plt.plot(x,y)
plt.show()


# In[ ]:




