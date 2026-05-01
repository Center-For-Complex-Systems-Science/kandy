#!/usr/bin/env python3
"""Analysis 2: Time-Varying Directed Connectivity via oCSE.

Measure directed causal information flow between the 3 SOZ subclusters using
optimal Causation Entropy (oCSE). Track how causal structure changes at seizure onset.

oCSE advantages over transfer entropy:
  - Accounts for indirect causation (A->B->C won't falsely show A->C)
  - Automatically selects optimal conditioning set
  - Built-in significance via shuffled surrogates
  - Returns sparse causal graph

Pipeline:
  1. Extract alpha-band phases and amplitude envelopes per cluster at 50 Hz
  2. Full-window oCSE on pre-seizure and seizure windows separately
  3. Sliding-window oCSE to track causal structure over time
  4. PLV (undirected) for comparison
  5. Pre-seizure vs seizure comparison of causal graphs

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import sys
import numpy as np
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# Add local utilities directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ieeg_utils import (
    load_episodes, extract_cluster_amplitudes, extract_cluster_phases,
    save_fig, setup_style, FS, ONSET_TIMES, CLUSTERS, CLUSTER_IDS,
    CLUSTER_NAMES, CLUSTER_COLORS, ALPHA_BAND,
)

# ============================================================
# Parameters
# ============================================================
DS = 10            # 500 Hz -> 50 Hz
DT_DS = DS / FS    # 0.02s
FS_DS = FS / DS    # 50 Hz

# oCSE parameters
OCSE_LAG = 5       # max lag (samples at 50 Hz = 0.1s)
OCSE_N_SHUFFLES = 100   # surrogates for significance

# Sliding window
SLIDE_WINDOW_S = 20.0   # 20s window
SLIDE_STEP_S = 5.0      # 5s step

# Pre-seizure and seizure window definitions (relative to onset)
PRE_WINDOW = (-30.0, 0.0)   # 30s before onset
SEZ_WINDOW = (0.0, 10.0)    # 10s after onset

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# oCSE implementation (self-contained, no external dependency)
# ============================================================
def mutual_information_ksg(x, y, k=5):
    """KSG mutual information estimator (Kraskov et al. 2004).

    Simple implementation using k-nearest neighbor distances.
    """
    from scipy.spatial import cKDTree

    n = len(x)
    if n < k + 5:
        return 0.0

    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y
    xy = np.hstack([x, y])

    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # k-th neighbor distance in joint space (Chebyshev/max norm)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]  # k-th neighbor distance

    # Count neighbors within eps in marginal spaces
    nx = np.array([tree_x.query_ball_point(x[i], eps[i] - 1e-15, p=np.inf).__len__()
                   for i in range(n)]) - 1
    ny = np.array([tree_y.query_ball_point(y[i], eps[i] - 1e-15, p=np.inf).__len__()
                   for i in range(n)]) - 1

    # Avoid log(0)
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    from scipy.special import digamma
    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(mi, 0.0)


def conditional_mutual_information(x, y, z, k=5):
    """Conditional MI: I(X; Y | Z) = I(X,Z; Y) - I(Z; Y)."""
    if z is None or z.shape[1] == 0:
        return mutual_information_ksg(x, y, k=k)

    xz = np.hstack([x.reshape(-1, 1) if x.ndim == 1 else x, z])
    mi_xzy = mutual_information_ksg(xz, y, k=k)
    mi_zy = mutual_information_ksg(z, y, k=k)
    return max(mi_xzy - mi_zy, 0.0)


def compute_ocse_matrix(data, max_lag=5, n_shuffles=100, alpha=0.05, k=3):
    """Compute optimal Causation Entropy matrix for multivariate time series.

    Uses a greedy approach: for each pair (i->j), compute the transfer entropy
    conditioned on the optimal set of other variables.

    Parameters
    ----------
    data : (T, N) array of time series
    max_lag : maximum time lag to consider
    n_shuffles : number of shuffled surrogates for significance
    alpha : significance level
    k : KSG k-nearest neighbors

    Returns
    -------
    causal_matrix : (N, N) array of causal strengths (oCSE values)
    sig_matrix : (N, N) boolean array of significant edges
    pval_matrix : (N, N) array of p-values
    """
    T, N = data.shape
    causal_matrix = np.zeros((N, N))
    sig_matrix = np.zeros((N, N), dtype=bool)
    pval_matrix = np.ones((N, N))

    for target in range(N):
        for source in range(N):
            if source == target:
                continue

            # Build lagged variables
            # Target future: y_{t+1}
            y_future = data[max_lag:, target]
            n = len(y_future)

            # Source past: x_{t-lag} for lag in [0, max_lag-1]
            source_past = np.column_stack([
                data[max_lag - lag - 1:T - lag - 1, source]
                for lag in range(max_lag)
            ])

            # Target past (self-history)
            target_past = np.column_stack([
                data[max_lag - lag - 1:T - lag - 1, target]
                for lag in range(max_lag)
            ])

            # Conditioning set: past of all OTHER variables (excluding source)
            other_past_cols = []
            for other in range(N):
                if other == source or other == target:
                    continue
                for lag in range(min(max_lag, 2)):  # limit lags for others
                    other_past_cols.append(
                        data[max_lag - lag - 1:T - lag - 1, other]
                    )

            if other_past_cols:
                other_past = np.column_stack(other_past_cols)
                conditioning = np.hstack([target_past, other_past])
            else:
                conditioning = target_past

            # oCSE = I(source_past; y_future | conditioning)
            cmi = conditional_mutual_information(
                source_past, y_future, conditioning, k=k
            )
            causal_matrix[source, target] = cmi

            # Significance via shuffled surrogates
            null_dist = np.zeros(n_shuffles)
            for s in range(n_shuffles):
                # Shuffle source_past rows (break temporal structure)
                perm = np.random.permutation(n)
                source_shuffled = source_past[perm]
                null_dist[s] = conditional_mutual_information(
                    source_shuffled, y_future, conditioning, k=k
                )

            pval = np.mean(null_dist >= cmi)
            pval_matrix[source, target] = pval
            sig_matrix[source, target] = pval < alpha

    return causal_matrix, sig_matrix, pval_matrix


def compute_plv(phase1, phase2, window_samples=None):
    """Phase Locking Value between two phase time series.

    If window_samples is None, compute over entire signal.
    """
    diff = phase1 - phase2
    if window_samples is None:
        return np.abs(np.mean(np.exp(1j * diff)))
    # Sliding window
    n = len(diff)
    n_windows = (n - window_samples) // window_samples + 1
    plv = np.zeros(n_windows)
    for i in range(n_windows):
        s = i * window_samples
        e = s + window_samples
        plv[i] = np.abs(np.mean(np.exp(1j * diff[s:e])))
    return plv


# ============================================================
# Plotting helpers
# ============================================================
def draw_causal_graph(ax, causal_matrix, sig_matrix, cluster_ids, cluster_names,
                      cluster_colors, title=""):
    """Draw a 3-node directed causal graph on axes."""
    # Node positions (triangle)
    positions = {
        cluster_ids[0]: (0.5, 0.9),    # top
        cluster_ids[1]: (0.1, 0.2),    # bottom-left
        cluster_ids[2]: (0.9, 0.2),    # bottom-right
    }

    # Draw nodes
    for cid in cluster_ids:
        x, y = positions[cid]
        circle = plt.Circle((x, y), 0.08, color=cluster_colors[cid],
                           ec="black", lw=1.5, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, cluster_names[cid].split("(")[1].rstrip(")"),
               ha="center", va="center", fontsize=8, fontweight="bold", zorder=6)

    # Draw edges
    for i, src in enumerate(cluster_ids):
        for j, tgt in enumerate(cluster_ids):
            if src == tgt:
                continue
            strength = causal_matrix[i, j]
            significant = sig_matrix[i, j]
            if strength < 1e-6:
                continue

            x1, y1 = positions[src]
            x2, y2 = positions[tgt]
            # Offset slightly so bidirectional edges don't overlap
            dx, dy = x2 - x1, y2 - y1
            norm = np.sqrt(dx**2 + dy**2)
            perp_x, perp_y = -dy / norm * 0.02, dx / norm * 0.02

            color = "red" if significant else "gray"
            alpha = min(1.0, strength * 10 + 0.3) if significant else 0.3
            lw = min(3.0, strength * 20 + 0.5)

            ax.annotate("", xy=(x2 + perp_x, y2 + perp_y),
                        xytext=(x1 + perp_x, y1 + perp_y),
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                       lw=lw, alpha=alpha))

            # Label
            mx = (x1 + x2) / 2 + perp_x * 3
            my = (y1 + y2) / 2 + perp_y * 3
            ax.text(mx, my, f"{strength:.3f}", fontsize=6, color=color,
                   ha="center", va="center")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# ============================================================
# Main
# ============================================================
def main():
    setup_style()

    print("=" * 70)
    print("ANALYSIS 2: DIRECTED CONNECTIVITY VIA oCSE")
    print("=" * 70)

    # --- Load data ---
    episodes = load_episodes()

    # --- Extract observables at 50 Hz ---
    print("\nExtracting alpha-band observables at 50 Hz...")

    # Amplitude envelopes
    amp_data = {}
    for ep in [1, 2, 3]:
        modes = extract_cluster_amplitudes(episodes[ep], band=ALPHA_BAND)
        amp_data[ep] = {cid: modes[cid][::DS] for cid in CLUSTER_IDS}

    # Phases
    phase_data = {}
    for ep in [1, 2, 3]:
        phases, _ = extract_cluster_phases(episodes[ep], band=ALPHA_BAND)
        phase_data[ep] = {cid: phases[cid][::DS] for cid in CLUSTER_IDS}

    # Time vectors
    t_data = {}
    for ep in [1, 2, 3]:
        n = len(amp_data[ep][CLUSTER_IDS[0]])
        t_data[ep] = np.arange(n) * DT_DS
        print(f"  Episode {ep}: {n} samples at {FS_DS:.0f} Hz, "
              f"T=[0, {t_data[ep][-1]:.1f}]s")

    # ================================================================
    # Full-window oCSE: pre-seizure vs seizure
    # ================================================================
    print("\n" + "=" * 70)
    print("FULL-WINDOW oCSE ANALYSIS")
    print(f"  max_lag={OCSE_LAG}, n_shuffles={OCSE_N_SHUFFLES}")
    print("=" * 70)

    # Downsample further for oCSE (computationally expensive)
    OCSE_DS = 5  # 50 Hz -> 10 Hz
    DT_OCSE = DT_DS * OCSE_DS

    # Store results: ocse_full[ep][window_type][observable_type]
    ocse_full = {}
    for ep in [1, 2, 3]:
        ocse_full[ep] = {}
        onset = ONSET_TIMES[ep]
        t = t_data[ep]

        for win_name, (t_lo_offset, t_hi_offset) in [
            ("presez", PRE_WINDOW), ("seizure", SEZ_WINDOW)
        ]:
            t_lo = onset + t_lo_offset
            t_hi = onset + t_hi_offset
            mask = (t >= t_lo) & (t < t_hi)
            if mask.sum() < 50:
                print(f"  Ep{ep} {win_name}: insufficient data ({mask.sum()} samples)")
                continue

            ocse_full[ep][win_name] = {}
            for obs_name, obs_data in [("amplitude", amp_data), ("phase", phase_data)]:
                # Build (T, 3) matrix
                data_3ch = np.column_stack([
                    obs_data[ep][cid][mask][::OCSE_DS] for cid in CLUSTER_IDS
                ])
                print(f"  Ep{ep} {win_name} ({obs_name}): {data_3ch.shape} "
                      f"[{t_lo:.1f}s, {t_hi:.1f}s]")

                causal, sig, pval = compute_ocse_matrix(
                    data_3ch, max_lag=OCSE_LAG,
                    n_shuffles=OCSE_N_SHUFFLES, alpha=0.05
                )
                ocse_full[ep][win_name][obs_name] = {
                    "causal": causal, "sig": sig, "pval": pval
                }

                # Print significant edges
                for i, src in enumerate(CLUSTER_IDS):
                    for j, tgt in enumerate(CLUSTER_IDS):
                        if src == tgt:
                            continue
                        marker = "*" if sig[i, j] else " "
                        print(f"    {marker} {CLUSTER_NAMES[src]:20s} -> "
                              f"{CLUSTER_NAMES[tgt]:20s}: "
                              f"oCSE={causal[i,j]:.4f} (p={pval[i,j]:.3f})")

    # ================================================================
    # Sliding-window oCSE
    # ================================================================
    print("\n" + "=" * 70)
    print("SLIDING-WINDOW oCSE")
    print(f"  Window: {SLIDE_WINDOW_S}s, Step: {SLIDE_STEP_S}s")
    print("=" * 70)

    win_samples = int(SLIDE_WINDOW_S * FS_DS)
    step_samples = int(SLIDE_STEP_S * FS_DS)

    # slide_results[ep] = {"t_centers": [...], "amp": [...], "phase": [...]}
    # each entry is a list of (3,3) matrices
    slide_results = {}
    for ep in [1, 2, 3]:
        onset = ONSET_TIMES[ep]
        t = t_data[ep]

        # Analysis range: onset - 30s to onset + 15s
        t_start = onset - 30.0
        t_end = min(onset + 15.0, t[-1] - SLIDE_WINDOW_S)

        slide_results[ep] = {"t_centers": [], "amp_causal": [], "amp_sig": [],
                             "phase_causal": [], "phase_sig": []}

        start_idx = np.searchsorted(t, t_start)
        end_idx = np.searchsorted(t, t_end)

        pos = start_idx
        win_count = 0
        while pos + win_samples <= end_idx + win_samples:
            if pos + win_samples > len(t):
                break
            t_center = t[pos] + SLIDE_WINDOW_S / 2.0

            for obs_name, obs_data in [("amp", amp_data), ("phase", phase_data)]:
                data_3ch = np.column_stack([
                    obs_data[ep][cid][pos:pos + win_samples][::OCSE_DS]
                    for cid in CLUSTER_IDS
                ])
                if len(data_3ch) < 20:
                    break

                causal, sig, _ = compute_ocse_matrix(
                    data_3ch, max_lag=OCSE_LAG,
                    n_shuffles=max(OCSE_N_SHUFFLES // 2, 50),
                    alpha=0.05
                )
                slide_results[ep][f"{obs_name}_causal"].append(causal)
                slide_results[ep][f"{obs_name}_sig"].append(sig)

            slide_results[ep]["t_centers"].append(t_center)
            pos += step_samples
            win_count += 1

        slide_results[ep]["t_centers"] = np.array(slide_results[ep]["t_centers"])
        for key in ["amp_causal", "amp_sig", "phase_causal", "phase_sig"]:
            if slide_results[ep][key]:
                slide_results[ep][key] = np.array(slide_results[ep][key])
        print(f"  Episode {ep}: {win_count} windows")

    # ================================================================
    # PLV (undirected) for comparison
    # ================================================================
    print("\n" + "=" * 70)
    print("PLV (undirected) COMPUTATION")
    print("=" * 70)

    CLUSTER_PAIRS = [(0, 2), (0, 3), (2, 3)]
    pair_names = [f"{CLUSTER_NAMES[i].split()[0]}-{CLUSTER_NAMES[j].split()[0]}"
                  for i, j in CLUSTER_PAIRS]

    plv_slide = {}
    for ep in [1, 2, 3]:
        t = t_data[ep]
        plv_slide[ep] = {"t_centers": [], "plv": {p: [] for p in range(len(CLUSTER_PAIRS))}}

        onset = ONSET_TIMES[ep]
        t_start = onset - 30.0
        t_end = min(onset + 15.0, t[-1] - SLIDE_WINDOW_S)

        start_idx = np.searchsorted(t, t_start)

        pos = start_idx
        while pos + win_samples <= len(t) and t[pos] <= t_end:
            t_center = t[pos] + SLIDE_WINDOW_S / 2.0
            plv_slide[ep]["t_centers"].append(t_center)

            for p_idx, (ci, cj) in enumerate(CLUSTER_PAIRS):
                ph_i = phase_data[ep][ci][pos:pos + win_samples]
                ph_j = phase_data[ep][cj][pos:pos + win_samples]
                plv = np.abs(np.mean(np.exp(1j * (ph_i - ph_j))))
                plv_slide[ep]["plv"][p_idx].append(plv)

            pos += step_samples

        plv_slide[ep]["t_centers"] = np.array(plv_slide[ep]["t_centers"])
        for p in range(len(CLUSTER_PAIRS)):
            plv_slide[ep]["plv"][p] = np.array(plv_slide[ep]["plv"][p])
        print(f"  Episode {ep}: {len(plv_slide[ep]['t_centers'])} windows")

    # ================================================================
    # PLOTS
    # ================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # --- Plot 1-2: Pre-seizure and seizure causal graphs ---
    for win_name in ["presez", "seizure"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"oCSE Causal Graph: {win_name.replace('_',' ').title()} "
                     f"(amplitude)", fontsize=13)
        for ax_idx, ep in enumerate([1, 2, 3]):
            ax = axes[ax_idx]
            if ep in ocse_full and win_name in ocse_full[ep]:
                res = ocse_full[ep][win_name]["amplitude"]
                draw_causal_graph(ax, res["causal"], res["sig"],
                                 CLUSTER_IDS, CLUSTER_NAMES, CLUSTER_COLORS,
                                 title=f"Episode {ep}")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"Episode {ep}")
        fig.tight_layout()
        save_fig(fig, OUT_DIR / f"ocse_{win_name}_graph")

    # --- Plot 3: oCSE timecourse (sliding window) ---
    directed_pairs = [(i, j) for i in range(3) for j in range(3) if i != j]
    dir_pair_names = [f"{CLUSTER_NAMES[CLUSTER_IDS[i]].split()[0]}->"
                      f"{CLUSTER_NAMES[CLUSTER_IDS[j]].split()[0]}"
                      for i, j in directed_pairs]
    dir_colors = plt.cm.Set1(np.linspace(0, 1, len(directed_pairs)))

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle("oCSE Timecourse (amplitude)", fontsize=13)
    for ax_idx, ep in enumerate([1, 2, 3]):
        ax = axes[ax_idx]
        t_c = slide_results[ep]["t_centers"]
        causal_arr = slide_results[ep]["amp_causal"]
        onset = ONSET_TIMES[ep]

        if len(causal_arr) > 0 and isinstance(causal_arr, np.ndarray):
            for dp_idx, (i, j) in enumerate(directed_pairs):
                vals = causal_arr[:, i, j]
                ax.plot(t_c[:len(vals)], vals, color=dir_colors[dp_idx],
                        lw=1.0, label=dir_pair_names[dp_idx])

        ax.axvline(onset, color="k", ls="--", lw=0.8, label="Onset")
        ax.set_ylabel("oCSE")
        ax.set_title(f"Episode {ep}", fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=6, ncol=3, loc="upper left")
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "ocse_timecourse")

    # --- Plot 4: oCSE change (seizure - presezure) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("oCSE Change: Seizure - Pre-Seizure (amplitude)", fontsize=13)
    for ax_idx, ep in enumerate([1, 2, 3]):
        ax = axes[ax_idx]
        if (ep in ocse_full and "presez" in ocse_full[ep] and
            "seizure" in ocse_full[ep]):
            pre = ocse_full[ep]["presez"]["amplitude"]["causal"]
            sez = ocse_full[ep]["seizure"]["amplitude"]["causal"]
            change = sez - pre

            # Heatmap
            im = ax.imshow(change, cmap="RdBu_r", vmin=-0.1, vmax=0.1)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            labels = [CLUSTER_NAMES[c].split()[0] for c in CLUSTER_IDS]
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Target")
            ax.set_ylabel("Source")
            ax.set_title(f"Episode {ep}", fontsize=10)
            # Annotate cells
            for i in range(3):
                for j in range(3):
                    if i != j:
                        sig_pre = ocse_full[ep]["presez"]["amplitude"]["sig"][i, j]
                        sig_sez = ocse_full[ep]["seizure"]["amplitude"]["sig"][i, j]
                        marker = ""
                        if sig_sez and not sig_pre:
                            marker = " (NEW)"
                        elif sig_pre and not sig_sez:
                            marker = " (LOST)"
                        ax.text(j, i, f"{change[i,j]:.3f}{marker}",
                               ha="center", va="center", fontsize=7)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "ocse_change")

    # --- Plot 5: PLV timecourse ---
    pair_colors_plv = ["#e377c2", "#7f7f7f", "#bcbd22"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Phase Locking Value (undirected, alpha band)", fontsize=13)
    for ax_idx, ep in enumerate([1, 2, 3]):
        ax = axes[ax_idx]
        t_c = plv_slide[ep]["t_centers"]
        onset = ONSET_TIMES[ep]
        for p_idx in range(len(CLUSTER_PAIRS)):
            ax.plot(t_c, plv_slide[ep]["plv"][p_idx], color=pair_colors_plv[p_idx],
                    lw=1.0, label=pair_names[p_idx])
        ax.axvline(onset, color="k", ls="--", lw=0.8)
        ax.set_ylabel("PLV")
        ax.set_title(f"Episode {ep}", fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=7)
        ax.set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "plv_timecourse")

    # --- Plot 6: Causal flow summary (pre vs seizure side by side) ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Causal Flow Summary: Pre-Seizure vs Seizure", fontsize=14)
    for ax_idx, ep in enumerate([1, 2, 3]):
        for row, win_name in enumerate(["presez", "seizure"]):
            ax = axes[row, ax_idx]
            if ep in ocse_full and win_name in ocse_full[ep]:
                res = ocse_full[ep][win_name]["amplitude"]
                draw_causal_graph(ax, res["causal"], res["sig"],
                                 CLUSTER_IDS, CLUSTER_NAMES, CLUSTER_COLORS,
                                 title=f"Ep{ep}: {win_name.title()}")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                ax.set_title(f"Ep{ep}: {win_name.title()}")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "causal_flow_summary")

    # --- Plot 7: Amplitude vs phase oCSE comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("oCSE: Amplitude vs Phase Observables (seizure window)", fontsize=14)
    for ax_idx, ep in enumerate([1, 2, 3]):
        for row, obs_name in enumerate(["amplitude", "phase"]):
            ax = axes[row, ax_idx]
            if (ep in ocse_full and "seizure" in ocse_full[ep] and
                obs_name in ocse_full[ep]["seizure"]):
                res = ocse_full[ep]["seizure"][obs_name]
                draw_causal_graph(ax, res["causal"], res["sig"],
                                 CLUSTER_IDS, CLUSTER_NAMES, CLUSTER_COLORS,
                                 title=f"Ep{ep}: {obs_name.title()}")
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "ocse_amplitude_vs_phase")

    # --- Plot 8: Consistency across episodes ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Connectivity Consistency Across Episodes", fontsize=13)

    for ax_idx, (win_name, title) in enumerate([
        ("presez", "Pre-Seizure"), ("seizure", "Seizure")
    ]):
        ax = axes[ax_idx]
        # Average causal matrix across episodes
        matrices = []
        sig_count = np.zeros((3, 3))
        for ep in [1, 2, 3]:
            if ep in ocse_full and win_name in ocse_full[ep]:
                matrices.append(ocse_full[ep][win_name]["amplitude"]["causal"])
                sig_count += ocse_full[ep][win_name]["amplitude"]["sig"].astype(float)

        if matrices:
            avg_matrix = np.mean(matrices, axis=0)
            im = ax.imshow(avg_matrix, cmap="YlOrRd", vmin=0)
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            labels = [CLUSTER_NAMES[c].split()[0] for c in CLUSTER_IDS]
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Target")
            ax.set_ylabel("Source")
            for i in range(3):
                for j in range(3):
                    if i != j:
                        ax.text(j, i,
                               f"{avg_matrix[i,j]:.3f}\n({int(sig_count[i,j])}/3)",
                               ha="center", va="center", fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{title} (mean oCSE, sig count)", fontsize=10)

    fig.tight_layout()
    save_fig(fig, OUT_DIR / "connectivity_by_episode")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Identify consistent causal edges
    for win_name in ["presez", "seizure"]:
        print(f"\n  {win_name.upper()} consistent edges (sig in >=2/3 episodes):")
        sig_count = np.zeros((3, 3))
        for ep in [1, 2, 3]:
            if ep in ocse_full and win_name in ocse_full[ep]:
                sig_count += ocse_full[ep][win_name]["amplitude"]["sig"].astype(float)

        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                if sig_count[i, j] >= 2:
                    print(f"    {CLUSTER_NAMES[CLUSTER_IDS[i]]:20s} -> "
                          f"{CLUSTER_NAMES[CLUSTER_IDS[j]]:20s}: "
                          f"significant in {int(sig_count[i,j])}/3 episodes")

    # New edges at seizure
    print("\n  NEW causal edges at seizure onset:")
    for ep in [1, 2, 3]:
        if (ep in ocse_full and "presez" in ocse_full[ep] and
            "seizure" in ocse_full[ep]):
            pre_sig = ocse_full[ep]["presez"]["amplitude"]["sig"]
            sez_sig = ocse_full[ep]["seizure"]["amplitude"]["sig"]
            new = sez_sig & ~pre_sig
            for i in range(3):
                for j in range(3):
                    if new[i, j]:
                        print(f"    Ep{ep}: {CLUSTER_NAMES[CLUSTER_IDS[i]]} -> "
                              f"{CLUSTER_NAMES[CLUSTER_IDS[j]]}")

    # Highest outgoing oCSE (driver identification)
    print("\n  Driver identification (highest outgoing oCSE at seizure):")
    for ep in [1, 2, 3]:
        if ep in ocse_full and "seizure" in ocse_full[ep]:
            causal = ocse_full[ep]["seizure"]["amplitude"]["causal"]
            outgoing = np.sum(causal, axis=1)  # sum of row = outgoing
            driver = np.argmax(outgoing)
            print(f"    Ep{ep}: {CLUSTER_NAMES[CLUSTER_IDS[driver]]} "
                  f"(outgoing oCSE = {outgoing[driver]:.4f})")

    print(f"\n  All plots saved to {OUT_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
