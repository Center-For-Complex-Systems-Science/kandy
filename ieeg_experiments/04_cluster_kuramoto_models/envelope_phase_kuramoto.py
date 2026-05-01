#!/usr/bin/env python3
"""Analysis 1: Envelope-of-Envelope Phase Dynamics (Kuramoto on slow phases).

Extract the slow (~0.1-0.5 Hz) phase of the alpha amplitude envelope per cluster,
then fit a Kuramoto-style coupling model using KANDy.

Pipeline:
  1. Alpha-band Hilbert envelope per cluster (SVD mode 0), 50 Hz
  2. Envelope PSD to find dominant slow oscillation frequency
  3. Bandpass envelope at identified frequency -> Hilbert -> slow phase psi_k(t)
  4. Downsample slow phase to 10 Hz, compute dpsi/dt via Savitzky-Golay
  5. Kuramoto lift (15 features):
     - sin(psi_i - psi_j), cos(psi_i - psi_j) for 3 unique pairs (6)
     - r_k envelope order parameter for 3 clusters (3)
     - ReLU(r_k - theta_k) for 3 clusters (3)
     - ReLU(r_k - theta_k) * sin(psi_k - psi_mean) for 3 clusters (3)
  6. Train KANDy [15, 3], grid=5, steps=200
  7. Rollout, edge activations, symbolic extraction

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import sys
import numpy as np
import torch
import sympy as sp
from scipy.signal import hilbert, butter, filtfilt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# Add utilities directory for imports
sys.path.insert(0, str(ROOT / "ieeg_experiments" / "03_clustering_and_supporting_analysis"))
from ieeg_utils import (
    load_episodes, extract_cluster_amplitudes, savgol_derivative, save_fig,
    setup_style, bandpass, FS, ONSET_TIMES, CLUSTERS, CLUSTER_IDS, CLUSTER_NAMES,
    CLUSTER_COLORS, ALPHA_BAND,
)

# KANDy imports
sys.path.insert(0, str(ROOT / "src"))
from kandy import KANDy, CustomLift
from kandy.symbolic import make_symbolic_lib, robust_auto_symbolic
from kandy.plotting import plot_all_edges, plot_loss_curves, use_pub_style

# ============================================================
# Parameters
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DS_ENVELOPE = 10    # 500 Hz -> 50 Hz for envelope extraction
DS_PHASE = 5        # 50 Hz -> 10 Hz for phase dynamics
DT_ENV = DS_ENVELOPE / FS   # 0.02s
FS_ENV = FS / DS_ENVELOPE   # 50 Hz
DT_PHASE = DT_ENV * DS_PHASE  # 0.1s
FS_PHASE = 1.0 / DT_PHASE     # 10 Hz

# Slow oscillation frequency search range
SLOW_FREQ_RANGE = (0.05, 2.0)  # Hz

# SG derivative params at 10 Hz
SG_WIN = 15     # ~1.5s window
SG_POLY = 4

# KANDy hyperparameters
GRID = 5
K_SPLINE = 3
STEPS = 200
LAMB = 0.0

# Unique cluster pairs (i < j)
CLUSTER_PAIRS = [(0, 2), (0, 3), (2, 3)]
N_PAIRS = len(CLUSTER_PAIRS)
N_CLUSTERS = len(CLUSTER_IDS)

# Feature count: 6 (sin/cos phase diffs) + 3 (r_k) + 3 (ReLU) + 3 (ReLU*sin) = 15
N_FEAT = 6 + 3 + 3 + 3

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================
def find_dominant_frequency(signal, fs, freq_range):
    """Find dominant frequency in a signal within freq_range via FFT."""
    n = len(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    fft_vals = np.abs(np.fft.rfft(signal - signal.mean()))
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not mask.any():
        return 0.0, freqs, fft_vals
    peak_idx = np.argmax(fft_vals[mask])
    peak_freq = freqs[mask][peak_idx]
    return peak_freq, freqs, fft_vals


def extract_slow_phase(envelope, fs, band_lo, band_hi):
    """Bandpass envelope at slow frequency, then Hilbert to get phase."""
    nyq = fs / 2.0
    lo = max(band_lo / nyq, 0.001)
    hi = min(band_hi / nyq, 0.999)
    if lo >= hi:
        return np.zeros_like(envelope), np.zeros_like(envelope)
    b, a = butter(3, [lo, hi], btype="band")
    env_filt = filtfilt(b, a, envelope)
    analytic = hilbert(env_filt)
    phase = np.angle(analytic)
    amplitude = np.abs(analytic)
    return phase, env_filt


def compute_order_param(phases_dict, cluster_ids):
    """Compute per-cluster order parameter r_k from phase time series."""
    # For cluster-level: r_k = |exp(i*psi_k)| = 1 (single phase per cluster)
    # Instead, use the envelope amplitude as a proxy for cluster coherence
    # This is actually the envelope amplitude, not phase-based r
    # We'll use the amplitude of the bandpassed envelope
    pass


# ============================================================
# Main
# ============================================================
def main():
    setup_style()

    print("=" * 70)
    print("ANALYSIS 1: ENVELOPE-OF-ENVELOPE PHASE DYNAMICS")
    print("=" * 70)

    # --- Load data ---
    episodes = load_episodes()

    # --- Step 1: Extract alpha envelopes at 50 Hz ---
    print("\nStep 1: Extracting alpha-band envelopes at 50 Hz...")
    env_data = {}   # env_data[ep][cid] = (T_env,) amplitude at 50 Hz
    t_env = {}
    for ep in [1, 2, 3]:
        modes = extract_cluster_amplitudes(episodes[ep], band=ALPHA_BAND)
        env_data[ep] = {cid: modes[cid][::DS_ENVELOPE] for cid in CLUSTER_IDS}
        n_env = len(env_data[ep][CLUSTER_IDS[0]])
        t_env[ep] = np.arange(n_env) * DT_ENV
        print(f"  Episode {ep}: {n_env} samples at {FS_ENV:.0f} Hz")

    # --- Step 2: Envelope PSD to find dominant slow frequency ---
    print("\nStep 2: Envelope PSD analysis...")
    peak_freqs = {}
    all_freqs_dict = {}
    all_fft_dict = {}
    for cid in CLUSTER_IDS:
        # Use episode 1 for frequency estimation (longest pre-seizure)
        sig = env_data[1][cid]
        peak_f, freqs, fft_v = find_dominant_frequency(sig, FS_ENV, SLOW_FREQ_RANGE)
        peak_freqs[cid] = peak_f
        all_freqs_dict[cid] = freqs
        all_fft_dict[cid] = fft_v
        print(f"  {CLUSTER_NAMES[cid]}: peak slow frequency = {peak_f:.3f} Hz")

    # Use the median peak frequency across clusters as the target band center
    median_peak = np.median(list(peak_freqs.values()))
    # Bandpass: median_peak * [0.5, 2.0] to capture the oscillation
    slow_band_lo = max(median_peak * 0.5, 0.03)
    slow_band_hi = min(median_peak * 2.0, FS_ENV / 2.0 - 0.1)
    print(f"\n  Target slow band: [{slow_band_lo:.3f}, {slow_band_hi:.3f}] Hz "
          f"(center: {median_peak:.3f} Hz)")

    # --- Plot: Envelope PSD ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Envelope PSD (alpha band, 50 Hz)", fontsize=13)
    for ax_idx, cid in enumerate(CLUSTER_IDS):
        ax = axes[ax_idx]
        freqs = all_freqs_dict[cid]
        fft_v = all_fft_dict[cid]
        mask = (freqs > 0) & (freqs <= 5.0)
        ax.semilogy(freqs[mask], fft_v[mask], color=CLUSTER_COLORS[cid], lw=0.8)
        ax.axvline(peak_freqs[cid], color="red", ls="--", lw=0.8,
                  label=f"Peak: {peak_freqs[cid]:.3f} Hz")
        ax.axvspan(slow_band_lo, slow_band_hi, alpha=0.1, color="orange",
                  label="Target band")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("FFT amplitude")
        ax.set_title(CLUSTER_NAMES[cid], fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "envelope_psd")

    # --- Step 3: Extract slow phase from bandpassed envelope ---
    print("\nStep 3: Extracting slow phase from bandpassed envelope...")
    phase_data = {}   # phase_data[ep][cid] = (T_env,) phase at 50 Hz
    env_filt_data = {}
    for ep in [1, 2, 3]:
        phase_data[ep] = {}
        env_filt_data[ep] = {}
        for cid in CLUSTER_IDS:
            phase, env_filt = extract_slow_phase(
                env_data[ep][cid], FS_ENV, slow_band_lo, slow_band_hi
            )
            phase_data[ep][cid] = phase
            env_filt_data[ep][cid] = env_filt

    # --- Plot: Envelope verification ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle("Bandpassed Envelope Verification (Episode 1)", fontsize=13)
    for ax_idx, cid in enumerate(CLUSTER_IDS):
        ax = axes[ax_idx]
        t = t_env[1]
        ax.plot(t, env_data[1][cid], color="gray", lw=0.5, alpha=0.5, label="Raw envelope")
        ax.plot(t, env_filt_data[1][cid], color=CLUSTER_COLORS[cid], lw=1.2,
                label="Bandpassed")
        ax.axvline(ONSET_TIMES[1], color="k", ls="--", lw=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(CLUSTER_NAMES[cid], fontsize=10)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "envelope_verification")

    # --- Step 4: Downsample to 10 Hz, compute dpsi/dt ---
    print("\nStep 4: Downsample to 10 Hz, compute derivatives...")
    psi_data = {}   # psi_data[ep][cid] = (T_phase,) at 10 Hz
    dpsi_data = {}  # dpsi_data[ep][cid] = (T_phase,) at 10 Hz
    r_data = {}     # r_data[ep][cid] = envelope amplitude of slow oscillation
    t_phase = {}
    for ep in [1, 2, 3]:
        psi_data[ep] = {}
        dpsi_data[ep] = {}
        r_data[ep] = {}
        for cid in CLUSTER_IDS:
            psi = phase_data[ep][cid][::DS_PHASE]
            psi_data[ep][cid] = psi
            # Envelope amplitude (proxy for cluster coherence at slow timescale)
            analytic = hilbert(env_filt_data[ep][cid])
            r = np.abs(analytic)[::DS_PHASE]
            r_data[ep][cid] = r
            # Derivative via SG (unwrap phase first)
            psi_unwrapped = np.unwrap(psi)
            dpsi = savgol_derivative(psi_unwrapped, DT_PHASE,
                                    window=SG_WIN, polyorder=SG_POLY)
            dpsi_data[ep][cid] = dpsi

        n_phase = len(psi_data[ep][CLUSTER_IDS[0]])
        t_phase[ep] = np.arange(n_phase) * DT_PHASE
        print(f"  Episode {ep}: {n_phase} samples at {FS_PHASE:.0f} Hz")

    # --- Plot: Slow phase traces ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Slow Phase Traces (unwrapped)", fontsize=13)
    for ax_idx, ep in enumerate([1, 2, 3]):
        ax = axes[ax_idx]
        for cid in CLUSTER_IDS:
            psi_uw = np.unwrap(psi_data[ep][cid])
            ax.plot(t_phase[ep], psi_uw, color=CLUSTER_COLORS[cid],
                    lw=0.8, label=CLUSTER_NAMES[cid])
        ax.axvline(ONSET_TIMES[ep], color="k", ls="--", lw=0.8)
        ax.set_ylabel(r"$\psi$ (rad)")
        ax.set_title(f"Episode {ep}", fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "slow_phase_traces")

    # --- Plot: Slow phase differences ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
    fig.suptitle("Slow Phase Differences", fontsize=13)
    pair_names = [f"{CLUSTER_NAMES[i].split()[0]}-{CLUSTER_NAMES[j].split()[0]}"
                  for i, j in CLUSTER_PAIRS]
    pair_colors = ["#e377c2", "#7f7f7f", "#bcbd22"]
    for ax_idx, ep in enumerate([1, 2, 3]):
        ax = axes[ax_idx]
        for p_idx, (ci, cj) in enumerate(CLUSTER_PAIRS):
            diff = psi_data[ep][ci] - psi_data[ep][cj]
            ax.plot(t_phase[ep], diff, color=pair_colors[p_idx],
                    lw=0.8, label=pair_names[p_idx])
        ax.axvline(ONSET_TIMES[ep], color="k", ls="--", lw=0.8)
        ax.set_ylabel(r"$\Delta\psi$ (rad)")
        ax.set_title(f"Episode {ep}", fontsize=10)
        if ax_idx == 0:
            ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    save_fig(fig, OUT_DIR / "slow_phase_diffs")

    # --- Step 5: Build Kuramoto lift features ---
    print("\nStep 5: Building Kuramoto lift features...")

    # Trim SG boundary artifacts
    TRIM = SG_WIN // 2 + 3

    feature_names = []
    # sin/cos phase diffs (6)
    for ci, cj in CLUSTER_PAIRS:
        feature_names.append(f"sin(psi{ci}-psi{cj})")
        feature_names.append(f"cos(psi{ci}-psi{cj})")
    # r_k (3)
    for cid in CLUSTER_IDS:
        feature_names.append(f"r_{cid}")
    # ReLU(r_k - theta_k) (3)
    for cid in CLUSTER_IDS:
        feature_names.append(f"ReLU(r_{cid})")
    # ReLU * sin(psi_k - psi_mean) (3)
    for cid in CLUSTER_IDS:
        feature_names.append(f"ReLU*sin_{cid}")

    assert len(feature_names) == N_FEAT, f"Expected {N_FEAT} features, got {len(feature_names)}"

    # Estimate ReLU thresholds from pre-seizure r values
    r_thresholds = {}
    for cid in CLUSTER_IDS:
        pre_vals = []
        for ep in [1, 2, 3]:
            onset = ONSET_TIMES[ep]
            t = t_phase[ep]
            pre_mask = (t >= onset - 15) & (t < onset)
            if pre_mask.sum() > 0:
                pre_vals.append(r_data[ep][cid][pre_mask].mean())
        r_thresholds[cid] = np.mean(pre_vals) if pre_vals else 0.0
    print(f"  ReLU thresholds: {r_thresholds}")

    def build_features(ep):
        """Build (T_trimmed, N_FEAT) feature matrix for one episode."""
        T_full = len(psi_data[ep][CLUSTER_IDS[0]])
        feats = np.zeros((T_full, N_FEAT))
        col = 0

        # sin/cos phase diffs
        for ci, cj in CLUSTER_PAIRS:
            diff = psi_data[ep][ci] - psi_data[ep][cj]
            feats[:, col] = np.sin(diff)
            feats[:, col + 1] = np.cos(diff)
            col += 2

        # r_k
        for cid in CLUSTER_IDS:
            feats[:, col] = r_data[ep][cid]
            col += 1

        # ReLU(r_k - theta)
        relu_vals = {}
        for cid in CLUSTER_IDS:
            relu = np.maximum(r_data[ep][cid] - r_thresholds[cid], 0.0)
            feats[:, col] = relu
            relu_vals[cid] = relu
            col += 1

        # ReLU * sin(psi_k - psi_mean)
        psi_mean = np.zeros(T_full)
        for cid in CLUSTER_IDS:
            psi_mean += psi_data[ep][cid]
        psi_mean /= N_CLUSTERS

        for cid in CLUSTER_IDS:
            feats[:, col] = relu_vals[cid] * np.sin(psi_data[ep][cid] - psi_mean)
            col += 1

        assert col == N_FEAT
        return feats[TRIM:-TRIM]

    def build_targets(ep):
        """Build (T_trimmed, N_CLUSTERS) target matrix (dpsi/dt)."""
        tgts = np.column_stack([dpsi_data[ep][cid] for cid in CLUSTER_IDS])
        return tgts[TRIM:-TRIM]

    # Concatenate all episodes
    all_feats = []
    all_targets = []
    boundaries = [0]
    for ep in [1, 2, 3]:
        feats = build_features(ep)
        tgts = build_targets(ep)
        n = min(len(feats), len(tgts))
        all_feats.append(feats[:n])
        all_targets.append(tgts[:n])
        boundaries.append(boundaries[-1] + n)

    X_all = np.vstack(all_feats)
    Y_all = np.vstack(all_targets)
    print(f"  Training data: X={X_all.shape}, Y={Y_all.shape}")
    print(f"  Feature names: {feature_names}")

    # --- Step 6: Train KANDy ---
    print("\nStep 6: Training KANDy...")

    lift = CustomLift(fn=lambda X: X, output_dim=N_FEAT, name="envelope_phase_identity")

    model = KANDy(
        lift=lift,
        grid=GRID,
        k=K_SPLINE,
        steps=STEPS,
        seed=SEED,
        device="cpu",
    )

    model.fit(
        X=X_all,
        X_dot=Y_all,
        val_frac=0.15,
        test_frac=0.15,
        lamb=LAMB,
        patience=0,
        verbose=True,
    )

    # --- One-step evaluation ---
    n_test = min(500, len(X_all) // 5)
    test_X = X_all[-n_test:]
    test_Y = Y_all[-n_test:]
    pred_Y = model.predict(test_X)
    mse = np.mean((pred_Y - test_Y) ** 2)
    ss_res = np.sum((pred_Y - test_Y) ** 2, axis=0)
    ss_tot = np.sum((test_Y - test_Y.mean(axis=0)) ** 2, axis=0)
    r2_per_output = 1.0 - ss_res / np.maximum(ss_tot, 1e-15)
    r2_overall = 1.0 - np.sum(ss_res) / np.sum(ss_tot)
    print(f"\n  One-step MSE: {mse:.6e}")
    print(f"  R^2 overall: {r2_overall:.4f}")
    for i, cid in enumerate(CLUSTER_IDS):
        print(f"    {CLUSTER_NAMES[cid]}: R^2 = {r2_per_output[i]:.4f}")

    # --- Step 7: Rollout, edge activations, symbolic ---
    print("\nStep 7: Post-training analysis...")

    # Rollout (if R^2 > 0)
    if r2_overall > 0:
        print("  Running rollout...")
        # Rollout on episode 1 (pre-seizure segment)
        ep_test = 1
        feats_ep = build_features(ep_test)
        tgts_ep = build_targets(ep_test)
        t_ep = t_phase[ep_test][TRIM:-TRIM]
        n_ep = len(feats_ep)

        # Start from 20% into the episode
        start_idx = n_ep // 5
        N_ROLLOUT = min(300, n_ep - start_idx)

        # State = [psi_0, psi_2, psi_3] (unwrapped phases)
        psi0 = np.array([np.unwrap(psi_data[ep_test][cid])[TRIM + start_idx]
                         for cid in CLUSTER_IDS])

        # Need r values and thresholds for feature reconstruction during rollout
        r_ep = {cid: r_data[ep_test][cid][TRIM:-TRIM] for cid in CLUSTER_IDS}

        def rollout_step(psi_state, step_idx):
            """Compute features from phase state for one timestep."""
            feats = np.zeros(N_FEAT)
            col = 0
            # sin/cos phase diffs
            psi_dict = {cid: psi_state[i] for i, cid in enumerate(CLUSTER_IDS)}
            for ci, cj in CLUSTER_PAIRS:
                diff = psi_dict[ci] - psi_dict[cj]
                feats[col] = np.sin(diff)
                feats[col + 1] = np.cos(diff)
                col += 2
            # r_k (use ground truth r since it's from data, not learned)
            idx = min(start_idx + step_idx, len(r_ep[CLUSTER_IDS[0]]) - 1)
            for cid in CLUSTER_IDS:
                feats[col] = r_ep[cid][idx]
                col += 1
            # ReLU
            relu_vals = {}
            for cid in CLUSTER_IDS:
                relu = max(r_ep[cid][idx] - r_thresholds[cid], 0.0)
                feats[col] = relu
                relu_vals[cid] = relu
                col += 1
            # ReLU * sin(psi_k - psi_mean)
            psi_mean = np.mean(list(psi_dict.values()))
            for cid in CLUSTER_IDS:
                feats[col] = relu_vals[cid] * np.sin(psi_dict[cid] - psi_mean)
                col += 1
            return feats

        traj = [psi0.copy()]
        state = psi0.copy()
        for step in range(N_ROLLOUT - 1):
            def f(s, si=step):
                feats = rollout_step(s, si)[None, :]
                return model.predict(feats).ravel()

            k1 = f(state)
            k2 = f(state + 0.5 * DT_PHASE * k1)
            k3 = f(state + 0.5 * DT_PHASE * k2)
            k4 = f(state + DT_PHASE * k3)
            state = state + (DT_PHASE / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            traj.append(state.copy())

        traj = np.array(traj)  # (N_ROLLOUT, 3)
        true_psi = np.column_stack([
            np.unwrap(psi_data[ep_test][cid])[TRIM + start_idx:TRIM + start_idx + N_ROLLOUT]
            for cid in CLUSTER_IDS
        ])
        n_roll = min(len(traj), len(true_psi))
        traj = traj[:n_roll]
        true_psi = true_psi[:n_roll]

        rmse_rollout = np.sqrt(np.mean((traj - true_psi) ** 2))
        print(f"  Rollout RMSE: {rmse_rollout:.4f}")

        # --- Plot: Rollout ---
        t_roll = np.arange(n_roll) * DT_PHASE
        fig, axes = plt.subplots(N_CLUSTERS, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"Slow Phase Rollout (RMSE={rmse_rollout:.4f})", fontsize=13)
        for i, cid in enumerate(CLUSTER_IDS):
            ax = axes[i]
            ax.plot(t_roll, true_psi[:, i], color=CLUSTER_COLORS[cid],
                    lw=1.2, label="True")
            ax.plot(t_roll, traj[:, i], color=CLUSTER_COLORS[cid],
                    lw=1.0, ls="--", label="KANDy")
            ax.set_ylabel(rf"$\psi_{{{cid}}}$")
            ax.set_title(CLUSTER_NAMES[cid], fontsize=10)
            if i == 0:
                ax.legend(fontsize=7)
        axes[-1].set_xlabel("Time (s)")
        fig.tight_layout()
        save_fig(fig, OUT_DIR / "rollout")

        # --- Plot: Order parameter ---
        r_true = np.abs(np.mean(np.exp(1j * true_psi), axis=1))
        r_pred = np.abs(np.mean(np.exp(1j * traj), axis=1))
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_roll, r_true, color="steelblue", lw=1.2, label="True")
        ax.plot(t_roll, r_pred, color="tomato", lw=1.0, ls="--", label="KANDy")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("r(t)")
        ax.set_title("Slow Phase Order Parameter")
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_fig(fig, OUT_DIR / "order_parameter")

    # --- Edge activations ---
    print("  Plotting edge activations...")
    use_pub_style()
    n_sub = min(5000, int(len(X_all) * 0.7))
    sub_idx = np.random.choice(int(len(X_all) * 0.7), n_sub, replace=False)
    train_X_t = torch.tensor(X_all[sub_idx], dtype=torch.float32)
    try:
        fig_edges, _ = plot_all_edges(
            model.model_,
            X=train_X_t,
            in_var_names=feature_names,
            out_var_names=[f"dpsi_{cid}/dt" for cid in CLUSTER_IDS],
            save=str(OUT_DIR / "edge_activations"),
        )
        plt.close(fig_edges)
    except Exception as e:
        print(f"  [WARN] Edge plot failed: {e}")

    # --- Loss curves ---
    if hasattr(model, "train_results_") and model.train_results_ is not None:
        fig_loss, _ = plot_loss_curves(
            model.train_results_,
            save=str(OUT_DIR / "loss_curves"),
        )
        plt.close(fig_loss)

    # --- Symbolic extraction ---
    if r2_overall > 0:
        print("  Symbolic extraction...")
        LINEAR_LIB = make_symbolic_lib({
            "x": (lambda x: x, lambda x: x, 1),
            "0": (lambda x: x * 0, lambda x: x * 0, 0),
        })

        sym_subset = torch.tensor(X_all[:2048], dtype=torch.float32)
        model.model_.save_act = True
        with torch.no_grad():
            model.model_(sym_subset)

        robust_auto_symbolic(
            model.model_,
            lib=LINEAR_LIB,
            r2_threshold=0.85,
            weight_simple=0.8,
            topk_edges=20,
        )

        try:
            exprs, vars_ = model.model_.symbolic_formula()
            sub_map = {sp.Symbol(str(v)): sp.Symbol(n)
                      for v, n in zip(vars_, feature_names)}
            for i, cid in enumerate(CLUSTER_IDS):
                sym = sp.sympify(exprs[i]).xreplace(sub_map)
                sym = sp.expand(sym).xreplace(
                    {n: round(float(n), 4) for n in sym.atoms(sp.Number)}
                )
                print(f"  dpsi_{cid}/dt = {sym}")
        except Exception as e:
            print(f"  [WARN] Symbolic formula extraction failed: {e}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Slow oscillation band: [{slow_band_lo:.3f}, {slow_band_hi:.3f}] Hz")
    print(f"  Peak frequencies: {peak_freqs}")
    print(f"  One-step R^2: {r2_overall:.4f}")
    for i, cid in enumerate(CLUSTER_IDS):
        print(f"    {CLUSTER_NAMES[cid]}: R^2 = {r2_per_output[i]:.4f}")
    if r2_overall > 0.1:
        print("  -> SIGNAL DETECTED: Slow Kuramoto coupling exists")
    elif r2_overall > 0:
        print("  -> WEAK SIGNAL: Some coupling, may need different features")
    else:
        print("  -> NO SIGNAL: Kuramoto wrong for this data at slow timescale")

    print(f"\n  All plots saved to {OUT_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
