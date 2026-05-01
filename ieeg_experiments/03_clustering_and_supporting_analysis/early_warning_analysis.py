#!/usr/bin/env python3
"""Analysis 3: Early Warning Signals + Stuart-Landau Fitting.

Detect pre-seizure critical slowing down (CSD) and determine bifurcation type
(Hopf vs saddle-focus) for each SOZ subcluster across 3 seizure episodes.

EWS indicators (20s sliding window, 1s step):
  - Variance, lag-1 autocorrelation, skewness (CSD markers)
  - DFA exponent alpha (long-range correlations)
  - Spectral index P_low/P_high (spectral reddening)
  - Kendall tau trend test per indicator

Stuart-Landau OLS fit: dA/dt = mu*A - alpha*A^3
  - Track mu(t): should cross zero at seizure onset (supercritical Hopf)
  - Phase portraits (A, dA/dt) colored by epoch

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import sys
import numpy as np
from scipy import stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add local utilities directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ieeg_utils import (
    load_episodes, extract_cluster_amplitudes, savgol_derivative, save_fig,
    setup_style, FS, ONSET_TIMES, CLUSTERS, CLUSTER_IDS, CLUSTER_NAMES,
    CLUSTER_COLORS, ALPHA_BAND,
)

# ============================================================
# Parameters
# ============================================================
DS = 100          # 500 Hz -> 5 Hz
DT_DS = DS / FS   # 0.2 s
FS_DS = 1.0 / DT_DS  # 5 Hz

# EWS sliding window parameters
EWS_WINDOW_S = 20.0    # 20s window
EWS_STEP_S = 1.0       # 1s step
EWS_END_OFFSET_S = 40.0  # start EWS analysis 40s before onset

# Stuart-Landau window
SL_WINDOW_S = 10.0
SL_STEP_S = 2.0

OUT_DIR = Path(__file__).resolve().parent
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EWS computation functions
# ============================================================
def compute_variance(x):
    return np.var(x)


def compute_lag1_autocorrelation(x):
    if len(x) < 3:
        return 0.0
    x_c = x - x.mean()
    c0 = np.sum(x_c ** 2)
    if c0 < 1e-15:
        return 0.0
    c1 = np.sum(x_c[:-1] * x_c[1:])
    return c1 / c0


def compute_skewness(x):
    return float(stats.skew(x, bias=False))


def compute_dfa_exponent(x, min_box=4, max_box=None):
    """Detrended Fluctuation Analysis exponent."""
    n = len(x)
    if max_box is None:
        max_box = n // 4
    if max_box < min_box + 2:
        return 0.5

    # Cumulative sum (profile)
    y = np.cumsum(x - x.mean())

    # Box sizes (log-spaced)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), num=20
    ).astype(int))
    box_sizes = box_sizes[(box_sizes >= min_box) & (box_sizes <= max_box)]
    if len(box_sizes) < 3:
        return 0.5

    fluct = []
    for bs in box_sizes:
        n_boxes = n // bs
        if n_boxes < 1:
            continue
        rms_vals = []
        for i in range(n_boxes):
            segment = y[i * bs:(i + 1) * bs]
            t_seg = np.arange(bs)
            coeffs = np.polyfit(t_seg, segment, 1)
            trend = np.polyval(coeffs, t_seg)
            rms_vals.append(np.sqrt(np.mean((segment - trend) ** 2)))
        if rms_vals:
            fluct.append(np.mean(rms_vals))
        else:
            fluct.append(np.nan)

    fluct = np.array(fluct)
    valid = ~np.isnan(fluct) & (fluct > 0)
    if valid.sum() < 3:
        return 0.5
    log_n = np.log10(box_sizes[valid].astype(float))
    log_f = np.log10(fluct[valid])
    slope, _, _, _, _ = stats.linregress(log_n, log_f)
    return slope


def compute_spectral_index(x, fs, f_split=None):
    """Ratio of low-frequency to high-frequency power (spectral reddening)."""
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(x - x.mean())) ** 2 / n
    if f_split is None:
        f_split = fs / 4.0  # split at Nyquist/2
    lo_mask = (freqs > 0) & (freqs <= f_split)
    hi_mask = freqs > f_split
    p_lo = np.sum(psd[lo_mask]) if lo_mask.any() else 1e-15
    p_hi = np.sum(psd[hi_mask]) if hi_mask.any() else 1e-15
    return p_lo / max(p_hi, 1e-15)


def sliding_window_ews(signal, fs, window_s, step_s, t_start_s, t_end_s, t_signal):
    """Compute all EWS indicators in sliding windows.

    Returns
    -------
    t_centers : (n_windows,) center times of each window
    results : dict of indicator name -> (n_windows,) values
    """
    win_samples = int(window_s * fs)
    step_samples = int(step_s * fs)

    t_centers = []
    results = {
        "variance": [], "autocorrelation": [], "skewness": [],
        "dfa": [], "spectral_index": [],
    }

    # Iterate windows
    start_idx = np.searchsorted(t_signal, t_start_s)
    end_idx = np.searchsorted(t_signal, t_end_s)

    pos = start_idx
    while pos + win_samples <= end_idx:
        chunk = signal[pos:pos + win_samples]
        t_center = t_signal[pos] + window_s / 2.0
        t_centers.append(t_center)

        results["variance"].append(compute_variance(chunk))
        results["autocorrelation"].append(compute_lag1_autocorrelation(chunk))
        results["skewness"].append(compute_skewness(chunk))
        results["dfa"].append(compute_dfa_exponent(chunk))
        results["spectral_index"].append(compute_spectral_index(chunk, fs))

        pos += step_samples

    t_centers = np.array(t_centers)
    for k in results:
        results[k] = np.array(results[k])
    return t_centers, results


def kendall_tau(t_centers, values):
    """Kendall tau correlation between time index and indicator values."""
    if len(values) < 3:
        return 0.0, 1.0
    tau, pval = stats.kendalltau(np.arange(len(values)), values)
    return tau, pval


# ============================================================
# Stuart-Landau fitting
# ============================================================
def fit_stuart_landau_window(A, dAdt):
    """OLS fit of dA/dt = mu*A - alpha*A^3.

    Returns mu, alpha, r2.
    """
    if len(A) < 5:
        return 0.0, 0.0, 0.0

    # Design matrix: [A, A^3]
    X_design = np.column_stack([A, A ** 3])
    # Solve: dAdt = X @ [mu, -alpha]
    try:
        coeffs, residuals, rank, sv = np.linalg.lstsq(X_design, dAdt, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0

    mu = coeffs[0]
    alpha = -coeffs[1]  # dA/dt = mu*A - alpha*A^3 => coeff of A^3 is -alpha

    pred = X_design @ coeffs
    ss_res = np.sum((dAdt - pred) ** 2)
    ss_tot = np.sum((dAdt - dAdt.mean()) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-15)

    return mu, alpha, r2


def sliding_stuart_landau(amplitude, derivative, t_signal, window_s, step_s,
                           t_start_s, t_end_s):
    """Fit Stuart-Landau in sliding windows.

    Returns t_centers, mu_vals, alpha_vals, r2_vals.
    """
    fs = 1.0 / (t_signal[1] - t_signal[0])
    win_samples = int(window_s * fs)
    step_samples = int(step_s * fs)

    t_centers, mu_vals, alpha_vals, r2_vals = [], [], [], []

    start_idx = np.searchsorted(t_signal, t_start_s)
    end_idx = np.searchsorted(t_signal, t_end_s)

    pos = start_idx
    while pos + win_samples <= end_idx:
        A_chunk = amplitude[pos:pos + win_samples]
        dA_chunk = derivative[pos:pos + win_samples]
        t_center = t_signal[pos] + window_s / 2.0

        mu, alpha, r2 = fit_stuart_landau_window(A_chunk, dA_chunk)

        t_centers.append(t_center)
        mu_vals.append(mu)
        alpha_vals.append(alpha)
        r2_vals.append(r2)

        pos += step_samples

    return (np.array(t_centers), np.array(mu_vals),
            np.array(alpha_vals), np.array(r2_vals))


# ============================================================
# Main
# ============================================================
def main():
    setup_style()

    print("=" * 70)
    print("ANALYSIS 3: EARLY WARNING SIGNALS + STUART-LANDAU FITTING")
    print("=" * 70)

    # --- Load data ---
    episodes = load_episodes()

    # --- Extract alpha amplitude envelopes per cluster, downsampled to 5 Hz ---
    print("\nExtracting alpha-band amplitude envelopes...")
    # amp_data[ep][cid] = (T_ds,) amplitude mode 0
    amp_data = {}
    t_data = {}
    for ep in [1, 2, 3]:
        modes = extract_cluster_amplitudes(episodes[ep], band=ALPHA_BAND)
        amp_data[ep] = {cid: modes[cid][::DS] for cid in CLUSTER_IDS}
        n_ds = len(amp_data[ep][CLUSTER_IDS[0]])
        t_data[ep] = np.arange(n_ds) * DT_DS
        print(f"  Episode {ep}: {n_ds} samples at {FS_DS:.0f} Hz, "
              f"T=[0, {t_data[ep][-1]:.1f}]s")

    # --- Compute derivatives for Stuart-Landau ---
    print("\nComputing Savitzky-Golay derivatives...")
    deriv_data = {}
    for ep in [1, 2, 3]:
        deriv_data[ep] = {}
        for cid in CLUSTER_IDS:
            deriv_data[ep][cid] = savgol_derivative(amp_data[ep][cid], DT_DS)

    # ================================================================
    # EWS: sliding window analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("EWS SLIDING-WINDOW ANALYSIS")
    print(f"  Window: {EWS_WINDOW_S}s, Step: {EWS_STEP_S}s")
    print("=" * 70)

    # ews_results[ep][cid] = (t_centers, results_dict)
    ews_results = {}
    for ep in [1, 2, 3]:
        ews_results[ep] = {}
        onset = ONSET_TIMES[ep]
        t_start = onset - EWS_END_OFFSET_S
        t_end = onset + 10.0  # include 10s into seizure
        if t_start < t_data[ep][0] + EWS_WINDOW_S:
            t_start = t_data[ep][0] + EWS_WINDOW_S

        for cid in CLUSTER_IDS:
            t_c, res = sliding_window_ews(
                amp_data[ep][cid], FS_DS, EWS_WINDOW_S, EWS_STEP_S,
                t_start, t_end, t_data[ep]
            )
            ews_results[ep][cid] = (t_c, res)
            print(f"  Ep{ep} {CLUSTER_NAMES[cid]}: {len(t_c)} windows")

    # ================================================================
    # Kendall tau trend tests
    # ================================================================
    print("\n" + "=" * 70)
    print("KENDALL TAU TREND TESTS (pre-seizure only)")
    print("=" * 70)

    indicators = ["variance", "autocorrelation", "skewness", "dfa", "spectral_index"]
    indicator_labels = {
        "variance": "Variance", "autocorrelation": "Lag-1 AC",
        "skewness": "Skewness", "dfa": "DFA exponent",
        "spectral_index": "Spectral Index",
    }

    # kendall[indicator][cid] = list of (tau, pval) per episode
    kendall = {ind: {cid: [] for cid in CLUSTER_IDS} for ind in indicators}
    for ep in [1, 2, 3]:
        onset = ONSET_TIMES[ep]
        for cid in CLUSTER_IDS:
            t_c, res = ews_results[ep][cid]
            pre_mask = t_c < onset
            for ind in indicators:
                vals_pre = res[ind][pre_mask]
                tau, pval = kendall_tau(t_c[pre_mask], vals_pre)
                kendall[ind][cid].append((tau, pval))

    # Print summary
    for ind in indicators:
        print(f"\n  {indicator_labels[ind]}:")
        for cid in CLUSTER_IDS:
            taus = [k[0] for k in kendall[ind][cid]]
            pvals = [k[1] for k in kendall[ind][cid]]
            sig = sum(1 for p in pvals if p < 0.05)
            print(f"    {CLUSTER_NAMES[cid]:20s}: tau={np.mean(taus):.3f} "
                  f"(range [{min(taus):.3f}, {max(taus):.3f}]), "
                  f"sig in {sig}/3 episodes")

    # ================================================================
    # Stuart-Landau sliding fit
    # ================================================================
    print("\n" + "=" * 70)
    print("STUART-LANDAU SLIDING FIT")
    print(f"  dA/dt = mu*A - alpha*A^3")
    print(f"  Window: {SL_WINDOW_S}s, Step: {SL_STEP_S}s")
    print("=" * 70)

    # sl_results[ep][cid] = (t_c, mu, alpha, r2)
    sl_results = {}
    for ep in [1, 2, 3]:
        sl_results[ep] = {}
        onset = ONSET_TIMES[ep]
        t_start = max(onset - 40.0, t_data[ep][0] + SL_WINDOW_S)
        t_end = min(onset + 15.0, t_data[ep][-1] - SL_WINDOW_S)

        for cid in CLUSTER_IDS:
            t_c, mu, alpha, r2 = sliding_stuart_landau(
                amp_data[ep][cid], deriv_data[ep][cid], t_data[ep],
                SL_WINDOW_S, SL_STEP_S, t_start, t_end
            )
            sl_results[ep][cid] = (t_c, mu, alpha, r2)

            # Find mu zero-crossing
            if len(mu) > 1:
                crossings = np.where(np.diff(np.sign(mu)))[0]
                if len(crossings) > 0:
                    t_cross = t_c[crossings[0]]
                    print(f"  Ep{ep} {CLUSTER_NAMES[cid]}: mu crosses zero at "
                          f"t={t_cross:.1f}s (onset={onset:.2f}s, "
                          f"delta={t_cross-onset:.1f}s)")
                else:
                    print(f"  Ep{ep} {CLUSTER_NAMES[cid]}: mu does NOT cross zero")
            else:
                print(f"  Ep{ep} {CLUSTER_NAMES[cid]}: insufficient data")

    # ================================================================
    # PLOTS
    # ================================================================
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # --- Plot 1-5: Individual EWS indicators ---
    for ind in indicators:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        fig.suptitle(f"EWS: {indicator_labels[ind]}", fontsize=13)

        for ax_idx, ep in enumerate([1, 2, 3]):
            ax = axes[ax_idx]
            onset = ONSET_TIMES[ep]
            for cid in CLUSTER_IDS:
                t_c, res = ews_results[ep][cid]
                ax.plot(t_c, res[ind], color=CLUSTER_COLORS[cid],
                        lw=1.2, label=CLUSTER_NAMES[cid])
            ax.axvline(onset, color="k", ls="--", lw=0.8, alpha=0.7, label="Onset")
            ax.set_ylabel(indicator_labels[ind])
            ax.set_title(f"Episode {ep}", fontsize=10)
            if ax_idx == 0:
                ax.legend(fontsize=7, loc="upper left")
            ax.set_xlabel("Time (s)")

        fig.tight_layout()
        save_fig(fig, OUT_DIR / f"ews_{ind}")

    # --- Plot 6: Kendall tau summary bar chart ---
    fig, axes = plt.subplots(1, len(indicators), figsize=(14, 4), sharey=True)
    fig.suptitle("Kendall Tau Trend Test (pre-seizure)", fontsize=13)

    x_pos = np.arange(len(CLUSTER_IDS))
    width = 0.25

    for ax_idx, ind in enumerate(indicators):
        ax = axes[ax_idx]
        for ep_idx, ep in enumerate([1, 2, 3]):
            taus = [kendall[ind][cid][ep_idx][0] for cid in CLUSTER_IDS]
            pvals = [kendall[ind][cid][ep_idx][1] for cid in CLUSTER_IDS]
            bars = ax.bar(x_pos + ep_idx * width, taus, width,
                         label=f"Ep{ep}", alpha=0.8)
            # Significance markers
            for i, (bar, pv) in enumerate(zip(bars, pvals)):
                if pv < 0.01:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           "**", ha="center", va="bottom", fontsize=8)
                elif pv < 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           "*", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_pos + width)
        ax.set_xticklabels([CLUSTER_NAMES[c].split(" ")[0] for c in CLUSTER_IDS],
                          fontsize=8)
        ax.set_title(indicator_labels[ind], fontsize=9)
        ax.axhline(0, color="gray", ls="-", lw=0.5)
        if ax_idx == 0:
            ax.set_ylabel("Kendall tau")
            ax.legend(fontsize=7)

    fig.tight_layout()
    save_fig(fig, OUT_DIR / "ews_kendall_summary")

    # --- Plot 7-8: Stuart-Landau mu(t) and R^2 ---
    for var_name, var_idx, ylabel in [("mu", 1, r"$\mu(t)$"), ("r2", 3, r"$R^2$")]:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
        fig.suptitle(f"Stuart-Landau: {ylabel}", fontsize=13)

        for ax_idx, ep in enumerate([1, 2, 3]):
            ax = axes[ax_idx]
            onset = ONSET_TIMES[ep]
            for cid in CLUSTER_IDS:
                t_c, mu, alpha, r2 = sl_results[ep][cid]
                data = [mu, alpha, r2][var_idx - 1] if var_idx != 3 else r2
                if var_name == "mu":
                    data = mu
                ax.plot(t_c, data, color=CLUSTER_COLORS[cid],
                        lw=1.2, label=CLUSTER_NAMES[cid])
            ax.axvline(onset, color="k", ls="--", lw=0.8, alpha=0.7, label="Onset")
            if var_name == "mu":
                ax.axhline(0, color="gray", ls=":", lw=0.5)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Episode {ep}", fontsize=10)
            if ax_idx == 0:
                ax.legend(fontsize=7, loc="upper left")
            ax.set_xlabel("Time (s)")

        fig.tight_layout()
        save_fig(fig, OUT_DIR / f"stuart_landau_{var_name}")

    # --- Plot 9-11: Phase portraits (A, dA/dt) for pre-seizure, transition, seizure ---
    for epoch_name, offset_before, offset_after in [
        ("presez", -30, -5), ("transition", -5, 5), ("seizure", 5, 15)
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"Phase Portrait: {epoch_name.replace('_',' ').title()}", fontsize=13)

        for ax_idx, ep in enumerate([1, 2, 3]):
            ax = axes[ax_idx]
            onset = ONSET_TIMES[ep]
            t = t_data[ep]

            for cid in CLUSTER_IDS:
                A = amp_data[ep][cid]
                dA = deriv_data[ep][cid]
                mask = (t >= onset + offset_before) & (t < onset + offset_after)
                if mask.sum() < 2:
                    continue
                sc = ax.scatter(A[mask], dA[mask], c=t[mask], cmap="viridis",
                               s=3, alpha=0.6, label=CLUSTER_NAMES[cid])

            ax.set_xlabel("Amplitude A")
            ax.set_ylabel("dA/dt")
            ax.set_title(f"Episode {ep}", fontsize=10)
            if ax_idx == 0:
                ax.legend(fontsize=7, markerscale=3)

        fig.tight_layout()
        save_fig(fig, OUT_DIR / f"phase_portrait_{epoch_name}")

    # --- Plot 12: Bifurcation summary composite ---
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

    # Row 1: Variance for each episode
    for ep_idx, ep in enumerate([1, 2, 3]):
        ax = fig.add_subplot(gs[0, ep_idx])
        onset = ONSET_TIMES[ep]
        for cid in CLUSTER_IDS:
            t_c, res = ews_results[ep][cid]
            ax.plot(t_c, res["variance"], color=CLUSTER_COLORS[cid],
                    lw=1.0, label=CLUSTER_NAMES[cid])
        ax.axvline(onset, color="k", ls="--", lw=0.8)
        ax.set_title(f"Ep{ep}: Variance", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        if ep_idx == 0:
            ax.legend(fontsize=6)

    # Row 2: mu(t) for each episode
    for ep_idx, ep in enumerate([1, 2, 3]):
        ax = fig.add_subplot(gs[1, ep_idx])
        onset = ONSET_TIMES[ep]
        for cid in CLUSTER_IDS:
            t_c, mu, alpha, r2 = sl_results[ep][cid]
            ax.plot(t_c, mu, color=CLUSTER_COLORS[cid], lw=1.0,
                    label=CLUSTER_NAMES[cid])
        ax.axvline(onset, color="k", ls="--", lw=0.8)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_title(f"Ep{ep}: Stuart-Landau mu(t)", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        if ep_idx == 0:
            ax.legend(fontsize=6)

    # Row 3: Phase portrait (transition window) for each episode
    for ep_idx, ep in enumerate([1, 2, 3]):
        ax = fig.add_subplot(gs[2, ep_idx])
        onset = ONSET_TIMES[ep]
        t = t_data[ep]
        for cid in CLUSTER_IDS:
            A = amp_data[ep][cid]
            dA = deriv_data[ep][cid]
            mask = (t >= onset - 5) & (t < onset + 5)
            if mask.sum() > 0:
                ax.scatter(A[mask], dA[mask], c=CLUSTER_COLORS[cid],
                          s=5, alpha=0.5, label=CLUSTER_NAMES[cid])
        ax.set_xlabel("A", fontsize=8)
        ax.set_ylabel("dA/dt", fontsize=8)
        ax.set_title(f"Ep{ep}: Phase portrait (transition)", fontsize=9)
        if ep_idx == 0:
            ax.legend(fontsize=6, markerscale=2)

    fig.suptitle("Bifurcation Summary", fontsize=14, y=0.98)
    save_fig(fig, OUT_DIR / "bifurcation_summary")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Determine which cluster shows earliest CSD
    for cid in CLUSTER_IDS:
        var_taus = [kendall["variance"][cid][ep][0] for ep in range(3)]
        ac_taus = [kendall["autocorrelation"][cid][ep][0] for ep in range(3)]
        skew_vals = []
        for ep in [1, 2, 3]:
            t_c, res = ews_results[ep][cid]
            pre_mask = t_c < ONSET_TIMES[ep]
            if pre_mask.sum() > 0:
                skew_vals.append(np.mean(np.abs(res["skewness"][pre_mask])))
        print(f"\n  {CLUSTER_NAMES[cid]}:")
        print(f"    Variance trend (tau):  {np.mean(var_taus):.3f}")
        print(f"    AC trend (tau):        {np.mean(ac_taus):.3f}")
        print(f"    Mean |skewness|:       {np.mean(skew_vals):.3f}")

    # Stuart-Landau zero crossings
    print("\n  Stuart-Landau mu zero-crossings:")
    for ep in [1, 2, 3]:
        onset = ONSET_TIMES[ep]
        for cid in CLUSTER_IDS:
            t_c, mu, _, _ = sl_results[ep][cid]
            crossings = np.where(np.diff(np.sign(mu)))[0]
            if len(crossings) > 0:
                t_cross = t_c[crossings[0]]
                print(f"    Ep{ep} {CLUSTER_NAMES[cid]}: t={t_cross:.1f}s "
                      f"(onset={onset:.2f}s, delta={t_cross-onset:+.1f}s)")

    print(f"\n  All plots saved to {OUT_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
