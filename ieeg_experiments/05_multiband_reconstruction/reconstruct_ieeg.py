#!/usr/bin/env python3
"""Reconstruct iEEG signals from the multiband Kuramoto model.

Pipeline:
  1. Load real data, bandpass into 5 bands, extract per-channel amplitudes
     and phase offsets relative to cluster mean
  2. Simulate multiband Kuramoto with coherence ramp (seizure onset)
  3. Reconstruct: x_ch(t) = Σ_b A_ch_b(t) * cos(θ_cluster(t) + δφ_ch)
     - Amplitudes from DATA (captures seizure amplitude increase)
     - Phases from MODEL (captures frequency/coupling dynamics)
  4. Compare real vs reconstructed iEEG, showing seizure emergence

Author: KANDy Researcher Agent
Date: 2026-03-31
"""

import numpy as np
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
from scipy.integrate import solve_ivp
from sklearn.cluster import SpectralClustering
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

OUT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "QM" / "PatientQM_02Clean.txt"

FS = 500
ONSET_TIME = 88.25
SEED = 42
np.random.seed(SEED)

BANDS = {
    "delta":     (1.0,  4.0),
    "theta":     (4.0,  8.0),
    "alpha":     (8.0, 13.0),
    "beta":      (13.0, 30.0),
    "low_gamma": (30.0, 50.0),
}
BAND_NAMES = list(BANDS.keys())
N_BANDS = len(BANDS)

# Clustering params
SG_WINDOW = 51
SG_POLY = 5
CLUSTER_K = 4
SOZ_CHANNELS = list(range(21, 31))

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Multiband Kuramoto RHS (same as parameter_sweep.py)
# ============================================================
N_CL = 3
N_STATE = N_BANDS * N_CL


def multiband_rhs(t, theta, r0, r2, r3):
    r0_val = r0(t) if callable(r0) else r0
    r2_val = r2(t) if callable(r2) else r2
    r3_val = r3(t) if callable(r3) else r3

    dtheta = np.zeros(N_STATE)

    # DELTA
    dtheta[0] = 13.28 + 28.16 * r0_val + 9.92 * r3_val - 20.02
    dtheta[1] = 14.85 - 5.12 * r2_val + 25.88 * r3_val - 14.04
    dtheta[2] = 13.17 - 0.12

    # THETA
    dtheta[3] = 34.70 - 16.94 * r0_val + 7.34
    dtheta[4] = 35.75 - 0.51
    dtheta[5] = 35.00 - 0.32

    # ALPHA
    th_a = theta[6:9]
    th_a_mean = th_a.mean()
    dtheta[6] = 63.68 + 2.14 * r0_val * np.sin(th_a[0] - th_a_mean) \
                - 21.17 * r3_val + 14.36
    dtheta[7] = 63.42 + 12.10 * r0_val - 18.38 * r3_val + 7.60
    dtheta[8] = 63.88 - 0.43

    # BETA
    dtheta[9] = 109.99 + 41.79 * r3_val - 29.68
    dtheta[10] = 112.09 + 38.92 * r3_val - 26.91
    dtheta[11] = 111.38 - 1.33

    # LOW GAMMA
    th_g = theta[12:15]
    th_g_mean = th_g.mean()
    dtheta[12] = 233.85 + 3.53 * r2_val * np.sin(th_g[1] - th_g_mean) + 0.84
    dtheta[13] = 230.19 + 41.35 * r0_val - 17.11
    dtheta[14] = 229.65 - 0.22

    return dtheta


# ============================================================
# Load data + clustering
# ============================================================
print("=" * 70)
print("RECONSTRUCT iEEG FROM MULTIBAND KURAMOTO MODEL")
print("=" * 70)

print("\nLoading data...")
X = np.loadtxt(DATA_PATH)
n_samples, n_ch = X.shape
t_data = np.arange(n_samples) / FS
ONSET_SAMPLE = int(ONSET_TIME * FS)
print(f"  Data: {X.shape}, {n_samples/FS:.1f}s")

# Clustering
print("\nClustering...")


def compute_embedding(data_segment, fs):
    T, C = data_segment.shape
    dt = 1.0 / fs
    embeddings = {}
    for ch in range(C):
        x = data_segment[:, ch].copy()
        mu, sigma = x.mean(), x.std()
        if sigma < 1e-12:
            sigma = 1.0
        x = (x - mu) / sigma
        x_s = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=0)
        x_d = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=1, delta=dt)
        x_dd = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=2, delta=dt)
        embeddings[ch] = np.column_stack([x_s, x_d, x_dd])
    return embeddings


sez_s = ONSET_SAMPLE
sez_e = min(n_samples, sez_s + int(10 * FS))
embs = compute_embedding(X[sez_s:sez_e], FS)
covs = {ch: np.cov(emb, rowvar=False) + 1e-10 * np.eye(3)
        for ch, emb in embs.items()}

D = np.zeros((n_ch, n_ch))
for i in range(n_ch):
    for j in range(i + 1, n_ch):
        d = np.linalg.norm(covs[i] - covs[j], "fro")
        D[i, j] = d
        D[j, i] = d

sigma_aff = np.median(D[D > 0])
affinity = np.exp(-D ** 2 / (2 * sigma_aff ** 2))
sc = SpectralClustering(n_clusters=CLUSTER_K, affinity="precomputed",
                        random_state=SEED, assign_labels="kmeans")
labels = sc.fit_predict(affinity)

soz_cluster_orig = np.bincount(labels[SOZ_CHANNELS]).argmax()
cluster_map = {soz_cluster_orig: 0}
next_label = 1
for c in range(CLUSTER_K):
    if c != soz_cluster_orig:
        cluster_map[c] = next_label
        next_label += 1
labels = np.array([cluster_map[l] for l in labels])

clusters_all = {c: np.where(labels == c)[0].tolist() for c in range(CLUSTER_K)}
bulk_id = max(clusters_all, key=lambda c: len(clusters_all[c]))
KEEP = [c for c in sorted(clusters_all.keys()) if c != bulk_id]
clusters = {c: clusters_all[c] for c in KEEP}
CL_NAMES = {0: "C0 (SOZ)", 1: "C1 (bulk)", 2: "C2 (pace)", 3: "C3 (bound)"}

# Channel-to-cluster mapping (index into KEEP array: 0, 1, 2)
ch_to_cl_idx = np.zeros(n_ch, dtype=int)
for ci, c in enumerate(KEEP):
    for ch in clusters[c]:
        ch_to_cl_idx[ch] = ci
# Bulk channels get mapped to nearest SOZ cluster (C0 by default)
for ch in clusters_all[bulk_id]:
    ch_to_cl_idx[ch] = 0

print(f"  Clusters: {[(CL_NAMES[c], len(clusters[c])) for c in KEEP]}")
print(f"  Bulk ({len(clusters_all[bulk_id])} ch) mapped to C0 for reconstruction")


# ============================================================
# Extract per-channel, per-band amplitudes and phase offsets
# ============================================================
print("\nExtracting per-channel amplitudes and phase offsets...")

nyq = FS / 2.0
# amp_env[band][ch] = (n_samples,) amplitude envelope
# phase_offset[band][ch] = scalar phase offset relative to cluster mean
amp_env = {}
phase_offset = {}
cluster_phase = {}  # cluster mean phase for reference

for bi, (band_name, (blo, bhi)) in enumerate(BANDS.items()):
    b, a = butter(4, [blo / nyq, min(bhi / nyq, 0.999)], btype="band")
    X_filt = filtfilt(b, a, X, axis=0)

    amp_env[band_name] = np.zeros((n_samples, n_ch))
    phase_offset[band_name] = np.zeros(n_ch)

    # Per-channel analytic signal
    ch_phases = np.zeros((n_samples, n_ch))
    for ch in range(n_ch):
        analytic = hilbert(X_filt[:, ch])
        amp_env[band_name][:, ch] = np.abs(analytic)
        ch_phases[:, ch] = np.angle(analytic)

    # Cluster mean phases
    cl_phases = np.zeros((n_samples, N_CL))
    for ci, c in enumerate(KEEP):
        z = np.exp(1j * ch_phases[:, clusters[c]]).mean(axis=1)
        cl_phases[:, ci] = np.angle(z)
    cluster_phase[band_name] = cl_phases

    # Phase offset: mean phase difference of each channel from its cluster
    for ch in range(n_ch):
        ci = ch_to_cl_idx[ch]
        diff = ch_phases[:, ch] - cl_phases[:, ci]
        # Circular mean of the difference
        phase_offset[band_name][ch] = np.angle(np.mean(np.exp(1j * diff)))

print("  Done.")


# ============================================================
# Smooth amplitude envelopes (for reconstruction)
# ============================================================
print("\nSmoothing amplitude envelopes...")
SMOOTH_WIN = int(0.5 * FS)  # 0.5s running average
kernel = np.ones(SMOOTH_WIN) / SMOOTH_WIN

amp_smooth = {}
for band_name in BAND_NAMES:
    amp_smooth[band_name] = np.zeros((n_samples, n_ch))
    for ch in range(n_ch):
        amp_smooth[band_name][:, ch] = np.convolve(
            amp_env[band_name][:, ch], kernel, mode="same"
        )


# ============================================================
# Simulate multiband Kuramoto (matching data time axis)
# ============================================================
print("\nSimulating multiband Kuramoto...")

# Coherence ramp at seizure onset
RAMP_DUR = 3.0
R0_PRE, R0_SEZ = 0.44, 0.75
R2_PRE, R2_SEZ = 0.86, 0.95
R3_PRE, R3_SEZ = 0.72, 0.95


def ramp(t, pre, sez):
    if t < ONSET_TIME:
        return pre
    elif t < ONSET_TIME + RAMP_DUR:
        frac = (t - ONSET_TIME) / RAMP_DUR
        return pre + frac * (sez - pre)
    else:
        return sez


r0_fn = lambda t: ramp(t, R0_PRE, R0_SEZ)
r2_fn = lambda t: ramp(t, R2_PRE, R2_SEZ)
r3_fn = lambda t: ramp(t, R3_PRE, R3_SEZ)

# Match initial phases to data at t=0
theta0 = np.zeros(N_STATE)
for bi, band_name in enumerate(BAND_NAMES):
    for ci in range(N_CL):
        theta0[bi * N_CL + ci] = cluster_phase[band_name][0, ci]

T_END = n_samples / FS
dt_sim = 1.0 / FS  # simulate at full 500 Hz

t_eval = np.arange(0, T_END, dt_sim)
print(f"  Simulating {len(t_eval)} timesteps ({T_END:.1f}s at {FS} Hz)...")

sol = solve_ivp(
    lambda t, y: multiband_rhs(t, y, r0_fn, r2_fn, r3_fn),
    [0, T_END], theta0, t_eval=t_eval,
    method="RK45", rtol=1e-8, atol=1e-10,
)
t_sim = sol.t
theta_sim = sol.y.T  # (T, 15)
n_sim = len(t_sim)
print(f"  Simulated: {theta_sim.shape}")


# ============================================================
# Reconstruct iEEG signals
# ============================================================
print("\nReconstructing iEEG signals...")

# x_ch(t) = Σ_b A_ch_b(t) * cos(θ_cluster(t) + δφ_ch_b)
n_recon = min(n_sim, n_samples)
X_recon = np.zeros((n_recon, n_ch))

for bi, band_name in enumerate(BAND_NAMES):
    for ch in range(n_ch):
        ci = ch_to_cl_idx[ch]
        phase_sim = theta_sim[:n_recon, bi * N_CL + ci]
        delta_phi = phase_offset[band_name][ch]
        amp = amp_smooth[band_name][:n_recon, ch]
        X_recon[:, ch] += amp * np.cos(phase_sim + delta_phi)

print(f"  Reconstructed: {X_recon.shape}")


# ============================================================
# PLOTS
# ============================================================
print("\nGenerating plots...")

t_plot = t_data[:n_recon]

# --- Plot 1: SOZ channels comparison (real vs reconstructed) ---
soz_ch = SOZ_CHANNELS[:6]  # show 6 SOZ channels
T_PLOT_START = ONSET_TIME - 15  # 15s before to 10s after
T_PLOT_END = ONSET_TIME + 10
mask = (t_plot >= T_PLOT_START) & (t_plot <= T_PLOT_END)

fig, axes = plt.subplots(len(soz_ch), 2, figsize=(16, 2.5 * len(soz_ch)),
                         sharex=True)
fig.suptitle("iEEG Reconstruction: Real (left) vs Model (right)\n"
             "SOZ Channels, Seizure Window", fontsize=14)

for i, ch in enumerate(soz_ch):
    # Real
    ax = axes[i, 0]
    ax.plot(t_plot[mask], X[mask, ch], color="k", lw=0.4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel(f"Ch {ch}", fontsize=9)
    if i == 0:
        ax.set_title("Real iEEG", fontsize=11)

    # Reconstructed
    ax = axes[i, 1]
    ax.plot(t_plot[mask], X_recon[mask, ch], color="steelblue", lw=0.4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8, alpha=0.6)
    if i == 0:
        ax.set_title("Multiband Kuramoto Model", fontsize=11)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "reconstruct_soz_channels")

# --- Plot 2: Full array overview (waterfall) ---
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("Full iEEG Array: Real vs Reconstructed (seizure window)",
             fontsize=14)

spacing = np.percentile(np.abs(X[mask]), 90) * 0.4
ch_show = list(range(0, n_ch, max(1, n_ch // 40)))  # ~40 channels

for ax_idx, (data, title, color) in enumerate([
    (X, "Real iEEG", "k"),
    (X_recon, "Multiband Kuramoto Model", "steelblue"),
]):
    ax = axes[ax_idx]
    for ci_plot, ch in enumerate(ch_show):
        sig = data[mask, ch]
        # Normalize per channel for display
        sig_norm = sig / max(np.std(sig), 1e-6) * spacing * 0.3
        ax.plot(t_plot[mask], sig_norm + ci_plot * spacing,
                color=color, lw=0.3, alpha=0.7)
        # Mark SOZ channels
        if ch in SOZ_CHANNELS:
            ax.plot(t_plot[mask][0] - 0.3,
                    ci_plot * spacing, "r>", ms=4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title, fontsize=12)
    ax.set_yticks([ci_plot * spacing for ci_plot, ch in enumerate(ch_show[::5])])
    ax.set_yticklabels([f"ch{ch}" for ch in ch_show[::5]], fontsize=7)

fig.tight_layout()
save_fig(fig, "reconstruct_full_array")

# --- Plot 3: Spectrograms comparison (SOZ channel) ---
ch_spec = 25  # SOZ channel
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"Spectrogram: Channel {ch_spec} (SOZ)", fontsize=14)

for ax_idx, (data, title) in enumerate([
    (X, "Real iEEG"),
    (X_recon, "Multiband Kuramoto Model"),
]):
    ax = axes[ax_idx]
    spec_data = data[mask, ch_spec]
    ax.specgram(spec_data, NFFT=256, Fs=FS, noverlap=200, cmap="viridis")
    ax.set_ylim(0, 60)
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=11)
    # Add onset marker
    onset_rel = ONSET_TIME - T_PLOT_START
    ax.axvline(onset_rel, color="red", ls="--", lw=1, alpha=0.7)

axes[-1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "reconstruct_spectrogram")

# --- Plot 4: Per-band reconstruction (single SOZ channel) ---
fig, axes = plt.subplots(N_BANDS + 1, 2, figsize=(16, 2.5 * (N_BANDS + 1)),
                         sharex=True)
fig.suptitle(f"Per-Band Reconstruction: Channel {ch_spec} (SOZ)\n"
             "Left: Real bandpassed, Right: Model", fontsize=14)

for bi, band_name in enumerate(BAND_NAMES):
    blo, bhi = BANDS[band_name]
    b, a = butter(4, [blo / nyq, min(bhi / nyq, 0.999)], btype="band")

    # Real bandpassed
    real_band = filtfilt(b, a, X[:n_recon, ch_spec])
    ax = axes[bi, 0]
    ax.plot(t_plot[mask], real_band[mask], color="k", lw=0.4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylabel(f"{band_name}\n({blo:.0f}-{bhi:.0f} Hz)", fontsize=8)
    if bi == 0:
        ax.set_title("Real (bandpassed)", fontsize=11)

    # Model single-band contribution
    ci = ch_to_cl_idx[ch_spec]
    phase_sim_band = theta_sim[:n_recon, bi * N_CL + ci]
    delta_phi = phase_offset[band_name][ch_spec]
    amp = amp_smooth[band_name][:n_recon, ch_spec]
    model_band = amp * np.cos(phase_sim_band + delta_phi)

    ax = axes[bi, 1]
    ax.plot(t_plot[mask], model_band[mask], color="steelblue", lw=0.4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8, alpha=0.6)
    if bi == 0:
        ax.set_title("Model (per-band)", fontsize=11)

# Bottom row: broadband sum
ax = axes[N_BANDS, 0]
ax.plot(t_plot[mask], X[mask, ch_spec], color="k", lw=0.4)
ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8, alpha=0.6)
ax.set_ylabel("Broadband\nsum", fontsize=8)
ax.set_xlabel("Time (s)")
ax.set_title("Real (raw)", fontsize=9)

ax = axes[N_BANDS, 1]
ax.plot(t_plot[mask], X_recon[mask, ch_spec], color="steelblue", lw=0.4)
ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8, alpha=0.6)
ax.set_xlabel("Time (s)")
ax.set_title("Model (all bands summed)", fontsize=9)

fig.tight_layout()
save_fig(fig, "reconstruct_per_band")

# --- Plot 5: Seizure onset zoom (2s before to 5s after) ---
T_ZOOM_START = ONSET_TIME - 2
T_ZOOM_END = ONSET_TIME + 5
zoom_mask = (t_plot >= T_ZOOM_START) & (t_plot <= T_ZOOM_END)

fig, axes = plt.subplots(4, 2, figsize=(16, 10), sharex=True)
fig.suptitle("Seizure Onset Zoom: Real vs Model\n"
             "4 SOZ channels, 2s before to 5s after onset", fontsize=14)

zoom_chs = [24, 25, 28, 30]
for i, ch in enumerate(zoom_chs):
    ax = axes[i, 0]
    ax.plot(t_plot[zoom_mask], X[zoom_mask, ch], color="k", lw=0.5)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=1)
    ax.set_ylabel(f"Ch {ch}", fontsize=9)
    if i == 0:
        ax.set_title("Real iEEG", fontsize=11)

    ax = axes[i, 1]
    ax.plot(t_plot[zoom_mask], X_recon[zoom_mask, ch], color="steelblue", lw=0.5)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=1)
    if i == 0:
        ax.set_title("Multiband Kuramoto Model", fontsize=11)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "reconstruct_seizure_zoom")


# ============================================================
# Summary
# ============================================================
# Correlation between real and reconstructed
corrs = []
for ch in SOZ_CHANNELS:
    c = np.corrcoef(X[mask, ch], X_recon[mask, ch])[0, 1]
    corrs.append(c)
print(f"\n  SOZ channel correlations (real vs model): "
      f"mean={np.mean(corrs):.3f}, range=[{min(corrs):.3f}, {max(corrs):.3f}]")

print(f"\nAll plots saved to {OUT_DIR}/")
print("Done.")
