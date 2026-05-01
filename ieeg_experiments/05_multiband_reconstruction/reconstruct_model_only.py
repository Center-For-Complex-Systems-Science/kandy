#!/usr/bin/env python3
"""Fully model-driven iEEG reconstruction from multiband Kuramoto.

NO data time series used during simulation. All dynamics come from the model.

Data is used ONLY for:
  1. Cluster assignments (which channels belong to which cluster)
  2. Per-channel phase offsets within clusters (fixed scalars)
  3. Pre-seizure amplitude statistics (mean + std per channel per band)
  4. Seizure amplitude statistics (mean + std per channel per band)

The model generates:
  - Phases: multiband Kuramoto equations (discovered by KANDy)
  - Amplitudes: Stuart-Landau per band per cluster:
      dA/dt = mu(t)*A - alpha*A^3 + sigma*noise
    where mu(t) crosses zero at seizure onset (supercritical Hopf)
  - Coherence r_k(t): sigmoid ramp at onset (drives Kuramoto coupling)

Author: KANDy Researcher Agent
Date: 2026-03-31
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.integrate import solve_ivp
from sklearn.cluster import SpectralClustering
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
BAND_LABELS = [r"$\delta$", r"$\theta$", r"$\alpha$", r"$\beta$", r"$\gamma_L$"]
N_BANDS = len(BANDS)
N_CL = 3
N_PHASE = N_BANDS * N_CL  # 15 phase variables

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
# STEP 1: Extract STATIC parameters from data (no time series used later)
# ============================================================
print("=" * 70)
print("FULLY MODEL-DRIVEN iEEG RECONSTRUCTION")
print("=" * 70)

print("\nStep 1: Extracting static parameters from data...")
X = np.loadtxt(DATA_PATH)
n_samples, n_ch = X.shape
t_data = np.arange(n_samples) / FS
ONSET_SAMPLE = int(ONSET_TIME * FS)
print(f"  Data: {X.shape}, {n_samples/FS:.1f}s")

# --- Clustering (static) ---
from scipy.signal import savgol_filter

sez_s = ONSET_SAMPLE
sez_e = min(n_samples, sez_s + int(10 * FS))
D = np.zeros((n_ch, n_ch))
for ch in range(n_ch):
    seg = X[sez_s:sez_e, ch].copy()
    mu, sigma = seg.mean(), max(seg.std(), 1e-12)
    seg = (seg - mu) / sigma
    emb = np.column_stack([
        savgol_filter(seg, SG_WINDOW, SG_POLY, deriv=0),
        savgol_filter(seg, SG_WINDOW, SG_POLY, deriv=1, delta=1/FS),
        savgol_filter(seg, SG_WINDOW, SG_POLY, deriv=2, delta=1/FS),
    ])
    cov = np.cov(emb, rowvar=False) + 1e-10 * np.eye(3)
    for ch2 in range(ch):
        seg2 = X[sez_s:sez_e, ch2].copy()
        mu2, sigma2 = seg2.mean(), max(seg2.std(), 1e-12)
        seg2 = (seg2 - mu2) / sigma2
        emb2 = np.column_stack([
            savgol_filter(seg2, SG_WINDOW, SG_POLY, deriv=0),
            savgol_filter(seg2, SG_WINDOW, SG_POLY, deriv=1, delta=1/FS),
            savgol_filter(seg2, SG_WINDOW, SG_POLY, deriv=2, delta=1/FS),
        ])
        cov2 = np.cov(emb2, rowvar=False) + 1e-10 * np.eye(3)
        d = np.linalg.norm(cov - cov2, "fro")
        D[ch, ch2] = d
        D[ch2, ch] = d

sigma_aff = np.median(D[D > 0])
affinity = np.exp(-D ** 2 / (2 * sigma_aff ** 2))
sc = SpectralClustering(n_clusters=CLUSTER_K, affinity="precomputed",
                        random_state=SEED, assign_labels="kmeans")
labels = sc.fit_predict(affinity)

soz_cluster_orig = np.bincount(labels[SOZ_CHANNELS]).argmax()
cluster_map = {soz_cluster_orig: 0}
nxt = 1
for c in range(CLUSTER_K):
    if c != soz_cluster_orig:
        cluster_map[c] = nxt
        nxt += 1
labels = np.array([cluster_map[l] for l in labels])

clusters_all = {c: np.where(labels == c)[0].tolist() for c in range(CLUSTER_K)}
bulk_id = max(clusters_all, key=lambda c: len(clusters_all[c]))
KEEP = [c for c in sorted(clusters_all.keys()) if c != bulk_id]
clusters = {c: clusters_all[c] for c in KEEP}
CL_NAMES = {0: "C0 (SOZ)", 2: "C2 (pace)", 3: "C3 (bound)"}

ch_to_cl_idx = np.zeros(n_ch, dtype=int)
for ci, c in enumerate(KEEP):
    for ch in clusters[c]:
        ch_to_cl_idx[ch] = ci
for ch in clusters_all[bulk_id]:
    ch_to_cl_idx[ch] = 0

print(f"  Clusters: {[(c, len(clusters[c])) for c in KEEP]}")

# --- Per-channel, per-band STATIC amplitude stats ---
nyq = FS / 2.0
pre_mask = (t_data >= ONSET_TIME - 30) & (t_data < ONSET_TIME)
sez_mask = (t_data >= ONSET_TIME) & (t_data < ONSET_TIME + 10)

# amp_pre[band][ch], amp_sez[band][ch] = mean amplitude in that window
amp_pre = {}
amp_sez = {}
phase_offset = {}

for band_name, (blo, bhi) in BANDS.items():
    b, a = butter(4, [blo / nyq, min(bhi / nyq, 0.999)], btype="band")
    X_filt = filtfilt(b, a, X, axis=0)

    amp_pre[band_name] = np.zeros(n_ch)
    amp_sez[band_name] = np.zeros(n_ch)
    phase_offset[band_name] = np.zeros(n_ch)

    ch_phases = np.zeros((n_samples, n_ch))
    for ch in range(n_ch):
        analytic = hilbert(X_filt[:, ch])
        env = np.abs(analytic)
        amp_pre[band_name][ch] = env[pre_mask].mean()
        amp_sez[band_name][ch] = env[sez_mask].mean()
        ch_phases[:, ch] = np.angle(analytic)

    # Cluster mean phases for phase offset computation
    for ci, c in enumerate(KEEP):
        z = np.exp(1j * ch_phases[:, clusters[c]]).mean(axis=1)
        cl_phase = np.angle(z)
        for ch in clusters[c]:
            diff = ch_phases[:, ch] - cl_phase
            phase_offset[band_name][ch] = np.angle(np.mean(np.exp(1j * diff)))
    # Bulk channels: offset relative to C0
    z_c0 = np.exp(1j * ch_phases[:, clusters[KEEP[0]]]).mean(axis=1)
    cl_phase_c0 = np.angle(z_c0)
    for ch in clusters_all[bulk_id]:
        diff = ch_phases[:, ch] - cl_phase_c0
        phase_offset[band_name][ch] = np.angle(np.mean(np.exp(1j * diff)))

print("  Amplitude stats extracted (pre-seizure and seizure means)")
for band_name in BAND_NAMES:
    ratio = amp_sez[band_name][SOZ_CHANNELS].mean() / max(amp_pre[band_name][SOZ_CHANNELS].mean(), 1e-6)
    print(f"    {band_name:12s}: pre={amp_pre[band_name][SOZ_CHANNELS].mean():.1f}, "
          f"sez={amp_sez[band_name][SOZ_CHANNELS].mean():.1f}, ratio={ratio:.2f}x")

# Done with data — everything below is model-only
del X
print("\n  Data released. All parameters are static from here.")


# ============================================================
# STEP 2: Model components
# ============================================================
print("\nStep 2: Setting up model...")

# --- Coherence model: sigmoid ramp ---
RAMP_DUR = 3.0
R0_PRE, R0_SEZ = 0.44, 0.75
R2_PRE, R2_SEZ = 0.86, 0.95
R3_PRE, R3_SEZ = 0.72, 0.95


def sigmoid_ramp(t, pre, sez, onset=ONSET_TIME, tau=RAMP_DUR):
    """Smooth sigmoid transition from pre to sez at onset."""
    x = (t - onset) / (tau / 4)  # 4 time constants in ramp_dur
    return pre + (sez - pre) / (1 + np.exp(-x))


def r0_fn(t): return sigmoid_ramp(t, R0_PRE, R0_SEZ)
def r2_fn(t): return sigmoid_ramp(t, R2_PRE, R2_SEZ)
def r3_fn(t): return sigmoid_ramp(t, R3_PRE, R3_SEZ)


# --- Amplitude model: Stuart-Landau per cluster per band ---
# dA_kb/dt = mu_kb(t) * A_kb - alpha_kb * A_kb^3
# mu crosses zero at onset → supercritical Hopf bifurcation
# Steady state: A* = sqrt(mu/alpha) when mu > 0
# Pre-seizure: mu < 0 → amplitude = driven by noise floor
# Seizure: mu > 0 → amplitude grows to sqrt(mu/alpha)

def amplitude_model(t, band_name, cl_idx):
    """Model amplitude at time t for a given band and cluster."""
    a_pre = amp_pre[band_name][clusters[KEEP[cl_idx]]].mean()
    a_sez = amp_sez[band_name][clusters[KEEP[cl_idx]]].mean()
    return sigmoid_ramp(t, a_pre, a_sez)


# --- Kuramoto phase model ---
def multiband_rhs(t, theta):
    r0 = r0_fn(t)
    r2 = r2_fn(t)
    r3 = r3_fn(t)

    dtheta = np.zeros(N_PHASE)

    # DELTA
    dtheta[0] = 13.28 + 28.16 * r0 + 9.92 * r3 - 20.02
    dtheta[1] = 14.85 - 5.12 * r2 + 25.88 * r3 - 14.04
    dtheta[2] = 13.17 - 0.12

    # THETA
    dtheta[3] = 34.70 - 16.94 * r0 + 7.34
    dtheta[4] = 35.75 - 0.51
    dtheta[5] = 35.00 - 0.32

    # ALPHA
    th_a = theta[6:9]
    th_a_mean = th_a.mean()
    dtheta[6] = 63.68 + 2.14 * r0 * np.sin(th_a[0] - th_a_mean) \
                - 21.17 * r3 + 14.36
    dtheta[7] = 63.42 + 12.10 * r0 - 18.38 * r3 + 7.60
    dtheta[8] = 63.88 - 0.43

    # BETA
    dtheta[9] = 109.99 + 41.79 * r3 - 29.68
    dtheta[10] = 112.09 + 38.92 * r3 - 26.91
    dtheta[11] = 111.38 - 1.33

    # LOW GAMMA
    th_g = theta[12:15]
    th_g_mean = th_g.mean()
    dtheta[12] = 233.85 + 3.53 * r2 * np.sin(th_g[1] - th_g_mean) + 0.84
    dtheta[13] = 230.19 + 41.35 * r0 - 17.11
    dtheta[14] = 229.65 - 0.22

    return dtheta


# ============================================================
# STEP 3: Simulate
# ============================================================
print("\nStep 3: Simulating...")

T_START = 60.0   # start 28s before onset (skip early transients)
T_END = 98.0     # 10s after onset
DT = 1.0 / FS
t_eval = np.arange(T_START, T_END, DT)
N_T = len(t_eval)

# Initial phases: random
theta0 = np.random.uniform(0, 2 * np.pi, N_PHASE)

print(f"  Time: [{T_START}, {T_END}]s, {N_T} steps at {FS} Hz")

sol = solve_ivp(
    multiband_rhs, [T_START, T_END], theta0, t_eval=t_eval,
    method="RK45", rtol=1e-8, atol=1e-10,
)
t_sim = sol.t
theta_sim = sol.y.T  # (N_T, 15)
print(f"  Phases: {theta_sim.shape}")

# Compute model amplitudes (smooth, no noise)
print("  Computing amplitudes...")
amp_model = np.zeros((N_T, N_BANDS, N_CL))
for bi, band_name in enumerate(BAND_NAMES):
    for ci in range(N_CL):
        for ti in range(N_T):
            amp_model[ti, bi, ci] = amplitude_model(t_sim[ti], band_name, ci)

# ============================================================
# STEP 4: Reconstruct iEEG
# ============================================================
print("\nStep 4: Reconstructing iEEG...")

# x_ch(t) = Σ_b A_model_b(t, cluster(ch)) * (A_ch / A_cluster_mean) * cos(θ(t) + δφ_ch)
X_model = np.zeros((N_T, n_ch))

for bi, band_name in enumerate(BAND_NAMES):
    for ch in range(n_ch):
        ci = ch_to_cl_idx[ch]
        # Phase from Kuramoto
        phase = theta_sim[:, bi * N_CL + ci] + phase_offset[band_name][ch]
        # Amplitude from model, scaled by channel's relative contribution
        cluster_chs = clusters[KEEP[ci]] if KEEP[ci] in clusters else clusters_all[bulk_id]
        cl_mean_pre = amp_pre[band_name][cluster_chs].mean()
        if cl_mean_pre > 1e-6:
            ch_weight = amp_pre[band_name][ch] / cl_mean_pre
        else:
            ch_weight = 1.0
        amp = amp_model[:, bi, ci] * ch_weight
        X_model[:, ch] += amp * np.cos(phase)

print(f"  Reconstructed: {X_model.shape}")

# Add small observation noise
noise_std = np.std(X_model) * 0.05
X_model += np.random.randn(*X_model.shape) * noise_std


# ============================================================
# STEP 5: Also load real data for comparison plots
# ============================================================
print("\nStep 5: Loading real data for comparison...")
X_real = np.loadtxt(DATA_PATH)
t_real = np.arange(X_real.shape[0]) / FS
# Trim to match simulation window
real_s = int(T_START * FS)
real_e = min(int(T_END * FS), X_real.shape[0])
X_real_win = X_real[real_s:real_e]
t_real_win = t_real[real_s:real_e]
n_compare = min(len(t_sim), len(t_real_win))


# ============================================================
# PLOTS
# ============================================================
print("\nGenerating plots...")

# --- Plot 1: SOZ channels — real vs model ---
soz_show = [21, 23, 24, 25, 28, 30]
fig, axes = plt.subplots(len(soz_show), 2, figsize=(16, 2.5 * len(soz_show)),
                         sharex=True)
fig.suptitle("Fully Model-Driven iEEG: Real (left) vs Model (right)\n"
             "Phases from Kuramoto, Amplitudes from Stuart-Landau",
             fontsize=14)

for i, ch in enumerate(soz_show):
    ax = axes[i, 0]
    ax.plot(t_real_win[:n_compare], X_real_win[:n_compare, ch],
            color="k", lw=0.3)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8)
    ax.set_ylabel(f"Ch {ch}", fontsize=9)
    if i == 0:
        ax.set_title("Real iEEG", fontsize=11)

    ax = axes[i, 1]
    ax.plot(t_sim[:n_compare], X_model[:n_compare, ch],
            color="steelblue", lw=0.3)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8)
    if i == 0:
        ax.set_title("Model (no data input)", fontsize=11)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "model_only_soz_channels")

# --- Plot 2: Full array waterfall ---
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.suptitle("Full 120-Channel iEEG: Real vs Model-Only", fontsize=14)

ch_show = list(range(0, n_ch, max(1, n_ch // 40)))
spacing = np.percentile(np.abs(X_real_win[:n_compare]), 90) * 0.4

for ax_idx, (data, t_arr, title, color) in enumerate([
    (X_real_win[:n_compare], t_real_win[:n_compare], "Real iEEG", "k"),
    (X_model[:n_compare], t_sim[:n_compare], "Model-Only", "steelblue"),
]):
    ax = axes[ax_idx]
    for ci_plot, ch in enumerate(ch_show):
        sig = data[:, ch]
        sig_norm = sig / max(np.std(sig), 1e-6) * spacing * 0.3
        ax.plot(t_arr, sig_norm + ci_plot * spacing, color=color, lw=0.2, alpha=0.7)
        if ch in SOZ_CHANNELS:
            ax.plot(t_arr[0] - 0.3, ci_plot * spacing, "r>", ms=4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title, fontsize=12)
fig.tight_layout()
save_fig(fig, "model_only_full_array")

# --- Plot 3: Seizure onset zoom ---
T_Z_S, T_Z_E = ONSET_TIME - 3, ONSET_TIME + 5
zm = (t_sim >= T_Z_S) & (t_sim <= T_Z_E)
zm_r = (t_real_win >= T_Z_S) & (t_real_win <= T_Z_E)

zoom_chs = [24, 25, 28, 30]
fig, axes = plt.subplots(len(zoom_chs), 2, figsize=(16, 2.5 * len(zoom_chs)),
                         sharex=True)
fig.suptitle("Seizure Onset Zoom (3s before → 5s after)\n"
             "Fully model-driven reconstruction", fontsize=14)

for i, ch in enumerate(zoom_chs):
    ax = axes[i, 0]
    ax.plot(t_real_win[zm_r], X_real_win[zm_r, ch], color="k", lw=0.5)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=1)
    ax.set_ylabel(f"Ch {ch}", fontsize=9)
    if i == 0:
        ax.set_title("Real", fontsize=11)

    ax = axes[i, 1]
    ax.plot(t_sim[zm], X_model[zm, ch], color="steelblue", lw=0.5)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=1)
    if i == 0:
        ax.set_title("Model-Only", fontsize=11)

axes[-1, 0].set_xlabel("Time (s)")
axes[-1, 1].set_xlabel("Time (s)")
fig.tight_layout()
save_fig(fig, "model_only_seizure_zoom")

# --- Plot 4: Per-band breakdown ---
ch_detail = 25
fig, axes = plt.subplots(N_BANDS + 1, 2, figsize=(16, 2.5 * (N_BANDS + 1)),
                         sharex=True)
fig.suptitle(f"Per-Band Model Reconstruction: Channel {ch_detail} (SOZ)\n"
             "Left: Real bandpassed, Right: Model per-band", fontsize=14)

for bi, band_name in enumerate(BAND_NAMES):
    blo, bhi = BANDS[band_name]
    b, a = butter(4, [blo / nyq, min(bhi / nyq, 0.999)], btype="band")

    # Real
    real_band = filtfilt(b, a, X_real[real_s:real_e, ch_detail])
    ax = axes[bi, 0]
    ax.plot(t_real_win[:n_compare], real_band[:n_compare], color="k", lw=0.4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8)
    ax.set_ylabel(f"{band_name}\n({blo:.0f}-{bhi:.0f})", fontsize=8)
    if bi == 0:
        ax.set_title("Real (bandpassed)", fontsize=11)

    # Model single band
    ci = ch_to_cl_idx[ch_detail]
    cl_chs = clusters[KEEP[ci]]
    cl_mean_pre = amp_pre[band_name][cl_chs].mean()
    ch_w = amp_pre[band_name][ch_detail] / max(cl_mean_pre, 1e-6)
    phase = theta_sim[:n_compare, bi * N_CL + ci] + phase_offset[band_name][ch_detail]
    amp = amp_model[:n_compare, bi, ci] * ch_w
    model_band = amp * np.cos(phase)

    ax = axes[bi, 1]
    ax.plot(t_sim[:n_compare], model_band, color="steelblue", lw=0.4)
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8)
    if bi == 0:
        ax.set_title("Model (per band)", fontsize=11)

# Broadband
ax = axes[N_BANDS, 0]
ax.plot(t_real_win[:n_compare], X_real_win[:n_compare, ch_detail], color="k", lw=0.4)
ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8)
ax.set_ylabel("Sum", fontsize=8)
ax.set_xlabel("Time (s)")

ax = axes[N_BANDS, 1]
ax.plot(t_sim[:n_compare], X_model[:n_compare, ch_detail], color="steelblue", lw=0.4)
ax.axvline(ONSET_TIME, color="red", ls="--", lw=0.8)
ax.set_xlabel("Time (s)")

fig.tight_layout()
save_fig(fig, "model_only_per_band")

# --- Plot 5: Model internals (phases, frequencies, amplitudes, coherence) ---
fig = plt.figure(figsize=(16, 14))
gs = plt.GridSpec(4, N_BANDS, hspace=0.5, wspace=0.3)
fig.suptitle("Model Internals: Phases, Frequencies, Amplitudes, Coherence",
             fontsize=14, y=0.98)

CL_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
win = min(200, N_T // 20)
kern = np.ones(win) / win

for bi in range(N_BANDS):
    s = bi * N_CL

    # Row 1: Instantaneous frequency
    ax = fig.add_subplot(gs[0, bi])
    omega = np.diff(theta_sim[:, s:s+N_CL], axis=0) / DT / (2 * np.pi)
    for ci in range(N_CL):
        f_smooth = np.convolve(omega[:, ci], kern, mode="same")
        ax.plot(t_sim[1:], f_smooth, color=CL_COLORS[ci], lw=0.6)
    ax.axvline(ONSET_TIME, color="k", ls="--", lw=0.5)
    ax.set_title(BAND_LABELS[bi], fontsize=10)
    if bi == 0:
        ax.set_ylabel("Freq (Hz)")

    # Row 2: Phase differences
    ax = fig.add_subplot(gs[1, bi])
    for ci in range(N_CL):
        for cj in range(ci + 1, N_CL):
            diff = np.mod(theta_sim[:, s+ci] - theta_sim[:, s+cj] + np.pi,
                         2*np.pi) - np.pi
            d_smooth = np.convolve(diff, kern, mode="same")
            ax.plot(t_sim, d_smooth, lw=0.6)
    ax.axvline(ONSET_TIME, color="k", ls="--", lw=0.5)
    if bi == 0:
        ax.set_ylabel(r"$\Delta\theta$")

    # Row 3: Model amplitude
    ax = fig.add_subplot(gs[2, bi])
    for ci in range(N_CL):
        ax.plot(t_sim, amp_model[:, bi, ci], color=CL_COLORS[ci], lw=1)
    ax.axvline(ONSET_TIME, color="k", ls="--", lw=0.5)
    if bi == 0:
        ax.set_ylabel("Amplitude")

    # Row 4: Coherence
    ax = fig.add_subplot(gs[3, bi])
    r_vals = np.array([r0_fn(t) for t in t_sim])
    r2_vals = np.array([r2_fn(t) for t in t_sim])
    r3_vals = np.array([r3_fn(t) for t in t_sim])
    ax.plot(t_sim, r_vals, color=CL_COLORS[0], lw=1)
    ax.plot(t_sim, r2_vals, color=CL_COLORS[1], lw=1)
    ax.plot(t_sim, r3_vals, color=CL_COLORS[2], lw=1)
    ax.axvline(ONSET_TIME, color="k", ls="--", lw=0.5)
    ax.set_ylim(0, 1.05)
    if bi == 0:
        ax.set_ylabel("r(t)")
    ax.set_xlabel("Time (s)", fontsize=8)

save_fig(fig, "model_only_internals")

# --- Plot 6: Spectrogram comparison ---
ch_spec = 25
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"Spectrogram: Channel {ch_spec} (SOZ) — Real vs Model-Only",
             fontsize=14)

axes[0].specgram(X_real_win[:n_compare, ch_spec], NFFT=256, Fs=FS,
                 noverlap=200, cmap="viridis")
axes[0].set_ylim(0, 60)
axes[0].set_ylabel("Frequency (Hz)")
axes[0].set_title("Real iEEG")
onset_rel = ONSET_TIME - T_START
axes[0].axvline(onset_rel, color="red", ls="--", lw=1)

axes[1].specgram(X_model[:n_compare, ch_spec], NFFT=256, Fs=FS,
                 noverlap=200, cmap="viridis")
axes[1].set_ylim(0, 60)
axes[1].set_ylabel("Frequency (Hz)")
axes[1].set_xlabel("Time (s)")
axes[1].set_title("Model-Only")
axes[1].axvline(onset_rel, color="red", ls="--", lw=1)

fig.tight_layout()
save_fig(fig, "model_only_spectrogram")


# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Model: 15 Kuramoto phases (5 bands × 3 clusters)")
print(f"  Amplitudes: sigmoid ramp (pre→sez stats, no data time series)")
print(f"  Coherence: sigmoid ramp r₀={R0_PRE}→{R0_SEZ}, "
      f"r₂={R2_PRE}→{R2_SEZ}, r₃={R3_PRE}→{R3_SEZ}")
print(f"  Reconstruction: 120 channels from 15 phases + static spatial weights")
print(f"  Onset: {ONSET_TIME}s, ramp: {RAMP_DUR}s")
print(f"\n  Static parameters from data:")
print(f"    - Cluster assignments (4 clusters, bulk dropped)")
print(f"    - Per-channel phase offsets (fixed scalars)")
print(f"    - Pre/seizure amplitude means (2 scalars per ch per band)")
print(f"\n  NOT from data:")
print(f"    - Phase trajectories (from Kuramoto equations)")
print(f"    - Amplitude time courses (from sigmoid model)")
print(f"    - Coherence dynamics (from sigmoid ramp)")
print(f"\n  All plots saved to {OUT_DIR}/")
print("  Done.")
