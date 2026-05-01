"""
Cluster-Resolved Kuramoto Order Parameter Analysis of iEEG Data
================================================================
Given differential embedding clusters (Frobenius / Spectral / k=4 / seizure window),
compute LOCAL order parameters for each cluster and compare synchronization dynamics
within vs between clusters, pre-seizure vs seizure.

Pipeline:
  1. Recompute best clustering (Frobenius, Spectral, k=4, seizure 87-97s)
  2. Hilbert-transform each channel to get instantaneous phase
  3. Compute cluster-level and global order parameters r(t)
  4. Statistical comparison: within-cluster vs between-cluster coherence
  5. Seizure-onset change significance per cluster
  6. Publication-quality plots

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Publication style
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ============================================================
# Parameters
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "E3Data.mat"
OUT_DIR = ROOT / "results" / "iEEG" / "kuramoto"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 500  # Hz
N_CHANNELS = 120
SOZ_CHANNELS = list(range(21, 31))  # 0-indexed
ONSET_SAMPLE = 43500  # 87.0s
ONSET_TIME = 87.0

# Differential embedding params (matching clustering scripts)
SG_WINDOW = 51
SG_POLY = 5

# Order parameter smoothing
SG_SMOOTH_WINDOW = 501  # 1s at 500 Hz for visualization smoothing
SG_SMOOTH_POLY = 3

# Bandpass for Hilbert transform (brain oscillation band)
# Broad band 1-100 Hz to capture all oscillatory content
BAND_LO = 1.0
BAND_HZ = 100.0

# Cluster config (best from previous analysis)
CLUSTER_K = 4
CLUSTER_METHOD = "spectral"  # spectral clustering

# Pre-seizure and seizure windows for statistics
PRE_WINDOW = (77.0, 87.0)   # 10s before onset
SEZ_WINDOW = (87.0, 97.0)   # 10s after onset

# Bootstrap parameters
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ============================================================
# Utility: Differential embedding clustering (copied from clustering scripts)
# ============================================================
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
        x_smooth = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=0)
        x_dot = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=1, delta=dt)
        x_ddot = savgol_filter(x, SG_WINDOW, SG_POLY, deriv=2, delta=dt)
        embeddings[ch] = np.column_stack([x_smooth, x_dot, x_ddot])
    return embeddings


def compute_covariance_descriptors(embeddings):
    covs = {}
    for ch, emb in embeddings.items():
        C = np.cov(emb, rowvar=False)
        C += 1e-10 * np.eye(3)
        covs[ch] = C
    return covs


def frobenius_distance_matrix(covs, n_ch):
    D = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            d = np.linalg.norm(covs[i] - covs[j], "fro")
            D[i, j] = d
            D[j, i] = d
    return D


def run_spectral_clustering(D, k, seed=42):
    sigma = np.median(D[D > 0])
    if sigma < 1e-12:
        sigma = 1.0
    affinity = np.exp(-D ** 2 / (2 * sigma ** 2))
    sc = SpectralClustering(n_clusters=k, affinity="precomputed",
                            random_state=seed, assign_labels="kmeans")
    return sc.fit_predict(affinity)


# ============================================================
# Utility: Phase extraction via Hilbert transform
# ============================================================
def bandpass_filter(data, lo, hi, fs, order=4):
    """Apply zero-phase Butterworth bandpass filter. data: (T, C)."""
    nyq = fs / 2.0
    lo_norm = max(lo / nyq, 0.001)
    hi_norm = min(hi / nyq, 0.999)
    b, a = butter(order, [lo_norm, hi_norm], btype="band")
    return filtfilt(b, a, data, axis=0)


def extract_phases(data, fs, lo=BAND_LO, hi=BAND_HZ):
    """
    Extract instantaneous phases for all channels via Hilbert transform.

    Parameters
    ----------
    data : (T, C) array
    fs : sampling rate
    lo, hi : bandpass limits

    Returns
    -------
    phases : (T, C) array of instantaneous phases in [-pi, pi]
    """
    # Bandpass filter to isolate oscillatory content
    filtered = bandpass_filter(data, lo, hi, fs)
    T, C = filtered.shape
    phases = np.zeros((T, C))
    for ch in range(C):
        analytic = hilbert(filtered[:, ch])
        phases[:, ch] = np.angle(analytic)
    return phases


# ============================================================
# Utility: Kuramoto order parameter
# ============================================================
def kuramoto_order_parameter(phases, channel_indices=None):
    """
    Compute Kuramoto order parameter r(t) for a subset of channels.

    r(t) = |1/N * sum_i exp(i * theta_i(t))|

    Parameters
    ----------
    phases : (T, C) array
    channel_indices : list/array of channel indices (default: all)

    Returns
    -------
    r : (T,) array of order parameter values in [0, 1]
    psi : (T,) array of mean phase angle
    """
    if channel_indices is None:
        channel_indices = np.arange(phases.shape[1])
    ch_idx = np.asarray(channel_indices)
    theta = phases[:, ch_idx]  # (T, |C_k|)
    z = np.exp(1j * theta).mean(axis=1)  # (T,)
    return np.abs(z), np.angle(z)


def pairwise_plv(phases, idx_a, idx_b):
    """
    Compute mean pairwise Phase Locking Value between two sets of channels.

    PLV_{ij} = |<exp(i*(theta_i - theta_j))>_t|
    Returns mean PLV across all (i,j) pairs with i in idx_a, j in idx_b.
    """
    plvs = []
    for i in idx_a:
        for j in idx_b:
            if i == j:
                continue
            diff = phases[:, i] - phases[:, j]
            plv = np.abs(np.exp(1j * diff).mean())
            plvs.append(plv)
    return np.mean(plvs) if plvs else 0.0


def within_cluster_plv(phases, cluster_indices):
    """Mean pairwise PLV within a cluster."""
    idx = list(cluster_indices)
    if len(idx) < 2:
        return 0.0
    plvs = []
    for ii in range(len(idx)):
        for jj in range(ii + 1, len(idx)):
            diff = phases[:, idx[ii]] - phases[:, idx[jj]]
            plv = np.abs(np.exp(1j * diff).mean())
            plvs.append(plv)
    return np.mean(plvs)


def between_cluster_plv(phases, idx_a, idx_b):
    """Mean pairwise PLV between two clusters."""
    return pairwise_plv(phases, idx_a, idx_b)


# ============================================================
# LOAD DATA
# ============================================================
print("=" * 70)
print("CLUSTER-RESOLVED KURAMOTO ORDER PARAMETER ANALYSIS")
print("=" * 70)

print("\nLoading data...")
mat = sio.loadmat(DATA_PATH)
X1 = mat["X1"]  # (49972, 120)
n_samples, n_ch = X1.shape
t_axis = np.arange(n_samples) / FS
print(f"  X1 shape: {X1.shape}, duration: {n_samples / FS:.1f}s")

soz_mask = np.array([1 if ch in SOZ_CHANNELS else 0 for ch in range(N_CHANNELS)])

# ============================================================
# STEP 1: Recompute cluster assignments (seizure window)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Recompute differential embedding clusters")
print("=" * 70)

sez_start = ONSET_SAMPLE
sez_end = min(n_samples, sez_start + int(10.0 * FS))
data_sez = X1[sez_start:sez_end, :]

embs_sez = compute_embedding(data_sez, FS)
covs_sez = compute_covariance_descriptors(embs_sez)
D_sez = frobenius_distance_matrix(covs_sez, N_CHANNELS)

labels = run_spectral_clustering(D_sez, CLUSTER_K)
ari = adjusted_rand_score(soz_mask, labels)
sil = silhouette_score(D_sez, labels, metric="precomputed")

# Identify which cluster contains SOZ
soz_labels = labels[SOZ_CHANNELS]
soz_cluster = np.bincount(soz_labels).argmax()
soz_purity = np.mean(soz_labels == soz_cluster)

# Build cluster membership dict
clusters = {}
for c in range(CLUSTER_K):
    members = np.where(labels == c)[0].tolist()
    clusters[c] = members

# Relabel so SOZ cluster is always cluster 0 for consistent plotting
cluster_map = {soz_cluster: 0}
next_label = 1
for c in range(CLUSTER_K):
    if c != soz_cluster:
        cluster_map[c] = next_label
        next_label += 1

labels_remapped = np.array([cluster_map[l] for l in labels])
clusters_remapped = {}
for c in range(CLUSTER_K):
    clusters_remapped[c] = np.where(labels_remapped == c)[0].tolist()

# Use remapped labels from now on
labels = labels_remapped
clusters = clusters_remapped
soz_cluster = 0

print(f"  Clustering: Frobenius, Spectral, k={CLUSTER_K}")
print(f"  ARI vs SOZ: {ari:.3f}, Silhouette: {sil:.3f}")
print(f"  SOZ purity in cluster 0: {soz_purity:.1%}")
print(f"\n  Cluster membership:")
for c in range(CLUSTER_K):
    members = clusters[c]
    n_soz = sum(1 for m in members if m in SOZ_CHANNELS)
    soz_str = f" (contains {n_soz} SOZ channels)" if n_soz > 0 else ""
    print(f"    Cluster {c}: {len(members)} channels{soz_str}")
    if n_soz > 0:
        soz_in = [m for m in members if m in SOZ_CHANNELS]
        print(f"      SOZ channels in cluster: {soz_in}")
        nonsoz_in = [m for m in members if m not in SOZ_CHANNELS]
        if nonsoz_in:
            print(f"      Non-SOZ channels in cluster: {sorted(nonsoz_in)}")


# ============================================================
# STEP 2: Extract instantaneous phases (full recording)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Extract instantaneous phases via Hilbert transform")
print("=" * 70)

phases = extract_phases(X1, FS, lo=BAND_LO, hi=BAND_HZ)
print(f"  Phases shape: {phases.shape}")
print(f"  Bandpass: {BAND_LO}-{BAND_HZ} Hz")


# ============================================================
# STEP 3: Compute order parameters
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Compute local and global order parameters")
print("=" * 70)

# Global order parameter
r_global, psi_global = kuramoto_order_parameter(phases)
print(f"  Global r: mean={r_global.mean():.4f}, std={r_global.std():.4f}")

# Per-cluster order parameters
r_clusters = {}
psi_clusters = {}
cluster_names = {}
for c in range(CLUSTER_K):
    ch_idx = clusters[c]
    r_c, psi_c = kuramoto_order_parameter(phases, ch_idx)
    r_clusters[c] = r_c
    psi_clusters[c] = psi_c
    n_soz = sum(1 for m in ch_idx if m in SOZ_CHANNELS)
    if c == 0:
        cluster_names[c] = f"Cluster 0 (SOZ, n={len(ch_idx)})"
    else:
        cluster_names[c] = f"Cluster {c} (n={len(ch_idx)})"
    print(f"  {cluster_names[c]}: mean r = {r_c.mean():.4f}, std = {r_c.std():.4f}")

# Smoothed versions for visualization
r_global_smooth = savgol_filter(r_global, SG_SMOOTH_WINDOW, SG_SMOOTH_POLY)
r_clusters_smooth = {}
for c in range(CLUSTER_K):
    r_clusters_smooth[c] = savgol_filter(r_clusters[c], SG_SMOOTH_WINDOW, SG_SMOOTH_POLY)


# ============================================================
# STEP 4: Pre-seizure vs Seizure statistics
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Pre-seizure vs Seizure window statistics")
print("=" * 70)

pre_idx = np.arange(int(PRE_WINDOW[0] * FS), int(PRE_WINDOW[1] * FS))
sez_idx = np.arange(int(SEZ_WINDOW[0] * FS), min(int(SEZ_WINDOW[1] * FS), n_samples))

# Ensure indices are valid
pre_idx = pre_idx[pre_idx < n_samples]
sez_idx = sez_idx[sez_idx < n_samples]

stats = {}

print(f"\n  {'':>30s} {'Pre-seizure':>15s} {'Seizure':>15s} {'Delta':>10s}")
print(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*10}")

# Global
r_pre = r_global[pre_idx]
r_sez = r_global[sez_idx]
delta = r_sez.mean() - r_pre.mean()
stats["Global"] = {"pre_mean": r_pre.mean(), "pre_std": r_pre.std(),
                    "sez_mean": r_sez.mean(), "sez_std": r_sez.std(),
                    "delta": delta}
print(f"  {'Global':>30s} {r_pre.mean():>7.4f}+/-{r_pre.std():.4f} "
      f"{r_sez.mean():>7.4f}+/-{r_sez.std():.4f} {delta:>+10.4f}")

for c in range(CLUSTER_K):
    r_pre_c = r_clusters[c][pre_idx]
    r_sez_c = r_clusters[c][sez_idx]
    delta_c = r_sez_c.mean() - r_pre_c.mean()
    stats[cluster_names[c]] = {"pre_mean": r_pre_c.mean(), "pre_std": r_pre_c.std(),
                                "sez_mean": r_sez_c.mean(), "sez_std": r_sez_c.std(),
                                "delta": delta_c}
    print(f"  {cluster_names[c]:>30s} {r_pre_c.mean():>7.4f}+/-{r_pre_c.std():.4f} "
          f"{r_sez_c.mean():>7.4f}+/-{r_sez_c.std():.4f} {delta_c:>+10.4f}")


# ============================================================
# STEP 5: Within-cluster vs Between-cluster phase coherence
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Within-cluster vs Between-cluster phase coherence (PLV)")
print("=" * 70)

# Compute on seizure window phases for computational tractability
phases_sez = phases[sez_idx, :]
phases_pre = phases[pre_idx, :]

# Within-cluster PLV
print("\n  Within-cluster PLV (seizure window):")
within_plvs_sez = {}
within_plvs_pre = {}
for c in range(CLUSTER_K):
    plv_sez = within_cluster_plv(phases_sez, clusters[c])
    plv_pre = within_cluster_plv(phases_pre, clusters[c])
    within_plvs_sez[c] = plv_sez
    within_plvs_pre[c] = plv_pre
    print(f"    {cluster_names[c]}: pre={plv_pre:.4f}, sez={plv_sez:.4f}, "
          f"delta={plv_sez - plv_pre:+.4f}")

# Between-cluster PLV (all pairs)
print("\n  Between-cluster PLV (seizure window):")
between_plvs_sez = {}
between_plvs_pre = {}
for ci in range(CLUSTER_K):
    for cj in range(ci + 1, CLUSTER_K):
        plv_sez = between_cluster_plv(phases_sez, clusters[ci], clusters[cj])
        plv_pre = between_cluster_plv(phases_pre, clusters[ci], clusters[cj])
        key = (ci, cj)
        between_plvs_sez[key] = plv_sez
        between_plvs_pre[key] = plv_pre
        print(f"    Cluster {ci} <-> Cluster {cj}: pre={plv_pre:.4f}, sez={plv_sez:.4f}, "
              f"delta={plv_sez - plv_pre:+.4f}")

# Summary: mean within vs mean between
all_within_sez = list(within_plvs_sez.values())
all_between_sez = list(between_plvs_sez.values())
all_within_pre = list(within_plvs_pre.values())
all_between_pre = list(between_plvs_pre.values())

print(f"\n  Summary:")
print(f"    Mean within-cluster PLV:  pre={np.mean(all_within_pre):.4f}, sez={np.mean(all_within_sez):.4f}")
print(f"    Mean between-cluster PLV: pre={np.mean(all_between_pre):.4f}, sez={np.mean(all_between_sez):.4f}")
print(f"    Ratio (within/between):   pre={np.mean(all_within_pre)/np.mean(all_between_pre):.3f}, "
      f"sez={np.mean(all_within_sez)/np.mean(all_between_sez):.3f}")


# ============================================================
# STEP 6: Statistical tests
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Statistical tests")
print("=" * 70)

# --- 6A: Permutation test for within > between cluster coherence ---
print("\n  6A: Permutation test: within-cluster PLV > between-cluster PLV")
print("      (Null: cluster labels carry no phase-coherence information)")

rng = np.random.RandomState(42)

# Observed statistic: mean within - mean between (seizure window)
observed_within = np.mean(all_within_sez)
observed_between = np.mean(all_between_sez)
observed_stat = observed_within - observed_between
print(f"      Observed: within={observed_within:.4f}, between={observed_between:.4f}, "
      f"diff={observed_stat:.4f}")

# Permutation: shuffle cluster labels, recompute within/between PLV
# For computational efficiency, precompute all pairwise PLVs once
print(f"      Precomputing full pairwise PLV matrix ({N_CHANNELS} x {N_CHANNELS})...")
plv_matrix_sez = np.zeros((N_CHANNELS, N_CHANNELS))
for i in range(N_CHANNELS):
    for j in range(i + 1, N_CHANNELS):
        diff = phases_sez[:, i] - phases_sez[:, j]
        plv = np.abs(np.exp(1j * diff).mean())
        plv_matrix_sez[i, j] = plv
        plv_matrix_sez[j, i] = plv

def compute_within_between_from_matrix(plv_mat, label_vec, k):
    """Compute mean within and between cluster PLV from precomputed matrix."""
    within_vals = []
    between_vals = []
    for c in range(k):
        members_c = np.where(label_vec == c)[0]
        if len(members_c) < 2:
            continue
        # Within
        for ii in range(len(members_c)):
            for jj in range(ii + 1, len(members_c)):
                within_vals.append(plv_mat[members_c[ii], members_c[jj]])
        # Between: pairs with other clusters
        for c2 in range(c + 1, k):
            members_c2 = np.where(label_vec == c2)[0]
            for ii in members_c:
                for jj in members_c2:
                    between_vals.append(plv_mat[ii, jj])
    return np.mean(within_vals) if within_vals else 0.0, \
           np.mean(between_vals) if between_vals else 0.0

print(f"      Running {N_PERMUTATIONS} permutations...")
perm_diffs = np.zeros(N_PERMUTATIONS)
for p in range(N_PERMUTATIONS):
    perm_labels = labels.copy()
    rng.shuffle(perm_labels)
    w, b = compute_within_between_from_matrix(plv_matrix_sez, perm_labels, CLUSTER_K)
    perm_diffs[p] = w - b

p_value_coherence = np.mean(perm_diffs >= observed_stat)
z_score_coherence = (observed_stat - perm_diffs.mean()) / max(perm_diffs.std(), 1e-12)
print(f"      Permutation p-value: {p_value_coherence:.4f}")
print(f"      z-score: {z_score_coherence:.1f}")
print(f"      Mean permuted diff: {perm_diffs.mean():.4f} +/- {perm_diffs.std():.4f}")

# --- 6B: Bootstrap CI on seizure-onset change for each cluster ---
print(f"\n  6B: Bootstrap CI on order parameter change at seizure onset")

bootstrap_results = {}
for c in range(CLUSTER_K):
    r_pre_c = r_clusters[c][pre_idx]
    r_sez_c = r_clusters[c][sez_idx]
    observed_delta = r_sez_c.mean() - r_pre_c.mean()

    # Bootstrap: resample within each window, compute delta
    boot_deltas = np.zeros(N_BOOTSTRAP)
    n_pre = len(r_pre_c)
    n_sez = len(r_sez_c)
    for b in range(N_BOOTSTRAP):
        boot_pre = rng.choice(r_pre_c, size=n_pre, replace=True).mean()
        boot_sez = rng.choice(r_sez_c, size=n_sez, replace=True).mean()
        boot_deltas[b] = boot_sez - boot_pre

    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)
    bootstrap_results[c] = {
        "delta": observed_delta,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "boot_deltas": boot_deltas,
    }
    sig = "YES" if (ci_lo > 0 or ci_hi < 0) else "NO"
    print(f"    {cluster_names[c]}: delta={observed_delta:+.4f}, "
          f"95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}], significant={sig}")

# Global bootstrap
r_pre_g = r_global[pre_idx]
r_sez_g = r_global[sez_idx]
observed_delta_g = r_sez_g.mean() - r_pre_g.mean()
boot_deltas_g = np.zeros(N_BOOTSTRAP)
for b in range(N_BOOTSTRAP):
    boot_pre = rng.choice(r_pre_g, size=len(r_pre_g), replace=True).mean()
    boot_sez = rng.choice(r_sez_g, size=len(r_sez_g), replace=True).mean()
    boot_deltas_g[b] = boot_sez - boot_pre
ci_lo_g = np.percentile(boot_deltas_g, 2.5)
ci_hi_g = np.percentile(boot_deltas_g, 97.5)
bootstrap_results["global"] = {
    "delta": observed_delta_g, "ci_lo": ci_lo_g, "ci_hi": ci_hi_g,
    "boot_deltas": boot_deltas_g,
}
sig_g = "YES" if (ci_lo_g > 0 or ci_hi_g < 0) else "NO"
print(f"    {'Global':>30s}: delta={observed_delta_g:+.4f}, "
      f"95% CI=[{ci_lo_g:+.4f}, {ci_hi_g:+.4f}], significant={sig_g}")


# --- 6C: Permutation test: is SOZ cluster delta significantly different from others? ---
print(f"\n  6C: Permutation test: SOZ cluster onset change vs non-SOZ clusters")

# Observed: difference of SOZ cluster delta minus mean of other cluster deltas
soz_delta = bootstrap_results[0]["delta"]
nonsoz_deltas = [bootstrap_results[c]["delta"] for c in range(1, CLUSTER_K)]
observed_diff_delta = soz_delta - np.mean(nonsoz_deltas)
print(f"      SOZ cluster delta: {soz_delta:+.4f}")
print(f"      Non-SOZ cluster deltas: {[f'{d:+.4f}' for d in nonsoz_deltas]}")
print(f"      Observed diff (SOZ - mean non-SOZ): {observed_diff_delta:+.4f}")

# Permutation: shuffle channels across clusters, recompute
print(f"      Running {N_PERMUTATIONS} permutations...")
perm_diff_deltas = np.zeros(N_PERMUTATIONS)

for p in range(N_PERMUTATIONS):
    perm_labels = labels.copy()
    rng.shuffle(perm_labels)
    # Compute order parameter for "cluster 0" under permuted labels
    perm_clusters = {}
    for c in range(CLUSTER_K):
        perm_clusters[c] = np.where(perm_labels == c)[0].tolist()

    r_pre_0, _ = kuramoto_order_parameter(phases[pre_idx, :], perm_clusters[0])
    r_sez_0, _ = kuramoto_order_parameter(phases[sez_idx, :], perm_clusters[0])
    delta_0 = r_sez_0.mean() - r_pre_0.mean()

    other_deltas = []
    for c in range(1, CLUSTER_K):
        r_pre_c, _ = kuramoto_order_parameter(phases[pre_idx, :], perm_clusters[c])
        r_sez_c, _ = kuramoto_order_parameter(phases[sez_idx, :], perm_clusters[c])
        other_deltas.append(r_sez_c.mean() - r_pre_c.mean())

    perm_diff_deltas[p] = delta_0 - np.mean(other_deltas)

p_value_delta = np.mean(np.abs(perm_diff_deltas) >= np.abs(observed_diff_delta))
z_score_delta = (observed_diff_delta - perm_diff_deltas.mean()) / max(perm_diff_deltas.std(), 1e-12)
print(f"      Two-sided permutation p-value: {p_value_delta:.4f}")
print(f"      z-score: {z_score_delta:.1f}")


# ============================================================
# PLOTS
# ============================================================
print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Colour palette for clusters
CLUSTER_COLORS = ["#D62728", "#1F77B4", "#2CA02C", "#FF7F0E", "#9467BD", "#8C564B"]


# --- PLOT 1: Order parameter time series by cluster ---
fig, axes = plt.subplots(CLUSTER_K + 1, 1, figsize=(14, 3 * (CLUSTER_K + 1)),
                          sharex=True, sharey=True)

# Global
ax = axes[0]
ax.plot(t_axis, r_global, color="gray", alpha=0.15, lw=0.3)
ax.plot(t_axis, r_global_smooth, color="black", lw=1.5, label="Global")
ax.axvline(ONSET_TIME, color="red", ls="--", lw=1.5, alpha=0.7, label="Seizure onset")
ax.axvspan(PRE_WINDOW[0], PRE_WINDOW[1], alpha=0.08, color="blue", label="Pre-seizure")
ax.axvspan(SEZ_WINDOW[0], SEZ_WINDOW[1], alpha=0.08, color="red", label="Seizure")
ax.set_ylabel("r(t)")
ax.set_title(f"Global Order Parameter (all {N_CHANNELS} channels)")
ax.legend(loc="upper right", fontsize=8)
ax.set_ylim(0, 1)

# Per cluster
for c in range(CLUSTER_K):
    ax = axes[c + 1]
    color = CLUSTER_COLORS[c]
    ax.plot(t_axis, r_clusters[c], color=color, alpha=0.15, lw=0.3)
    ax.plot(t_axis, r_clusters_smooth[c], color=color, lw=1.5,
            label=cluster_names[c])
    ax.axvline(ONSET_TIME, color="red", ls="--", lw=1.5, alpha=0.7)
    ax.axvspan(PRE_WINDOW[0], PRE_WINDOW[1], alpha=0.08, color="blue")
    ax.axvspan(SEZ_WINDOW[0], SEZ_WINDOW[1], alpha=0.08, color="red")
    ax.set_ylabel("r(t)")
    ax.set_title(cluster_names[c])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)

axes[-1].set_xlabel("Time (s)")
fig.suptitle("Kuramoto Order Parameter by Cluster", fontsize=14, y=1.01)
fig.tight_layout()
save_fig(fig, "order_parameter_by_cluster")


# --- PLOT 2: All clusters overlaid (smoothed only) ---
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t_axis, r_global_smooth, color="black", lw=2, label="Global", alpha=0.8)
for c in range(CLUSTER_K):
    ax.plot(t_axis, r_clusters_smooth[c], color=CLUSTER_COLORS[c], lw=1.5,
            label=cluster_names[c], alpha=0.8)
ax.axvline(ONSET_TIME, color="red", ls="--", lw=2, alpha=0.7, label="Seizure onset")
ax.axvspan(PRE_WINDOW[0], PRE_WINDOW[1], alpha=0.08, color="blue")
ax.axvspan(SEZ_WINDOW[0], SEZ_WINDOW[1], alpha=0.08, color="red")
ax.set_xlabel("Time (s)")
ax.set_ylabel("r(t) (smoothed)")
ax.set_title("Cluster Order Parameters Overlaid (Savitzky-Golay smoothed, 1s window)")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 1)
fig.tight_layout()
save_fig(fig, "order_parameter_overlay")


# --- PLOT 3: Statistics bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3A: Mean r (pre vs sez)
ax = axes[0]
names = ["Global"] + [f"C{c}" for c in range(CLUSTER_K)]
pre_means = [stats["Global"]["pre_mean"]]
sez_means = [stats["Global"]["sez_mean"]]
pre_stds = [stats["Global"]["pre_std"]]
sez_stds = [stats["Global"]["sez_std"]]
bar_colors_pre = ["black"]
bar_colors_sez = ["gray"]

for c in range(CLUSTER_K):
    pre_means.append(stats[cluster_names[c]]["pre_mean"])
    sez_means.append(stats[cluster_names[c]]["sez_mean"])
    pre_stds.append(stats[cluster_names[c]]["pre_std"])
    sez_stds.append(stats[cluster_names[c]]["sez_std"])
    bar_colors_pre.append(CLUSTER_COLORS[c])
    bar_colors_sez.append(CLUSTER_COLORS[c])

x_pos = np.arange(len(names))
width = 0.35
bars_pre = ax.bar(x_pos - width / 2, pre_means, width, yerr=pre_stds,
                   color=bar_colors_pre, alpha=0.5, edgecolor="black", lw=0.5,
                   capsize=3, label="Pre-seizure")
bars_sez = ax.bar(x_pos + width / 2, sez_means, width, yerr=sez_stds,
                   color=bar_colors_sez, alpha=0.9, edgecolor="black", lw=0.5,
                   capsize=3, label="Seizure")
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_ylabel("Mean r")
ax.set_title("(A) Mean Order Parameter")
ax.legend(fontsize=8)

# 3B: Delta r at onset
ax = axes[1]
deltas = [bootstrap_results.get("global", {}).get("delta", 0)]
ci_los = [bootstrap_results.get("global", {}).get("ci_lo", 0)]
ci_his = [bootstrap_results.get("global", {}).get("ci_hi", 0)]
for c in range(CLUSTER_K):
    deltas.append(bootstrap_results[c]["delta"])
    ci_los.append(bootstrap_results[c]["ci_lo"])
    ci_his.append(bootstrap_results[c]["ci_hi"])

bar_colors = ["black"] + CLUSTER_COLORS[:CLUSTER_K]
err_lo = [d - l for d, l in zip(deltas, ci_los)]
err_hi = [h - d for d, h in zip(deltas, ci_his)]
ax.bar(x_pos, deltas, color=bar_colors, alpha=0.8, edgecolor="black", lw=0.5)
ax.errorbar(x_pos, deltas, yerr=[err_lo, err_hi], fmt="none", ecolor="black",
            capsize=4, lw=1.5)
ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_ylabel(r"$\Delta r$ (seizure $-$ pre-seizure)")
ax.set_title("(B) Change at Seizure Onset (95% CI)")

# 3C: PLV within vs between
ax = axes[2]
within_vals = [within_plvs_sez[c] for c in range(CLUSTER_K)]
# For between, compute mean between each cluster and all others
between_vals_per_cluster = []
for c in range(CLUSTER_K):
    btwn = []
    for ci, cj in between_plvs_sez:
        if ci == c or cj == c:
            btwn.append(between_plvs_sez[(ci, cj)])
    between_vals_per_cluster.append(np.mean(btwn) if btwn else 0)

x_pos_c = np.arange(CLUSTER_K)
names_c = [f"C{c}" for c in range(CLUSTER_K)]
width_c = 0.35
ax.bar(x_pos_c - width_c / 2, within_vals, width_c,
       color=[CLUSTER_COLORS[c] for c in range(CLUSTER_K)],
       alpha=0.9, edgecolor="black", lw=0.5, label="Within-cluster")
ax.bar(x_pos_c + width_c / 2, between_vals_per_cluster, width_c,
       color=[CLUSTER_COLORS[c] for c in range(CLUSTER_K)],
       alpha=0.3, edgecolor="black", lw=0.5, label="Between-cluster")
ax.set_xticks(x_pos_c)
ax.set_xticklabels(names_c)
ax.set_ylabel("Mean PLV")
ax.set_title("(C) Phase Locking Value (seizure)")
ax.legend(fontsize=8)

fig.suptitle("Cluster Synchronization Statistics", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, "order_parameter_stats")


# --- PLOT 4: Onset change detail ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 4A: Zoomed time series around onset
ax = axes[0]
zoom_start = 75.0
zoom_end = 100.0
zoom_idx = (t_axis >= zoom_start) & (t_axis <= zoom_end)

ax.plot(t_axis[zoom_idx], r_global_smooth[zoom_idx], color="black", lw=2,
        label="Global", alpha=0.8)
for c in range(CLUSTER_K):
    ax.plot(t_axis[zoom_idx], r_clusters_smooth[c][zoom_idx],
            color=CLUSTER_COLORS[c], lw=1.5, label=cluster_names[c], alpha=0.8)
ax.axvline(ONSET_TIME, color="red", ls="--", lw=2, alpha=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel("r(t) (smoothed)")
ax.set_title("(A) Order Parameter Near Seizure Onset")
ax.legend(fontsize=7, loc="upper left")
ax.set_ylim(0, 1)

# 4B: Bootstrap distributions of delta
ax = axes[1]
# Show violin/histogram for each cluster's bootstrap delta
positions = list(range(CLUSTER_K + 1))
bp_data = [bootstrap_results["global"]["boot_deltas"]]
bp_labels = ["Global"]
bp_colors = ["black"]
for c in range(CLUSTER_K):
    bp_data.append(bootstrap_results[c]["boot_deltas"])
    bp_labels.append(f"C{c}")
    bp_colors.append(CLUSTER_COLORS[c])

parts = ax.violinplot(bp_data, positions=positions, showmeans=True, showextrema=False)
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(bp_colors[i])
    pc.set_alpha(0.6)
parts["cmeans"].set_color("black")

# Overlay observed deltas
for i, d in enumerate([bootstrap_results["global"]["delta"]] +
                       [bootstrap_results[c]["delta"] for c in range(CLUSTER_K)]):
    ax.plot(i, d, "ko", markersize=8, zorder=5)

ax.axhline(0, color="gray", ls=":", lw=0.5)
ax.set_xticks(positions)
ax.set_xticklabels(bp_labels)
ax.set_ylabel(r"$\Delta r$ (seizure $-$ pre-seizure)")
ax.set_title("(B) Bootstrap Distribution of Onset Change")

fig.suptitle("Seizure Onset Synchronization Dynamics", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, "order_parameter_onset_change")


# --- PLOT 5: Permutation test visualizations ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 5A: Within vs between coherence permutation
ax = axes[0]
ax.hist(perm_diffs, bins=60, density=True, alpha=0.7, color="#1F77B4",
        edgecolor="gray", lw=0.5, label=f"Null (n={N_PERMUTATIONS})")
ax.axvline(observed_stat, color="#D62728", lw=2, ls="--",
           label=f"Observed = {observed_stat:.4f}")
ax.set_xlabel("Within minus Between PLV")
ax.set_ylabel("Density")
ax.set_title(f"(A) Cluster Coherence Test\np = {p_value_coherence:.4f}, "
             f"z = {z_score_coherence:.1f}")
ax.legend(fontsize=8)

# 5B: SOZ cluster onset change permutation
ax = axes[1]
ax.hist(perm_diff_deltas, bins=60, density=True, alpha=0.7, color="#1F77B4",
        edgecolor="gray", lw=0.5, label=f"Null (n={N_PERMUTATIONS})")
ax.axvline(observed_diff_delta, color="#D62728", lw=2, ls="--",
           label=f"Observed = {observed_diff_delta:+.4f}")
ax.axvline(-observed_diff_delta, color="#D62728", lw=1, ls=":", alpha=0.5)
ax.set_xlabel(r"$\Delta r_{\rm SOZ}$ minus mean $\Delta r_{\rm other}$")
ax.set_ylabel("Density")
ax.set_title(f"(B) SOZ-Specific Onset Change Test\np = {p_value_delta:.4f}, "
             f"z = {z_score_delta:.1f}")
ax.legend(fontsize=8)

fig.suptitle("Permutation Tests for Cluster Synchronization", fontsize=14, y=1.02)
fig.tight_layout()
save_fig(fig, "order_parameter_permutation_tests")


# --- PLOT 6: PLV matrix organized by cluster ---
fig, ax = plt.subplots(figsize=(10, 8))

# Sort channels by cluster
sorted_channels = []
cluster_boundaries = [0]
for c in range(CLUSTER_K):
    sorted_channels.extend(sorted(clusters[c]))
    cluster_boundaries.append(len(sorted_channels))

plv_sorted = plv_matrix_sez[np.ix_(sorted_channels, sorted_channels)]
im = ax.imshow(plv_sorted, cmap="hot", aspect="auto", vmin=0, vmax=0.5)
plt.colorbar(im, ax=ax, label="PLV", fraction=0.046, pad=0.04)

# Draw cluster boundaries
for b in cluster_boundaries[1:-1]:
    ax.axhline(b - 0.5, color="cyan", lw=1.5, ls="--")
    ax.axvline(b - 0.5, color="cyan", lw=1.5, ls="--")

# Label clusters
for c in range(CLUSTER_K):
    mid = (cluster_boundaries[c] + cluster_boundaries[c + 1]) / 2
    label = f"C{c}" + (" (SOZ)" if c == 0 else "")
    ax.text(-3, mid, label, ha="right", va="center", fontsize=9,
            color=CLUSTER_COLORS[c], fontweight="bold")

ax.set_title("Pairwise PLV Matrix (seizure window, channels sorted by cluster)")
ax.set_xlabel("Channel (sorted)")
ax.set_ylabel("Channel (sorted)")
fig.tight_layout()
save_fig(fig, "order_parameter_plv_matrix")


# ============================================================
# SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

print(f"\n{'Metric':<35s} {'Global':>10s}", end="")
for c in range(CLUSTER_K):
    print(f" {'C'+str(c)+(' (SOZ)' if c==0 else ''):>15s}", end="")
print()
print("-" * (35 + 10 + 15 * CLUSTER_K))

# Cluster size
print(f"{'N channels':<35s} {N_CHANNELS:>10d}", end="")
for c in range(CLUSTER_K):
    print(f" {len(clusters[c]):>15d}", end="")
print()

# N SOZ
print(f"{'N SOZ channels':<35s} {len(SOZ_CHANNELS):>10d}", end="")
for c in range(CLUSTER_K):
    n_soz = sum(1 for m in clusters[c] if m in SOZ_CHANNELS)
    print(f" {n_soz:>15d}", end="")
print()

# Pre-seizure r
print(f"{'Pre-seizure mean r':<35s} {stats['Global']['pre_mean']:>10.4f}", end="")
for c in range(CLUSTER_K):
    print(f" {stats[cluster_names[c]]['pre_mean']:>15.4f}", end="")
print()

# Seizure r
print(f"{'Seizure mean r':<35s} {stats['Global']['sez_mean']:>10.4f}", end="")
for c in range(CLUSTER_K):
    print(f" {stats[cluster_names[c]]['sez_mean']:>15.4f}", end="")
print()

# Delta r
print(f"{'Delta r (onset change)':<35s} {stats['Global']['delta']:>+10.4f}", end="")
for c in range(CLUSTER_K):
    print(f" {stats[cluster_names[c]]['delta']:>+15.4f}", end="")
print()

# 95% CI
ci_str_g = f"[{bootstrap_results['global']['ci_lo']:+.3f}, {bootstrap_results['global']['ci_hi']:+.3f}]"
print(f"{'95% CI':<35s} {ci_str_g:>10s}", end="")
for c in range(CLUSTER_K):
    ci_str = f"[{bootstrap_results[c]['ci_lo']:+.3f}, {bootstrap_results[c]['ci_hi']:+.3f}]"
    print(f" {ci_str:>15s}", end="")
print()

# Within-cluster PLV (seizure)
print(f"{'Within-cluster PLV (sez)':<35s} {'--':>10s}", end="")
for c in range(CLUSTER_K):
    print(f" {within_plvs_sez[c]:>15.4f}", end="")
print()

# Between-cluster PLV (seizure)
print(f"{'Between-cluster PLV (sez, mean)':<35s} {'--':>10s}", end="")
for c in range(CLUSTER_K):
    print(f" {between_vals_per_cluster[c]:>15.4f}", end="")
print()

print(f"\n  Permutation test (within > between PLV): p = {p_value_coherence:.4f}, z = {z_score_coherence:.1f}")
print(f"  Permutation test (SOZ onset diff): p = {p_value_delta:.4f}, z = {z_score_delta:.1f}")

print(f"\n  All plots saved to: {OUT_DIR}")
print(f"  Prefix: order_parameter_*")
print("\nDone.")
