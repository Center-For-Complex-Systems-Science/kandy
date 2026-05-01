"""
Differential Embedding Clustering of iEEG Channels
=====================================================
Builds (x, x_dot, x_ddot) embeddings for each channel, computes pairwise
covariance-based distances, and clusters channels to see whether the seizure
onset zone (SOZ) separates from non-SOZ channels.

Two windows are analysed:
  - Pre-seizure: 10 s window ending 2 s before onset
  - Seizure: 10 s window starting at onset

Similarity metric: Frobenius distance between 3x3 covariance matrices of the
differential embedding, plus log-Euclidean distance as a Riemannian alternative.

Clustering: Agglomerative (Ward) and Spectral, with k chosen by silhouette score.

Author: KANDy Researcher Agent
Date: 2026-03-25
"""

import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import MDS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
SOZ_CHANNELS = list(range(21, 31))  # 0-indexed: 21-30 inclusive

# Episode 3 seizure onset (we use Ep3 since X1 is a single episode of ~100s)
ONSET_SAMPLE = 43500  # 87.0 s
ONSET_TIME = 87.0

# Windows
PRE_DURATION = 10.0   # seconds
PRE_GAP = 2.0         # gap before onset
SEZ_DURATION = 10.0   # seconds

# Savitzky-Golay derivative parameters
SG_WINDOW = 51   # ~100 ms at 500 Hz (odd)
SG_POLY = 5

# Clustering
K_RANGE = range(2, 8)

# ============================================================
# Load data
# ============================================================
print("Loading data...")
mat = sio.loadmat(DATA_PATH)
X1 = mat["X1"]  # (49972, 120)
n_samples, n_ch = X1.shape
print(f"  X1 shape: {X1.shape}, duration: {n_samples/FS:.1f} s")

# ============================================================
# Define windows
# ============================================================
pre_end = int(ONSET_SAMPLE - PRE_GAP * FS)
pre_start = int(pre_end - PRE_DURATION * FS)
sez_start = ONSET_SAMPLE
sez_end = int(sez_start + SEZ_DURATION * FS)

# Clip to valid range
pre_start = max(0, pre_start)
sez_end = min(n_samples, sez_end)

print(f"  Pre-seizure window: samples {pre_start}-{pre_end} "
      f"({pre_start/FS:.1f}-{pre_end/FS:.1f} s)")
print(f"  Seizure window: samples {sez_start}-{sez_end} "
      f"({sez_start/FS:.1f}-{sez_end/FS:.1f} s)")

# ============================================================
# Compute differential embeddings
# ============================================================
def compute_embedding(data_segment, fs, sg_window=SG_WINDOW, sg_poly=SG_POLY):
    """
    Compute (x, x_dot, x_ddot) for each channel.

    Parameters
    ----------
    data_segment : (T, C) array
    fs : sampling rate

    Returns
    -------
    embeddings : dict mapping channel_idx -> (T, 3) array
    """
    T, C = data_segment.shape
    dt = 1.0 / fs
    embeddings = {}
    for ch in range(C):
        x = data_segment[:, ch].copy()
        # Normalise each channel to zero mean, unit variance for fair comparison
        mu, sigma = x.mean(), x.std()
        if sigma < 1e-12:
            sigma = 1.0
        x = (x - mu) / sigma
        # Savitzky-Golay smoothing + derivatives
        x_smooth = savgol_filter(x, sg_window, sg_poly, deriv=0)
        x_dot = savgol_filter(x, sg_window, sg_poly, deriv=1, delta=dt)
        x_ddot = savgol_filter(x, sg_window, sg_poly, deriv=2, delta=dt)
        embeddings[ch] = np.column_stack([x_smooth, x_dot, x_ddot])
    return embeddings


def compute_covariance_descriptors(embeddings):
    """
    Compute 3x3 covariance matrix for each channel's embedding.
    Returns dict ch -> (3, 3) covariance matrix.
    """
    covs = {}
    for ch, emb in embeddings.items():
        C = np.cov(emb, rowvar=False)
        # Regularise for numerical stability
        C += 1e-10 * np.eye(3)
        covs[ch] = C
    return covs


def frobenius_distance_matrix(covs, n_ch):
    """Pairwise Frobenius distance between covariance matrices."""
    D = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            d = np.linalg.norm(covs[i] - covs[j], "fro")
            D[i, j] = d
            D[j, i] = d
    return D


def log_euclidean_distance_matrix(covs, n_ch):
    """Pairwise Log-Euclidean distance between SPD covariance matrices."""
    log_covs = {}
    for ch, C in covs.items():
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-10)
        log_covs[ch] = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
    D = np.zeros((n_ch, n_ch))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            d = np.linalg.norm(log_covs[i] - log_covs[j], "fro")
            D[i, j] = d
            D[j, i] = d
    return D


def cluster_and_evaluate(D, n_ch, soz_mask, k_range=K_RANGE):
    """
    Try agglomerative (Ward) and spectral clustering for each k.
    Return best configuration by silhouette score.
    """
    results = []

    # Convert distance to affinity for spectral clustering
    sigma = np.median(D[D > 0])
    affinity = np.exp(-D**2 / (2 * sigma**2))

    for k in k_range:
        # Ward (needs condensed distance)
        D_safe = D.copy()
        np.fill_diagonal(D_safe, 0)
        Z = linkage(squareform(D_safe), method="ward")
        labels_ward = fcluster(Z, k, criterion="maxclust") - 1
        sil_ward = silhouette_score(D, labels_ward, metric="precomputed")
        ari_ward = adjusted_rand_score(soz_mask, labels_ward)
        nmi_ward = normalized_mutual_info_score(soz_mask, labels_ward)

        # Spectral
        sc = SpectralClustering(
            n_clusters=k, affinity="precomputed", random_state=42,
            assign_labels="kmeans"
        )
        labels_spec = sc.fit_predict(affinity)
        sil_spec = silhouette_score(D, labels_spec, metric="precomputed")
        ari_spec = adjusted_rand_score(soz_mask, labels_spec)
        nmi_spec = normalized_mutual_info_score(soz_mask, labels_spec)

        results.append({
            "k": k, "method": "Ward", "labels": labels_ward,
            "silhouette": sil_ward, "ARI": ari_ward, "NMI": nmi_ward,
            "linkage": Z,
        })
        results.append({
            "k": k, "method": "Spectral", "labels": labels_spec,
            "silhouette": sil_spec, "ARI": ari_spec, "NMI": nmi_spec,
        })

    # Best by silhouette
    best = max(results, key=lambda r: r["silhouette"])
    return results, best


# ============================================================
# Run analysis for both windows
# ============================================================
soz_mask = np.array([1 if ch in SOZ_CHANNELS else 0 for ch in range(N_CHANNELS)])

windows = {
    "Pre-seizure": X1[pre_start:pre_end, :],
    "Seizure": X1[sez_start:sez_end, :],
}

all_results = {}
for win_name, data_seg in windows.items():
    print(f"\n{'='*60}")
    print(f"Window: {win_name} ({data_seg.shape[0]} samples, {data_seg.shape[0]/FS:.1f} s)")
    print(f"{'='*60}")

    # 1. Compute embeddings
    embs = compute_embedding(data_seg, FS)

    # 2. Covariance descriptors
    covs = compute_covariance_descriptors(embs)

    # 3. Distance matrices
    D_frob = frobenius_distance_matrix(covs, N_CHANNELS)
    D_logeuc = log_euclidean_distance_matrix(covs, N_CHANNELS)

    # 4. Cluster and evaluate
    for dist_name, D in [("Frobenius", D_frob), ("LogEuclidean", D_logeuc)]:
        print(f"\n  Distance: {dist_name}")
        results, best = cluster_and_evaluate(D, N_CHANNELS, soz_mask)

        key = f"{win_name}_{dist_name}"
        all_results[key] = {
            "results": results,
            "best": best,
            "D": D,
            "covs": covs,
            "embs": embs,
        }

        print(f"    Best: k={best['k']}, method={best['method']}, "
              f"silhouette={best['silhouette']:.3f}, "
              f"ARI={best['ARI']:.3f}, NMI={best['NMI']:.3f}")

        # SOZ cluster purity
        soz_labels = best["labels"][SOZ_CHANNELS]
        dominant_cluster = np.bincount(soz_labels).argmax()
        soz_purity = np.mean(soz_labels == dominant_cluster)
        print(f"    SOZ purity (fraction in dominant cluster): {soz_purity:.2f}")
        print(f"    SOZ dominant cluster: {dominant_cluster}")
        non_soz_in_dominant = np.sum(
            (best["labels"] == dominant_cluster) & (soz_mask == 0)
        )
        print(f"    Non-SOZ channels in SOZ cluster: {non_soz_in_dominant}")

        # Print all k results
        print(f"\n    {'k':>3} {'Method':>10} {'Silh':>7} {'ARI':>7} {'NMI':>7}")
        for r in sorted(results, key=lambda r: (r["k"], r["method"])):
            print(f"    {r['k']:>3} {r['method']:>10} "
                  f"{r['silhouette']:>7.3f} {r['ARI']:>7.3f} {r['NMI']:>7.3f}")


# ============================================================
# PLOTS
# ============================================================
print("\n\nGenerating plots...")


def save_fig(fig, name):
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# Colour maps
soz_colors = ["#D62728" if ch in SOZ_CHANNELS else "#1F77B4" for ch in range(N_CHANNELS)]
soz_cmap = {0: "#1F77B4", 1: "#D62728"}  # blue=non-SOZ, red=SOZ

# ============================================================
# 1. Example differential embeddings (3D) for SOZ vs non-SOZ
# ============================================================
fig = plt.figure(figsize=(14, 6))

for idx, (win_name, data_seg) in enumerate(windows.items()):
    embs = compute_embedding(data_seg, FS)
    ax = fig.add_subplot(1, 2, idx + 1, projection="3d")

    # Plot a few non-SOZ channels in blue
    for ch in [0, 5, 50, 80, 110]:
        e = embs[ch]
        ax.plot(e[::5, 0], e[::5, 1], e[::5, 2],
                color="#1F77B4", alpha=0.15, lw=0.5)

    # Plot SOZ channels in red
    for ch in SOZ_CHANNELS:
        e = embs[ch]
        ax.plot(e[::5, 0], e[::5, 1], e[::5, 2],
                color="#D62728", alpha=0.4, lw=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel(r"$\dot{x}$")
    ax.set_zlabel(r"$\ddot{x}$")
    ax.set_title(f"{win_name}")
    ax.view_init(elev=25, azim=45)

fig.suptitle("Differential Embeddings: SOZ (red) vs Non-SOZ (blue)", y=1.02)
fig.tight_layout()
save_fig(fig, "differential_embeddings_3d")


# ============================================================
# 2. Distance matrices (heatmaps) for each window x metric
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for col, dist_name in enumerate(["Frobenius", "LogEuclidean"]):
    for row, win_name in enumerate(["Pre-seizure", "Seizure"]):
        key = f"{win_name}_{dist_name}"
        D = all_results[key]["D"]
        ax = axes[row, col]

        im = ax.imshow(D, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Highlight SOZ channels
        for ch in SOZ_CHANNELS:
            ax.axhline(ch, color="red", alpha=0.3, lw=0.5)
            ax.axvline(ch, color="red", alpha=0.3, lw=0.5)

        ax.set_title(f"{win_name} -- {dist_name}")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Channel")

fig.suptitle("Pairwise Distance Matrices (red lines = SOZ channels 21-30)", y=1.01)
fig.tight_layout()
save_fig(fig, "distance_matrices")


# ============================================================
# 3. MDS embedding coloured by SOZ membership + cluster labels
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for col, win_name in enumerate(["Pre-seizure", "Seizure"]):
    for row, dist_name in enumerate(["Frobenius", "LogEuclidean"]):
        key = f"{win_name}_{dist_name}"
        D = all_results[key]["D"]
        best = all_results[key]["best"]

        # MDS for 2D embedding
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42,
                  normalized_stress="auto")
        coords = mds.fit_transform(D)

        ax = axes[row, col]

        # Plot non-SOZ
        non_soz = [ch for ch in range(N_CHANNELS) if ch not in SOZ_CHANNELS]
        ax.scatter(coords[non_soz, 0], coords[non_soz, 1],
                  c=[best["labels"][ch] for ch in non_soz],
                  cmap="Set2", s=30, alpha=0.6, edgecolors="gray", lw=0.3,
                  marker="o")

        # Plot SOZ with stars
        ax.scatter(coords[SOZ_CHANNELS, 0], coords[SOZ_CHANNELS, 1],
                  c=[best["labels"][ch] for ch in SOZ_CHANNELS],
                  cmap="Set2", s=150, alpha=0.9, edgecolors="red", lw=1.5,
                  marker="*", zorder=5)

        # Label SOZ channels
        for ch in SOZ_CHANNELS:
            ax.annotate(str(ch), (coords[ch, 0], coords[ch, 1]),
                       fontsize=6, ha="center", va="bottom",
                       color="red", fontweight="bold")

        ax.set_title(f"{win_name} -- {dist_name}\n"
                    f"Best: k={best['k']}, {best['method']}, "
                    f"Sil={best['silhouette']:.3f}")
        ax.set_xlabel("MDS dim 1")
        ax.set_ylabel("MDS dim 2")

# Legend
star_patch = plt.Line2D([0], [0], marker="*", color="w", markeredgecolor="red",
                        markerfacecolor="gray", markersize=12, label="SOZ channels")
circle_patch = plt.Line2D([0], [0], marker="o", color="w", markeredgecolor="gray",
                          markerfacecolor="gray", markersize=8, label="Non-SOZ channels")
fig.legend(handles=[star_patch, circle_patch], loc="upper center", ncol=2,
          bbox_to_anchor=(0.5, 1.03))
fig.suptitle("MDS Embedding with Cluster Labels (stars = SOZ)", y=1.06)
fig.tight_layout()
save_fig(fig, "mds_clustering")


# ============================================================
# 4. Dendrograms with SOZ highlighted
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for idx, win_name in enumerate(["Pre-seizure", "Seizure"]):
    key = f"{win_name}_Frobenius"
    D = all_results[key]["D"]

    D_safe = D.copy()
    np.fill_diagonal(D_safe, 0)
    Z = linkage(squareform(D_safe), method="ward")

    ax = axes[idx]

    # Custom label colours
    def leaf_label_func(ch):
        if ch in SOZ_CHANNELS:
            return f"*{ch}*"
        return str(ch)

    dn = dendrogram(
        Z, ax=ax, leaf_rotation=90, leaf_font_size=5,
        labels=[leaf_label_func(ch) for ch in range(N_CHANNELS)],
        color_threshold=0,
        above_threshold_color="gray",
    )

    # Colour SOZ leaf labels red
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        txt = lbl.get_text()
        if txt.startswith("*") and txt.endswith("*"):
            lbl.set_color("red")
            lbl.set_fontweight("bold")

    ax.set_title(f"{win_name} (Frobenius, Ward)")
    ax.set_ylabel("Distance")

fig.suptitle("Hierarchical Clustering Dendrograms (red labels = SOZ)", y=1.02)
fig.tight_layout()
save_fig(fig, "dendrograms")


# ============================================================
# 5. Cluster composition bar chart
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for col, win_name in enumerate(["Pre-seizure", "Seizure"]):
    for row, dist_name in enumerate(["Frobenius", "LogEuclidean"]):
        key = f"{win_name}_{dist_name}"
        best = all_results[key]["best"]
        labels = best["labels"]
        k = best["k"]

        ax = axes[row, col]

        soz_counts = []
        non_soz_counts = []
        for c in range(k):
            members = np.where(labels == c)[0]
            n_soz = sum(1 for m in members if m in SOZ_CHANNELS)
            n_nonsoz = len(members) - n_soz
            soz_counts.append(n_soz)
            non_soz_counts.append(n_nonsoz)

        x_pos = np.arange(k)
        width = 0.35
        ax.bar(x_pos - width/2, non_soz_counts, width, label="Non-SOZ",
               color="#1F77B4", alpha=0.8)
        ax.bar(x_pos + width/2, soz_counts, width, label="SOZ",
               color="#D62728", alpha=0.8)

        ax.set_xlabel("Cluster")
        ax.set_ylabel("Number of channels")
        ax.set_title(f"{win_name} -- {dist_name} (k={k})")
        ax.set_xticks(x_pos)
        ax.legend()

fig.suptitle("Cluster Composition: SOZ vs Non-SOZ", y=1.02)
fig.tight_layout()
save_fig(fig, "cluster_composition")


# ============================================================
# 6. SOZ separation score across k values
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, win_name in enumerate(["Pre-seizure", "Seizure"]):
    ax = axes[idx]

    for dist_name, ls in [("Frobenius", "-"), ("LogEuclidean", "--")]:
        key = f"{win_name}_{dist_name}"
        res_list = all_results[key]["results"]

        for method, marker in [("Ward", "o"), ("Spectral", "s")]:
            ks = []
            aris = []
            nmis = []
            sils = []
            for r in res_list:
                if r["method"] == method:
                    ks.append(r["k"])
                    aris.append(r["ARI"])
                    nmis.append(r["NMI"])
                    sils.append(r["silhouette"])

            label = f"{dist_name[:4]}/{method}"
            ax.plot(ks, aris, ls=ls, marker=marker, label=f"{label} ARI",
                   alpha=0.7)

    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Adjusted Rand Index (vs SOZ labels)")
    ax.set_title(f"{win_name}")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xticks(list(K_RANGE))
    ax.axhline(0, color="gray", ls=":", lw=0.5)

fig.suptitle("SOZ Separation: ARI vs k", y=1.02)
fig.tight_layout()
save_fig(fig, "ari_vs_k")


# ============================================================
# 7. Covariance eigenvalue spectra for SOZ vs non-SOZ
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for idx, (win_name, data_seg) in enumerate(windows.items()):
    key = f"{win_name}_Frobenius"
    covs = all_results[key]["covs"]

    ax = axes[idx]

    # SOZ eigenvalues
    soz_eigvals = []
    for ch in SOZ_CHANNELS:
        eigvals = np.sort(np.linalg.eigvalsh(covs[ch]))[::-1]
        soz_eigvals.append(eigvals)
    soz_eigvals = np.array(soz_eigvals)

    # Non-SOZ eigenvalues
    non_soz_eigvals = []
    for ch in range(N_CHANNELS):
        if ch not in SOZ_CHANNELS:
            eigvals = np.sort(np.linalg.eigvalsh(covs[ch]))[::-1]
            non_soz_eigvals.append(eigvals)
    non_soz_eigvals = np.array(non_soz_eigvals)

    # Plot means + std
    ax.errorbar([1, 2, 3], soz_eigvals.mean(axis=0), yerr=soz_eigvals.std(axis=0),
                fmt="o-", color="#D62728", label="SOZ", capsize=4, lw=2)
    ax.errorbar([1, 2, 3], non_soz_eigvals.mean(axis=0), yerr=non_soz_eigvals.std(axis=0),
                fmt="s-", color="#1F77B4", label="Non-SOZ", capsize=4, lw=2)

    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{win_name}")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"])
    ax.legend()
    ax.set_yscale("log")

fig.suptitle("Covariance Eigenvalue Spectra: SOZ vs Non-SOZ", y=1.02)
fig.tight_layout()
save_fig(fig, "eigenvalue_spectra")


# ============================================================
# 8. Average within-group vs between-group distances
# ============================================================
print("\n\n" + "="*60)
print("QUANTITATIVE SEPARATION ANALYSIS")
print("="*60)

for win_name in ["Pre-seizure", "Seizure"]:
    for dist_name in ["Frobenius", "LogEuclidean"]:
        key = f"{win_name}_{dist_name}"
        D = all_results[key]["D"]

        soz_idx = np.array(SOZ_CHANNELS)
        non_soz_idx = np.array([ch for ch in range(N_CHANNELS) if ch not in SOZ_CHANNELS])

        # Within-SOZ distances
        d_within_soz = []
        for i in range(len(soz_idx)):
            for j in range(i + 1, len(soz_idx)):
                d_within_soz.append(D[soz_idx[i], soz_idx[j]])

        # Within-non-SOZ distances
        d_within_nonsoz = []
        for i in range(len(non_soz_idx)):
            for j in range(i + 1, len(non_soz_idx)):
                d_within_nonsoz.append(D[non_soz_idx[i], non_soz_idx[j]])

        # Between-group distances
        d_between = []
        for i in soz_idx:
            for j in non_soz_idx:
                d_between.append(D[i, j])

        d_within_soz = np.array(d_within_soz)
        d_within_nonsoz = np.array(d_within_nonsoz)
        d_between = np.array(d_between)

        # Separation ratio: between / max(within_soz, within_nonsoz)
        sep_ratio = d_between.mean() / max(d_within_soz.mean(), d_within_nonsoz.mean())

        print(f"\n  {win_name} -- {dist_name}:")
        print(f"    Within-SOZ mean distance:     {d_within_soz.mean():.4f} +/- {d_within_soz.std():.4f}")
        print(f"    Within-non-SOZ mean distance:  {d_within_nonsoz.mean():.4f} +/- {d_within_nonsoz.std():.4f}")
        print(f"    Between-group mean distance:   {d_between.mean():.4f} +/- {d_between.std():.4f}")
        print(f"    Separation ratio (between/within): {sep_ratio:.3f}")
        print(f"    SOZ is {'MORE' if d_within_soz.mean() < d_within_nonsoz.mean() else 'LESS'} "
              f"tightly clustered than non-SOZ")


# ============================================================
# 9. Channel-level distance-to-SOZ-centroid
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, win_name in enumerate(["Pre-seizure", "Seizure"]):
    key = f"{win_name}_Frobenius"
    D = all_results[key]["D"]

    # Mean distance of each channel to all SOZ channels
    soz_idx = np.array(SOZ_CHANNELS)
    dist_to_soz = D[:, soz_idx].mean(axis=1)

    ax = axes[idx]

    colors = ["#D62728" if ch in SOZ_CHANNELS else "#1F77B4" for ch in range(N_CHANNELS)]
    ax.bar(range(N_CHANNELS), dist_to_soz, color=colors, alpha=0.7, width=1.0)

    ax.axvspan(SOZ_CHANNELS[0] - 0.5, SOZ_CHANNELS[-1] + 0.5, alpha=0.1, color="red",
               label="SOZ region")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Mean Frobenius distance to SOZ")
    ax.set_title(f"{win_name}")
    ax.legend()

fig.suptitle("Distance to SOZ Centroid per Channel (red = SOZ channels)", y=1.02)
fig.tight_layout()
save_fig(fig, "distance_to_soz")


# ============================================================
# 10. Pre-seizure vs Seizure: change in clustering
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# For each channel, compute how its distance to SOZ changes from pre -> seizure
key_pre = "Pre-seizure_Frobenius"
key_sez = "Seizure_Frobenius"
D_pre = all_results[key_pre]["D"]
D_sez = all_results[key_sez]["D"]

soz_idx = np.array(SOZ_CHANNELS)
dist_to_soz_pre = D_pre[:, soz_idx].mean(axis=1)
dist_to_soz_sez = D_sez[:, soz_idx].mean(axis=1)

# Normalise for comparison
dist_pre_norm = dist_to_soz_pre / dist_to_soz_pre.max()
dist_sez_norm = dist_to_soz_sez / dist_to_soz_sez.max()

delta = dist_sez_norm - dist_pre_norm  # positive = moved AWAY from SOZ

colors = ["#D62728" if ch in SOZ_CHANNELS else "#1F77B4" for ch in range(N_CHANNELS)]
ax.bar(range(N_CHANNELS), delta, color=colors, alpha=0.7, width=1.0)
ax.axhline(0, color="black", lw=0.5)
ax.axvspan(SOZ_CHANNELS[0] - 0.5, SOZ_CHANNELS[-1] + 0.5, alpha=0.1, color="red",
           label="SOZ region")
ax.set_xlabel("Channel")
ax.set_ylabel(r"$\Delta$ normalised distance to SOZ (seizure $-$ pre-seizure)")
ax.set_title("Change in Distance to SOZ: Seizure vs Pre-seizure")
ax.legend()
fig.tight_layout()
save_fig(fig, "delta_distance_to_soz")


# ============================================================
# Summary
# ============================================================
print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)

for win_name in ["Pre-seizure", "Seizure"]:
    for dist_name in ["Frobenius", "LogEuclidean"]:
        key = f"{win_name}_{dist_name}"
        best = all_results[key]["best"]
        soz_labels = best["labels"][SOZ_CHANNELS]
        dominant = np.bincount(soz_labels).argmax()
        purity = np.mean(soz_labels == dominant)
        nonsoz_in_dom = np.sum((best["labels"] == dominant) & (soz_mask == 0))

        print(f"\n  {win_name} / {dist_name}:")
        print(f"    Best clustering: k={best['k']}, {best['method']}")
        print(f"    Silhouette: {best['silhouette']:.3f}")
        print(f"    ARI vs SOZ: {best['ARI']:.3f}")
        print(f"    NMI vs SOZ: {best['NMI']:.3f}")
        print(f"    SOZ purity: {purity:.2f} ({int(purity*10)}/10 in cluster {dominant})")
        print(f"    Non-SOZ in SOZ cluster: {nonsoz_in_dom}/{N_CHANNELS - len(SOZ_CHANNELS)}")

print(f"\n\nAll plots saved to: {OUT_DIR}")
print("Done.")
