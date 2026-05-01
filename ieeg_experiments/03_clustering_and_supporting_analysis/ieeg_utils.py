"""Shared utilities for iEEG Kuramoto analyses.

Common data loading, signal processing, and plotting functions used by:
- early_warning/early_warning_analysis.py
- envelope_phase/envelope_phase_kuramoto.py
- directed_connectivity/directed_connectivity_analysis.py
"""

import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, savgol_filter, butter, filtfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Constants
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "E3Data.mat"
FS = 500  # Hz

ONSET_TIMES = {1: 80.25, 2: 88.25, 3: 87.00}  # seconds

# 3 SOZ subclusters (0-indexed channels)
CLUSTERS = {
    0: [12, 15, 24, 28, 29, 30, 33],  # C0: SOZ core (7 ch)
    2: [25, 26],                        # C2: pacemaker (2 ch)
    3: [23, 27],                        # C3: boundary (2 ch)
}
CLUSTER_IDS = sorted(CLUSTERS.keys())  # [0, 2, 3]
CLUSTER_NAMES = {
    0: "C0 (SOZ core)",
    2: "C2 (pacemaker)",
    3: "C3 (boundary)",
}
CLUSTER_COLORS = {0: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}

ALPHA_BAND = (8.0, 13.0)

# Envelope processing defaults
SMOOTH_WIN_S = 4.0   # 4s running average for Hilbert envelope
SG_DERIV_WIN = 13    # Savitzky-Golay derivative window (at downsampled rate)
SG_DERIV_POLY = 4    # Savitzky-Golay derivative polynomial order


# ============================================================
# Data loading
# ============================================================
def load_episodes(data_path=DATA_PATH):
    """Load E3Data.mat and return dict {1: X1, 2: X2, 3: X3}.

    Each X_i is (n_samples, n_channels) float64.
    """
    mat = sio.loadmat(data_path)
    episodes = {}
    for ep in [1, 2, 3]:
        key = f"X{ep}"
        episodes[ep] = mat[key].astype(np.float64)
    print(f"Loaded episodes: " +
          ", ".join(f"X{ep}: {episodes[ep].shape}" for ep in [1, 2, 3]))
    return episodes


# ============================================================
# Signal processing
# ============================================================
def bandpass(x, lo, hi, fs, order=4):
    """Apply zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    b, a = butter(order, [max(lo / nyq, 0.001), min(hi / nyq, 0.999)], btype="band")
    return filtfilt(b, a, x, axis=0)


def extract_cluster_amplitudes(X_raw, clusters=None, band=None, fs=FS,
                                smooth_s=SMOOTH_WIN_S):
    """Extract per-cluster SVD mode 0 of log(Hilbert envelope).

    Pipeline per channel: bandpass -> Hilbert -> |analytic| -> smooth -> log(A+1)
    Then per cluster: SVD mode 0 of (T, n_ch) matrix.

    Parameters
    ----------
    X_raw : (n_samples, n_channels) array
    clusters : dict, default CLUSTERS
    band : (lo, hi) tuple, default ALPHA_BAND
    fs : sampling rate
    smooth_s : smoothing window in seconds

    Returns
    -------
    modes : dict {cluster_id: (n_samples,) amplitude mode 0}
    """
    if clusters is None:
        clusters = CLUSTERS
    if band is None:
        band = ALPHA_BAND
    band_lo, band_hi = band

    n_samples, n_ch = X_raw.shape
    smooth_samples = int(smooth_s * fs)
    smooth_kernel = np.ones(smooth_samples) / smooth_samples

    nyq = fs / 2.0
    b, a = butter(4, [max(band_lo / nyq, 0.001), min(band_hi / nyq, 0.999)],
                  btype="band")

    modes = {}
    for cid in sorted(clusters.keys()):
        ch_list = [ch for ch in clusters[cid] if ch < n_ch]
        if len(ch_list) == 0:
            modes[cid] = np.zeros(n_samples)
            continue

        amps = np.zeros((n_samples, len(ch_list)))
        for j, ch in enumerate(ch_list):
            sig_filt = filtfilt(b, a, X_raw[:, ch])
            analytic = hilbert(sig_filt)
            inst_amp = np.abs(analytic)
            amp_smooth = np.convolve(inst_amp, smooth_kernel, mode="same")
            amps[:, j] = np.log(amp_smooth + 1.0)

        # SVD: take mode 0
        amp_mean = amps.mean(axis=0)
        amp_std = amps.std(axis=0)
        amp_std[amp_std < 1e-10] = 1.0
        amps_centered = (amps - amp_mean) / amp_std

        if amps_centered.shape[1] == 1:
            modes[cid] = amps_centered[:, 0]
        else:
            U, S, Vt = np.linalg.svd(amps_centered, full_matrices=False)
            modes[cid] = U[:, 0] * S[0]

    return modes


def extract_cluster_phases(X_raw, clusters=None, band=None, fs=FS):
    """Extract per-cluster circular mean phase via Hilbert transform.

    Pipeline: bandpass -> Hilbert -> angle -> circular mean across cluster channels.

    Returns
    -------
    phases : dict {cluster_id: (n_samples,) circular mean phase}
    per_channel_phases : dict {cluster_id: (n_samples, n_ch) per-channel phases}
    """
    if clusters is None:
        clusters = CLUSTERS
    if band is None:
        band = ALPHA_BAND
    band_lo, band_hi = band

    n_samples, n_ch = X_raw.shape
    nyq = fs / 2.0
    b, a = butter(4, [max(band_lo / nyq, 0.001), min(band_hi / nyq, 0.999)],
                  btype="band")

    phases = {}
    per_channel_phases = {}
    for cid in sorted(clusters.keys()):
        ch_list = [ch for ch in clusters[cid] if ch < n_ch]
        if len(ch_list) == 0:
            phases[cid] = np.zeros(n_samples)
            per_channel_phases[cid] = np.zeros((n_samples, 0))
            continue

        ch_phases = np.zeros((n_samples, len(ch_list)))
        for j, ch in enumerate(ch_list):
            sig_filt = filtfilt(b, a, X_raw[:, ch])
            analytic = hilbert(sig_filt)
            ch_phases[:, j] = np.angle(analytic)

        per_channel_phases[cid] = ch_phases
        # Circular mean
        z = np.exp(1j * ch_phases).mean(axis=1)
        phases[cid] = np.angle(z)

    return phases, per_channel_phases


def savgol_derivative(x, dt, window=SG_DERIV_WIN, polyorder=SG_DERIV_POLY, deriv=1):
    """Savitzky-Golay derivative with proper dt scaling."""
    return savgol_filter(x, window, polyorder, deriv=deriv, delta=dt)


# ============================================================
# Plotting
# ============================================================
def setup_style():
    """Set publication-quality plot style."""
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


def save_fig(fig, filepath):
    """Save figure as PNG + PDF at 300 dpi.

    Parameters
    ----------
    fig : matplotlib Figure
    filepath : str or Path, without extension (e.g., 'early_warning/ews_variance')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(str(filepath) + f".{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {filepath.name}")
