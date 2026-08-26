"""
Hamamatsu streak camera — spectral scan analysis (core engine)
----------------------------------------------------------------
Adapted from the timing-analysis tool for images with:
    vertical axis   = time  (as before, ns)
    horizontal axis = wavelength (nm), calibrated via the streak
                       software's embedded per-pixel calibration LUT

Key difference from the original space-axis parser: these files use
ScalingXType=2 / ScalingYType=2 ("calibration file" mode), meaning the
axis is NOT linspace(0,range,N) or (arange(N)+offset)*scale — it is a
literal float32 LUT stored at a byte offset inside the .img file itself.
Treating it as linear silently gives a wrong wavelength axis (and,
per the sample file, a wrong absolute time axis too — the calibrated
time range does not start at 0 and is not identical to the header's
nominal "Time Range").

Folder / metadata convention (confirm before relying on it):
    scan_folder/<shot_id>/<shot_id>.img      one .img per shot subfolder
    scan_params.csv  columns: shot_id, <param_col>   (delay, pressure, ...)

Physics / algorithm summary
----------------------------
1. Axis calibration  : read embedded LUT when ScalingType==2, else fall
                        back to the linear (scale, offset) formula.
2. Background         : per-row low-percentile baseline subtraction
                        (default 10th pct), clipped at 0. Assumes each
                        time row has >=10% of its wavelength channels
                        off any spectral line — true here since the
                        lines are ~50-100 nm wide within a 316 nm window.
3. Line identification: time-integrated spectrum I(lambda); peaks found
                        with scipy.signal.find_peaks (prominence-based);
                        the two most prominent are kept and *relabeled
                        by ascending wavelength* (peak1 = bluer,
                        peak2 = redder) so line identity stays stable
                        across a scan even if relative intensity flips.
                        Window bounds = local minima (peak bases), not
                        an arbitrary relative-height cut, so windows
                        stay adjacent to line morphology instead of an
                        arbitrary fraction of peak height.
4. Time evolution     : adaptive-recentering intensity centroid.
                        Seeded at the time row with best window SNR,
                        walked outward in both time directions; window
                        center is only updated on rows that clear an
                        SNR threshold, otherwise that row is recorded
                        as NaN (never fabricated/interpolated) and the
                        window holds its last valid position. This is
                        deliberately more conservative than a fixed
                        window over the whole streak, which fails once
                        drift/shift approaches the window edge.

Stated validity limits:
  - Assumes each of the two lines is not multi-modal within its window;
    if the two lines merge (bases overlap) at some point in time, the
    centroid will be biased toward the stronger line. No deconvolution
    (e.g. two-Gaussian fit) is attempted here — flag such shots for
    manual review rather than silently reporting a contaminated value.
  - SNR gating is per-row, based on local (in-window) noise in a
    background region of that same row — not a single global noise
    figure — because the continuum background can trend with time.
"""

from __future__ import annotations

import os
import re
import struct
import glob
import json
import dataclasses
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.ndimage import uniform_filter1d


# ─────────────────────────────────────────────────────────────────────────────
# 1. Hamamatsu reader with LUT-aware axis calibration
# ─────────────────────────────────────────────────────────────────────────────

def _read_scaling_axis(header_text: str, raw: bytes, axis_letter: str, n_expected: int):
    """Read one Scaling{X,Y} axis: LUT (Type=2) if present, else linear."""
    type_m = re.search(rf'Scaling{axis_letter}Type=(\d+)', header_text)
    stype = int(type_m.group(1)) if type_m else 1

    unit_m = re.search(rf'Scaling{axis_letter}Unit="([^"]*)"', header_text)
    unit = unit_m.group(1) if (unit_m and unit_m.group(1)) else "px"

    scale_m = re.search(rf'Scaling{axis_letter}Scale=([\d.eE+\-]+)', header_text)
    scale = float(scale_m.group(1)) if scale_m else 1.0

    if stype == 2:
        # "+OFFSET" -> count implied = n_expected ; "#OFFSET,COUNT" -> explicit count
        m_hash = re.search(rf'Scaling{axis_letter}ScalingFile="?#(\d+),(\d+)"?', header_text)
        m_plus = re.search(rf'Scaling{axis_letter}ScalingFile="?\+(\d+)"?', header_text)
        off = count = None
        if m_hash:
            off, count = int(m_hash.group(1)), int(m_hash.group(2))
        elif m_plus:
            off, count = int(m_plus.group(1)), n_expected
        if off is not None:
            axis = np.array(struct.unpack_from(f"<{count}f", raw, off), dtype=np.float64)
            return axis, unit, stype

    # linear fallback (Type==1, or Type==2 with unparsable ScalingFile spec)
    axis = np.arange(n_expected, dtype=np.float64) * scale
    return axis, unit, stype


def parse_hamamatsu_spectral(filepath: str):
    """
    Parse a Hamamatsu .img streak file with a spectral (wavelength) X axis.

    Returns
    -------
    data : (h, w) float32 array, time x wavelength, half-swapped exactly as
           the acquisition software's own display convention (see note below).
    meta : dict with 'lambda_axis' (nm), 'time_axis' (ns), units, raw header.

    Note on the half-swap: the raw sensor readout is column-swapped
    (data[:, w//2:] concatenated with data[:, :w//2]) — this is a hardware
    readout-order correction independent of what physical quantity the
    X axis represents. Verified against the sample file that the
    calibration LUT is already expressed in the POST-swap column order
    (the wavelength LUT is strictly monotonic across the full width with
    no discontinuity at the swap boundary) — so the LUT must NOT be
    swapped again; only the data is.
    """
    with open(filepath, "rb") as f:
        raw = f.read()
    w, h = struct.unpack_from("<HH", raw, 4)
    if w == 0 or h == 0:
        raise ValueError(f"Bad dimensions w={w} h={h}")
    header_size = len(raw) - w * h * 2
    if header_size < 0:
        raise ValueError("File too small")
    header_text = raw[64:header_size].decode("ascii", errors="ignore")

    lam_axis, lam_unit, lam_type = _read_scaling_axis(header_text, raw, "X", w)
    t_axis, t_unit, t_type = _read_scaling_axis(header_text, raw, "Y", h)

    meta = {
        "width": w, "height": h,
        "lambda_axis": lam_axis, "lambda_unit": lam_unit, "lambda_scaling_type": lam_type,
        "time_axis": t_axis, "time_unit": t_unit, "time_scaling_type": t_type,
        "header_text": header_text,
    }

    data = np.frombuffer(raw[header_size:], dtype=np.uint16).reshape((h, w))
    half = w // 2
    data = np.concatenate([data[:, half:], data[:, :half]], axis=1)
    return data.astype(np.float32), meta


# ─────────────────────────────────────────────────────────────────────────────
# 2. Background subtraction
# ─────────────────────────────────────────────────────────────────────────────

def apply_gain_nd_correction(data: np.ndarray, gain: float = 1.0, nd_od: float = 0.0) -> np.ndarray:
    """
    Back-correct raw ADC counts to an absolute/no-filter-equivalent scale,
    so intensities are comparable across shots taken with different MCP
    gain settings and/or ND filters in front of the streak.

    correction: data_abs = data * 10**nd_od / gain

    ND physics is unambiguous: a filter of optical density OD transmits
    10**(-OD) of incident light, so dividing by that (= multiplying by
    10**(+OD)) recovers the pre-filter intensity. nd_od=0 (no filter,
    default) is a no-op.

    Gain is NOT assumed to have a known nonlinear response here — this
    treats `gain` as a plain linear divisor. That is only correct if your
    shotbook's gain column already stores an actual multiplication factor.
    Hamamatsu MCP gain-vs-setting curves are typically nonlinear
    (roughly exponential over much of their range) — if your shotbook
    stores the raw camera setting (e.g. an integer 0-63) rather than a
    calibrated multiplication factor, do NOT pass it straight through:
    convert it first via your camera's own gain calibration curve (from
    its manual/calibration sheet), since I don't have that curve for
    your specific tube and won't assume one.
    """
    return data * (10.0 ** nd_od) / max(gain, 1e-12)


def subtract_row_background(data: np.ndarray, pct: float = 10.0) -> np.ndarray:
    """Per-row low-percentile baseline subtraction, clipped at 0."""
    baseline = np.percentile(data, pct, axis=1, keepdims=True)
    return np.clip(data - baseline, 0, None)


def detect_artifact_rows(data: np.ndarray, mad_factor: float = 8.0) -> np.ndarray:
    """
    Flag rows whose mean level is a robust outlier vs the rest of the shot —
    e.g. end-of-sweep saturation/blooming on the streak tube/MCP, seen on the
    real sample file (last 4 of 968 rows: mean 13k-25k counts vs a normal
    peak-signal row's ~450, with pixels clipped at the 16-bit ceiling).
    These rows are broadband (not a narrow line) and, if not excluded,
    corrupt centroid tracking: a saturated near-flat row has artificially
    LOW row-noise, which makes the SNR gate falsely pass it with high
    confidence, so it must be screened out before tracking, not caught by
    the SNR threshold itself.

    mad_factor=8 (robust z-score cutoff) is generous — verified to isolate
    exactly the true artifact rows on the sample file with a wide margin
    (thresh sat well above the 99.5th percentile of legitimate rows).
    Bit-depth-agnostic (percentile/MAD-based, not a hardcoded ADC ceiling).
    """
    row_mean = data.mean(axis=1)
    med = np.median(row_mean)
    robust_std = 1.4826 * np.median(np.abs(row_mean - med))
    thresh = med + mad_factor * max(robust_std, 1e-6)
    return row_mean > thresh


# ─────────────────────────────────────────────────────────────────────────────
# 3. Time-integrated spectrum + two-peak detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class PeakWindow:
    label: str          # "peak1" (bluer) or "peak2" (redder)
    center_idx: int
    center_nm: float
    left_idx: int
    right_idx: int
    prominence: float


def detect_two_peaks(lam_axis: np.ndarray, spectrum: np.ndarray,
                      smooth_px: int = 5, prominence_frac: float = 0.05,
                      min_separation_nm: float = 10.0,
                      width_rel_height: float = 0.85) -> list:
    """
    Find up to 2 dominant peaks in the time-integrated spectrum.

    Window bounds use scipy.signal.peak_widths at width_rel_height (width
    of the peak at a fraction of its prominence-referenced height) — NOT
    find_peaks' left_bases/right_bases. Those bases are a topographic
    quantity (they extend until a HIGHER neighboring peak is found, for
    prominence bookkeeping) and on a noisy or multi-featured spectrum can
    pathologically span nearly the entire array — confirmed on real data:
    left_base/right_base gave ~640 px windows (half the whole array) for
    both lines here, which is not a usable window for centroid tracking.
    peak_widths gives a bounded width that actually scales with the line
    shape. width_rel_height=0.85 is generous (captures most of a
    quasi-Gaussian line's wings without reaching into a neighboring
    feature) but is a tunable assumption — narrow it if two lines sit
    close enough that their wings would overlap at this height fraction.

    Returns peaks sorted by ascending wavelength, labeled peak1/peak2.
    Returns fewer than 2 entries if fewer peaks clear the threshold —
    caller must handle / flag this (do not assume exactly 2).
    """
    spec_s = uniform_filter1d(spectrum, size=max(1, smooth_px))
    dlam = np.median(np.diff(lam_axis))
    min_dist_px = max(1, int(round(min_separation_nm / abs(dlam))))

    peaks, props = find_peaks(
        spec_s,
        prominence=max(spec_s.max() * prominence_frac, 1e-9),
        distance=min_dist_px,
    )
    if len(peaks) == 0:
        return []

    order = np.argsort(props["prominences"])[::-1]
    top = peaks[order[:2]]
    top_prom_idx = order[:2]

    widths, width_heights, left_ips, right_ips = peak_widths(
        spec_s, top, rel_height=width_rel_height)

    result = []
    for k, (p, i) in enumerate(zip(top, top_prom_idx)):
        left_idx = int(np.clip(np.floor(left_ips[k]), 0, len(lam_axis) - 1))
        right_idx = int(np.clip(np.ceil(right_ips[k]), 0, len(lam_axis) - 1))
        result.append({
            "center_idx": int(p),
            "center_nm": float(lam_axis[p]),
            "left_idx": left_idx,
            "right_idx": right_idx,
            "prominence": float(props["prominences"][i]),
        })
    result.sort(key=lambda d: d["center_nm"])

    # if the two windows overlap (lines closer than their combined widths),
    # clip them to a shared midpoint rather than letting them double-count
    # the same pixels in two different centroid calculations
    if len(result) == 2 and result[0]["right_idx"] > result[1]["left_idx"]:
        mid = (result[0]["center_idx"] + result[1]["center_idx"]) // 2
        result[0]["right_idx"] = mid
        result[1]["left_idx"] = mid

    labels = ["peak1", "peak2"] if len(result) == 2 else ["peak1"]
    out = [PeakWindow(label=lab, **r) for lab, r in zip(labels, result)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Adaptive per-row centroid tracking
# ─────────────────────────────────────────────────────────────────────────────

def _row_noise(data_bg, row, lo, hi):
    n_cols = data_bg.shape[1]
    mask = np.ones(n_cols, dtype=bool)
    mask[max(0, lo):min(n_cols, hi)] = False
    bg_vals = data_bg[row, mask]
    return float(np.std(bg_vals)) if bg_vals.size else 0.0


def _centroid_and_intensity(data_bg, lam_axis, row, lo, hi):
    n_cols = data_bg.shape[1]
    seg = data_bg[row, lo:hi]
    total = float(seg.sum())
    if total <= 0:
        return None, 0.0
    idxs = np.arange(lo, hi)
    c_idx = float(np.sum(idxs * seg) / total)
    c_nm = float(np.interp(c_idx, np.arange(n_cols), lam_axis))
    return c_nm, total


def track_peaks_vs_time(data_bg: np.ndarray, lam_axis: np.ndarray,
                         windows: list, snr_thresh: float = 3.0,
                         max_recenter_frac: float = 0.5,
                         max_drift_frac: float = 1.5,
                         min_separation_frac: float = 0.5):
    """
    Track 1 or 2 spectral lines' centroid wavelength + window-integrated
    intensity across every time row, with adaptively re-centered windows.

    Two safeguards beyond a naive per-row centroid walk (both needed —
    verified on real data that a per-row-only step cap still lets a weak
    line get pulled into a much stronger neighboring line's wing over
    hundreds of rows):
      - max_drift_frac    : hard leash on total displacement from the
                             line's ORIGINAL (time-integrated) position,
                             in units of its own initial half-width.
      - min_separation_frac: when 2 windows are tracked together, neither
                             center may cross the other's — enforced every
                             row, independent of SNR — because two distinct
                             physical lines should not swap wavelength
                             order. Uses the smaller of the two initial
                             half-widths as the separation unit.

    Returns dict: label -> (lam_t, inten_t, valid), same length as before.
    """
    n_rows, n_cols = data_bg.shape
    half_widths = {w.label: max(2, (w.right_idx - w.left_idx) // 2) for w in windows}
    max_steps = {lab: max(1, int(round(hw * max_recenter_frac))) for lab, hw in half_widths.items()}
    max_drifts = {lab: int(round(hw * max_drift_frac)) for lab, hw in half_widths.items()}
    orig_centers = {w.label: w.center_idx for w in windows}
    min_sep = int(round(min(half_widths.values()) * min_separation_frac)) if len(windows) == 2 else 0

    out = {w.label: {"lam": np.full(n_rows, np.nan),
                      "inten": np.full(n_rows, np.nan),
                      "valid": np.zeros(n_rows, dtype=bool)} for w in windows}

    # common seed row: best combined SNR proxy = sum of each original window's signal
    combined_seed_signal = np.zeros(n_rows)
    for w in windows:
        combined_seed_signal += data_bg[:, w.left_idx:w.right_idx].sum(axis=1)
    seed_row = int(np.argmax(combined_seed_signal))

    def process_direction(rows):
        centers = dict(orig_centers)
        for row in rows:
            candidates = {}
            for w in windows:
                lab = w.label
                hw = half_widths[lab]
                center = centers[lab]
                lo, hi = max(0, center - hw), min(n_cols, center + hw)
                noise = _row_noise(data_bg, row, lo, hi)
                c_nm, total = _centroid_and_intensity(data_bg, lam_axis, row, lo, hi)
                snr = (total / (noise * np.sqrt(max(hi - lo, 1)))) if noise > 0 else (np.inf if total > 0 else 0.0)
                candidates[lab] = (c_nm, total, snr, center)

            # tentative new centers (only for rows that clear SNR + per-row step cap + drift leash)
            new_centers = dict(centers)
            for w in windows:
                lab = w.label
                c_nm, total, snr, center = candidates[lab]
                if c_nm is None or snr < snr_thresh:
                    continue
                target_idx = int(np.searchsorted(lam_axis, c_nm))
                step_capped = int(np.clip(target_idx, center - max_steps[lab], center + max_steps[lab]))
                drift_capped = int(np.clip(step_capped,
                                            orig_centers[lab] - max_drifts[lab],
                                            orig_centers[lab] + max_drifts[lab]))
                new_centers[lab] = drift_capped

            # non-crossing constraint between the two lines (ordering fixed by initial sort)
            if len(windows) == 2:
                lab_lo, lab_hi = windows[0].label, windows[1].label  # lo = bluer, hi = redder
                if new_centers[lab_lo] > new_centers[lab_hi] - min_sep:
                    mid = (new_centers[lab_lo] + new_centers[lab_hi]) // 2
                    new_centers[lab_lo] = min(new_centers[lab_lo], mid - min_sep // 2)
                    new_centers[lab_hi] = max(new_centers[lab_hi], mid + min_sep // 2)

            for w in windows:
                lab = w.label
                c_nm, total, snr, _ = candidates[lab]
                if c_nm is not None and snr >= snr_thresh:
                    out[lab]["lam"][row] = c_nm
                    out[lab]["inten"][row] = total
                    out[lab]["valid"][row] = True
                centers[lab] = new_centers[lab]

    process_direction(range(seed_row, n_rows))
    process_direction(range(seed_row, -1, -1))

    return {lab: (v["lam"], v["inten"], v["valid"]) for lab, v in out.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-shot analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_shot(img_path: str, bg_pct: float = 10.0,
                  smooth_px: int = 5, prominence_frac: float = 0.05,
                  min_separation_nm: float = 10.0, snr_thresh: float = 3.0,
                  manual_windows: Optional[list] = None,
                  gain: float = 1.0, nd_od: float = 0.0, delay: float = 0.0):
    """
    Run the full per-shot pipeline. If manual_windows is given (list of
    PeakWindow, from GUI override), skip auto-detection and use those
    instead — this is the "manual override" half of the hybrid workflow.

    gain, nd_od : back-correct raw counts to an absolute/comparable scale
        before anything else (background subtraction, artifact screening,
        peak detection) — see apply_gain_nd_correction for the exact
        formula and its stated assumptions. Defaults (1.0, 0.0) are a
        no-op, so existing single-shot usage is unaffected.
    delay       : shift this shot's time axis by -delay (ns), so shots
        taken with different trigger delays land on a common physical
        time base for cross-shot comparison. Sign convention: aligned_t
        = raw_t - delay. If your comparison plots come out shifted the
        wrong way, negate your shotbook delay column — I can't verify
        the polarity of your specific delay convention from here.
        Default 0.0 leaves the raw streak-frame time axis unchanged.

    Returns a dict with data/meta/spectrum/windows/traces — enough for
    both headless CSV/plot output and the GUI to render. 'time_axis' is
    the (possibly delay-shifted) axis used everywhere downstream;
    'time_axis_raw' preserves the original for reference.
    """
    data, meta = parse_hamamatsu_spectral(img_path)
    data = apply_gain_nd_correction(data, gain=gain, nd_od=nd_od)
    artifact_rows = detect_artifact_rows(data)
    data_bg = subtract_row_background(data, pct=bg_pct)
    data_bg[artifact_rows, :] = 0.0   # exclude from tracking AND time-integration
    lam = meta["lambda_axis"]
    t_raw = meta["time_axis"]
    t = t_raw - delay

    spectrum = data_bg.sum(axis=0)

    if manual_windows is not None:
        windows = manual_windows
    else:
        windows = detect_two_peaks(lam, spectrum, smooth_px=smooth_px,
                                    prominence_frac=prominence_frac,
                                    min_separation_nm=min_separation_nm)

    traces = {}
    if windows:
        tracked = track_peaks_vs_time(data_bg, lam, windows, snr_thresh=snr_thresh)
        for win in windows:
            lam_t, inten_t, valid = tracked[win.label]
            traces[win.label] = {"lambda_nm": lam_t, "intensity": inten_t, "valid": valid, "window": win}

    n_artifact_rows = int(artifact_rows.sum())
    needs_review = (len(windows) != 2) or (n_artifact_rows > 0)
    if not needs_review:
        for tr in traces.values():
            if tr["valid"].mean() < 0.5:   # more than half the shot untracked -> flag
                needs_review = True

    return {
        "img_path": img_path, "data": data, "data_bg": data_bg, "meta": meta,
        "lambda_axis": lam, "time_axis": t, "time_axis_raw": t_raw, "spectrum": spectrum,
        "windows": windows, "traces": traces, "needs_review": needs_review,
        "artifact_rows": artifact_rows, "n_artifact_rows": n_artifact_rows,
        "gain": gain, "nd_od": nd_od, "delay": delay,
    }


def save_shot_csv(result: dict, out_csv: str):
    t = result["time_axis"]
    cols = {"time_ns": t}
    for label in ("peak1", "peak2"):
        if label in result["traces"]:
            tr = result["traces"][label]
            cols[f"{label}_lambda_nm"] = tr["lambda_nm"]
            cols[f"{label}_intensity"] = tr["intensity"]
        else:
            cols[f"{label}_lambda_nm"] = np.nan
            cols[f"{label}_intensity"] = np.nan
    pd.DataFrame(cols).to_csv(out_csv, index=False)


def save_shot_plots(result: dict, out_prefix: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lam = result["lambda_axis"]; t = result["time_axis"]

    # QA figure 1: time-integrated spectrum with windows marked
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(lam, result["spectrum"], color="#333", lw=1)
    colors = {"peak1": "#e87040", "peak2": "#4fa8e8"}
    for win in result["windows"]:
        c = colors.get(win.label, "green")
        ax.axvspan(lam[win.left_idx], lam[win.right_idx], color=c, alpha=0.15)
        ax.axvline(win.center_nm, color=c, ls="--", lw=1)
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Integrated counts (bg-sub)")
    ax.set_title(os.path.basename(result["img_path"]))
    fig.tight_layout(); fig.savefig(out_prefix + "_spectrum.png", dpi=130); plt.close(fig)

    # QA figure 2: streak image with tracked traces overlaid.
    # vmax = a robust high percentile, NOT data.max(): a handful of
    # saturated/outlier pixels (real saturation, or a strong shot after
    # gain/ND correction pushes values much higher) will otherwise pin
    # the color scale so high that the actual line signal reads as
    # near-black — this was reported as "not showing anything".
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.percentile(result["data_bg"], 99.5)
    if vmax <= 0:
        vmax = result["data_bg"].max() or 1.0
    im = ax.imshow(result["data_bg"], aspect="auto", origin="lower", cmap="inferno",
                    extent=[lam[0], lam[-1], t[0], t[-1]], vmin=0, vmax=vmax)
    for label, tr in result["traces"].items():
        c = colors.get(label, "green")
        ax.plot(tr["lambda_nm"], t, color=c, lw=1.2, label=label)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Time (ns)" + (" [delay-aligned]" if result.get("delay", 0.0) else ""))
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="Counts (bg-sub, 99.5th-pct clip)")
    fig.tight_layout(); fig.savefig(out_prefix + "_streak.png", dpi=130); plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Batch driver over scan_folder/<shot_id>/<shot_id>.img + external CSV
# ─────────────────────────────────────────────────────────────────────────────

def find_shot_images(scan_folder: str):
    """
    Locate shot images, auto-detecting one of two layouts:
      (a) scan_folder/<shot_id>/<shot_id>.img   (subfolder per shot)
      (b) scan_folder/<shot_id>.img              (flat folder)
    Tries (a) first; if it finds nothing, falls back to (b) — so a flat
    folder of .img files works without needing subfolders.

    Returns (shots, warnings): shots is shot_id -> img_path;
    warnings is a list of human-readable strings (e.g. multiple .img in
    one subfolder) — returned rather than only printed, so a GUI can
    surface them instead of them vanishing into a console the user may
    not have open.
    """
    shots = {}
    warnings = []

    for entry in sorted(os.listdir(scan_folder)):
        sub = os.path.join(scan_folder, entry)
        if not os.path.isdir(sub):
            continue
        imgs = sorted(glob.glob(os.path.join(sub, "*.img")))
        if not imgs:
            continue
        if len(imgs) > 1:
            warnings.append(f"{entry}/: {len(imgs)} .img files found, using {os.path.basename(imgs[0])}")
        shots[entry] = imgs[0]

    if shots:
        return shots, warnings

    # fallback: flat folder, one .img per shot, shot_id = filename stem
    flat_imgs = sorted(glob.glob(os.path.join(scan_folder, "*.img")))
    for p in flat_imgs:
        shot_id = os.path.splitext(os.path.basename(p))[0]
        shots[shot_id] = p
    if flat_imgs:
        warnings.append(f"No subfolder-per-shot layout found; using flat layout "
                         f"({len(flat_imgs)} .img files directly in {scan_folder}).")
    return shots, warnings


def load_shotbook(shotbook_path: str, id_col: str, columns: list) -> dict:
    """
    Load one or more shotbook columns, matched to shot_id the same way as
    before (direct string match, or numeric fallback: folder name's
    embedded integer vs a bare-integer id_col, leading zeros normalized
    on both sides).

    columns: list of column names to pull (None/'' entries are skipped).
    Returns shot_id -> {col_name: value}. Missing/blank cells come
    through as NaN (pandas default) — callers decide the fallback
    (e.g. gain=1, nd_od=0 if not given for a shot).
    """
    df = pd.read_excel(shotbook_path, sheet_name=0)
    df.columns = df.columns.str.strip()
    cols = [c for c in columns if c]
    missing = [c for c in [id_col] + cols if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in shotbook: {missing}. "
                          f"Available: {list(df.columns)}")
    df = df.dropna(subset=[id_col])

    direct, numeric = {}, {}
    for _, row in df.iterrows():
        rec = {c: row[c] for c in cols}
        direct[str(row[id_col])] = rec
        try:
            numeric[str(int(float(row[id_col])))] = rec
        except (ValueError, TypeError):
            pass

    class _ShotbookLookup(dict):
        def __missing__(self_, shot_id):
            if shot_id in direct:
                return direct[shot_id]
            m = re.search(r'\d+', shot_id)
            if m:
                key = str(int(m.group()))
                if key in numeric:
                    return numeric[key]
            raise KeyError(shot_id)

        def __contains__(self_, shot_id):
            if shot_id in direct:
                return True
            m = re.search(r'\d+', shot_id)
            return bool(m and str(int(m.group())) in numeric)

    return _ShotbookLookup(direct)


def load_scan_params(shotbook_path: str, id_col: str, param_col: str) -> dict:
    """Backward-compatible single-column wrapper around load_shotbook."""
    lookup = load_shotbook(shotbook_path, id_col, [param_col])

    class _Single(dict):
        def __missing__(self_, shot_id):
            return lookup[shot_id][param_col]

        def __contains__(self_, shot_id):
            return shot_id in lookup

    return _Single()





# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross-shot comparison plots (compare the scan against a shotbook parameter)
# ─────────────────────────────────────────────────────────────────────────────

PEAK_COLORS = {"peak1": "#e87040", "peak2": "#4fa8e8"}


def plot_summary_vs_param(results: dict, param_values: dict, param_name: str, ax=None):
    """
    One point per shot: peak's time-mean wavelength vs the shotbook
    parameter, with the trace's std as an error bar (a spread-in-time
    indicator, not a measurement uncertainty — label says so).
    results: shot_id -> analyze_shot() result. param_values: shot_id -> value.
    """
    import matplotlib.pyplot as plt
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))
    for label, color in PEAK_COLORS.items():
        xs, ys, yerr = [], [], []
        for shot_id, r in results.items():
            if shot_id not in param_values:
                continue
            tr = r["traces"].get(label)
            if tr is None or not tr["valid"].any():
                continue
            xs.append(param_values[shot_id])
            ys.append(float(np.nanmean(tr["lambda_nm"])))
            yerr.append(float(np.nanstd(tr["lambda_nm"])))
        if xs:
            order = np.argsort(xs)
            xs, ys, yerr = np.array(xs)[order], np.array(ys)[order], np.array(yerr)[order]
            ax.errorbar(xs, ys, yerr=yerr, fmt="o-", color=color, label=label,
                         capsize=3, markersize=5)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Mean \u03bb (nm)  [error bar = std over time, not measurement error]")
    ax.legend(fontsize=9)
    ax.set_title(f"Line position vs {param_name}")
    if fig is not None:
        fig.tight_layout()
    return ax.figure


def plot_time_traces_by_param(results: dict, param_values: dict, param_name: str,
                               peak_label: str = "peak1", quantity: str = "lambda_nm", ax=None):
    """
    Overlay every shot's time-resolved trace (lambda(t) or intensity(t)
    for one peak), colored by the shotbook parameter value, with a
    colorbar. Time axis is each shot's ALIGNED time_axis (raw - delay),
    so shots with different trigger delays land on a shared physical
    time base — meaningless to overlay otherwise.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    vals = [param_values[sid] for sid in results if sid in param_values]
    if not vals:
        ax.text(0.5, 0.5, "No matched shots", transform=ax.transAxes, ha="center")
        return ax.figure
    vmin, vmax = min(vals), max(vals)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1)
    cmap = cm.get_cmap("viridis")

    for shot_id, r in results.items():
        if shot_id not in param_values:
            continue
        tr = r["traces"].get(peak_label)
        if tr is None:
            continue
        y = tr[quantity]
        t = r["time_axis"]
        color = cmap(norm(param_values[shot_id]))
        ax.plot(t, y, color=color, lw=1.2, alpha=0.85)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    if fig is not None:
        fig.colorbar(sm, ax=ax, label=param_name)
    ax.set_xlabel("Time (ns)" + (" [delay-aligned]" if any(r.get("delay", 0.0) for r in results.values()) else ""))
    ylabel = "Wavelength (nm)" if quantity == "lambda_nm" else "Intensity (bg-sub, calibrated)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{peak_label} {ylabel.split()[0].lower()}(t), colored by {param_name}")
    if fig is not None:
        fig.tight_layout()
    return ax.figure


def build_comparison_plots(results: dict, param_values: dict, param_name: str, out_dir: str):
    """Save the standard set of cross-shot comparison PNGs to out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plot_summary_vs_param(results, param_values, param_name)
    fig.savefig(os.path.join(out_dir, f"summary_lambda_vs_{param_name}.png"), dpi=130)
    plt.close(fig)

    for label in ("peak1", "peak2"):
        for qty, tag in (("lambda_nm", "lambda"), ("intensity", "intensity")):
            fig = plot_time_traces_by_param(results, param_values, param_name,
                                             peak_label=label, quantity=qty)
            fig.savefig(os.path.join(out_dir, f"timeresolved_{label}_{tag}_by_{param_name}.png"), dpi=130)
            plt.close(fig)


def run_batch(scan_folder: str, shotbook_path: str, id_col: str, param_col: str,
              gain_col: str = None, nd_col: str = None, delay_col: str = None,
              out_dir: str = "", **analyze_kwargs):
    """scan_folder/<shot_id>/<shot_id>.img + a shotbook (.xlsx) providing
    id_col (matched to shot_id), param_col (the scan variable to compare
    against), and optionally gain_col/nd_col/delay_col for per-shot
    absolute-scale correction and time-axis alignment (see
    apply_gain_nd_correction / analyze_shot for the exact conventions).
    Missing/blank cells for gain/nd/delay default to no-op (gain=1,
    nd_od=0, delay=0) for that shot."""
    os.makedirs(out_dir, exist_ok=True)
    shots, warnings = find_shot_images(scan_folder)
    for w in warnings:
        print(f"[warn] {w}")

    extra_cols = [c for c in (gain_col, nd_col, delay_col) if c]
    shotbook = load_shotbook(shotbook_path, id_col, [param_col] + extra_cols)
    params = {sid: rec[param_col] for sid, rec in
              ((s, shotbook[s]) for s in shots if s in shotbook)}

    matched = [sid for sid in shots if sid in shotbook]
    if shots and not matched:
        print(f"[warn] Found {len(shots)} shot(s) on disk but none matched the shotbook.")
        print(f"[warn]   example folder/file shot_ids found: {list(shots)[:5]}")
        print(f"[warn]   example '{id_col}' values in shotbook: "
              f"{list(pd.read_excel(shotbook_path, sheet_name=0)[id_col].dropna().astype(str))[:5]}")
    elif not shots:
        print(f"[warn] No .img files found under {scan_folder} "
              f"(checked both <shot_id>/<shot_id>.img and flat <shot_id>.img layouts).")

    manifest = []
    long_rows = []
    summary_rows = []
    results = {}

    for shot_id, img_path in shots.items():
        if shot_id not in shotbook:
            print(f"[warn] {shot_id}: no entry in {shotbook_path} (col '{id_col}') — skipped")
            continue
        rec = shotbook[shot_id]
        gain = rec.get(gain_col, 1.0) if gain_col else 1.0
        nd_od = rec.get(nd_col, 0.0) if nd_col else 0.0
        delay = rec.get(delay_col, 0.0) if delay_col else 0.0
        gain = 1.0 if (gain is None or (isinstance(gain, float) and np.isnan(gain))) else float(gain)
        nd_od = 0.0 if (nd_od is None or (isinstance(nd_od, float) and np.isnan(nd_od))) else float(nd_od)
        delay = 0.0 if (delay is None or (isinstance(delay, float) and np.isnan(delay))) else float(delay)

        result = analyze_shot(img_path, gain=gain, nd_od=nd_od, delay=delay, **analyze_kwargs)
        results[shot_id] = result
        shot_out = os.path.join(out_dir, shot_id)
        os.makedirs(shot_out, exist_ok=True)
        save_shot_csv(result, os.path.join(shot_out, f"{shot_id}_peaks.csv"))
        save_shot_plots(result, os.path.join(shot_out, shot_id))

        param_val = params[shot_id]
        t = result["time_axis"]
        for i in range(len(t)):
            row = {"shot_id": shot_id, param_col: param_val, "time_ns": t[i]}
            for label in ("peak1", "peak2"):
                tr = result["traces"].get(label)
                row[f"{label}_lambda_nm"] = tr["lambda_nm"][i] if tr else np.nan
                row[f"{label}_intensity"] = tr["intensity"][i] if tr else np.nan
            long_rows.append(row)

        summary = {"shot_id": shot_id, param_col: param_val,
                   "needs_review": result["needs_review"], "n_peaks_found": len(result["windows"]),
                   "gain": gain, "nd_od": nd_od, "delay": delay}
        for label in ("peak1", "peak2"):
            tr = result["traces"].get(label)
            if tr is not None and tr["valid"].any():
                summary[f"{label}_lambda_mean_nm"] = float(np.nanmean(tr["lambda_nm"]))
                summary[f"{label}_lambda_std_nm"] = float(np.nanstd(tr["lambda_nm"]))
            else:
                summary[f"{label}_lambda_mean_nm"] = np.nan
                summary[f"{label}_lambda_std_nm"] = np.nan
        summary_rows.append(summary)
        manifest.append({"shot_id": shot_id, "img_path": img_path,
                         "needs_review": result["needs_review"]})
        status = "REVIEW" if result["needs_review"] else "ok"
        print(f"[{status}] {shot_id}: {len(result['windows'])} peak(s) found")

    pd.DataFrame(long_rows).to_csv(os.path.join(out_dir, "scan_timeresolved.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "scan_summary.csv"), index=False)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    if results:
        build_comparison_plots(results, params, param_col, out_dir)

    return manifest
