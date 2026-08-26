"""
Hamamatsu streak camera — transverse super-Gaussian fit, vertical lineout,
and grouped shot analysis.
================================================================================

Pipeline (per shot)
-------------------
1.  Read .img (Hamamatsu HPD-TA raw) → data[h, w], float32.
      rows = time (sweep), cols = transverse (spatial, slit axis).
2.  Row-wise background from a USER-DRAWN off-signal column band:
      bg(row) = median( data[row, c0:c1] )      → d = data - bg[:, None]
3.  Time-integrated transverse profile over the active row gate:
      P(x) = Σ_rows d[r, x]
4.  Super-Gaussian fit (free order n):
      P(x) = A · exp( -2 · |(x - x0)/w|^(2n) ) + c
      w = 1/e² half-width (n = 1 → w = 2σ).
5.  Vertical lineout, summed over a user-defined half-width W about x0:
      S(t) = Σ_{x = x0-W}^{x0+W} d[row(t), x]
6.  Time axis from the embedded per-row scaling LUT (ScalingYScalingFile=*off).
      The LUT origin is preserved and its declared unit is converted to ns:
      t_abs(row) = LUT_abs_ns[row] + sign · (DG_ns + EXTRA_ns)

Timing / grouping
-----------------
    DG_ns   : shotbook delay converted to ns from the selected/header unit
    EXTRA_ns: single user-defined additive delay applied to every shot
    t0_ref  : auto-detected on the REFERENCE lineouts (criterion selectable),
              + global reference offset + optional per-reference offset
    window  : HPD-TA Time Range setting. References are matched to shots on
              the same sweep window before any DG-based selection.
    Δt      : computed with one consistent helper for plotting and export.

Shots are grouped by any subset of shotbook columns. Within a group, shots whose
DG differ by more than a tolerance are either (a) interpolated onto a common Δt
grid or (b) split into separate sub-groups. Group traces are reported as
mean ± std over shots (no amplitude normalisation: MCP gain / ND are treated as
grouping keys, not correctable factors — mixed-sensitivity groups are flagged).

Outputs
-------
    <out>/shots/shot_<n>.csv         t_abs, dt, lineout, (fit params in header)
    <out>/shots/shot_<n>.png         image + profile + fit + lineout
    <out>/groups/<group>.csv         dt, mean, std, n_shots
    <out>/groups/<group>.png
    <out>/summary.csv                one row per shot: fit params + kept columns
    <out>/session.json               full GUI state (reproducibility)
"""

import sys
import os
import re
import json
import struct
import glob
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QComboBox, QSizePolicy, QTabWidget,
    QFrame, QDoubleSpinBox, QPushButton, QLineEdit,
    QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView,
    QGroupBox, QSpinBox, QCheckBox, QMessageBox,
    QSplitter, QProgressBar, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QTreeWidget, QTreeWidgetItem, QScrollArea, QDialog
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QKeySequence

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ═════════════════════════════════════════════════════════════════════════════
#  Readers
# ═════════════════════════════════════════════════════════════════════════════

_TIME_UNIT_TO_NS = {
    "ps": 1e-3, "ns": 1.0,
    "us": 1e3, "µs": 1e3, "μs": 1e3,
    "ms": 1e6, "s": 1e9,
}


def _time_unit_factor_ns(unit):
    """Return the multiplicative factor from *unit* to ns, or None."""
    if unit is None:
        return None
    key = str(unit).strip().replace("μ", "µ").lower()
    return _TIME_UNIT_TO_NS.get(key)


def _prepare_time_axis_ns(data, lut, meta, source):
    """
    Convert the camera time calibration to ns without discarding its origin.

    The image rows are reversed together with the LUT when the calibration is
    decreasing.  This gives every downstream routine a strictly increasing
    physical-time axis; leading-edge detection and ``np.interp`` both rely on
    that ordering.
    """
    if source == "lut":
        unit = meta.get("y_unit")
    else:
        unit = meta.get("time_unit")
    fac = _time_unit_factor_ns(unit)
    unit_assumed = fac is None
    if fac is None:
        # HPD-TA LUTs are normally stored in the displayed Y unit.  Retain the
        # old numerical behaviour for malformed/legacy headers, but expose the
        # assumption in metadata rather than silently relabelling it.
        fac = 1.0
    t_abs_ns = np.asarray(lut, dtype=float) * fac
    reversed_rows = bool(t_abs_ns[-1] < t_abs_ns[0])
    if reversed_rows:
        t_abs_ns = t_abs_ns[::-1].copy()
        data = np.asarray(data)[::-1, :].copy()
    if not np.all(np.diff(t_abs_ns) > 0):
        raise ValueError("Camera time calibration is not strictly monotonic")
    meta["time_unit_input"] = unit
    meta["time_unit_assumed_ns"] = unit_assumed
    meta["time_reversed"] = reversed_rows
    meta["time_axis_abs"] = t_abs_ns
    meta["time_axis_rel"] = t_abs_ns - t_abs_ns[0]
    # Backward-compatible key, now explicitly absolute rather than re-zeroed.
    meta["time_axis"] = meta["time_axis_abs"]
    meta["time_origin_ns"] = float(t_abs_ns[0])
    meta["dt_mean"] = float(np.mean(np.diff(t_abs_ns)))
    return data, meta

def _parse_header_text(header_text, h, w):
    """Common HPD-TA header parsing (identical text block in .img and .tif)."""
    meta = {"width": w, "height": h, "header": header_text}

    m = re.search(r'Time Range="?([\d.]+)\s*(\w+)"?', header_text)
    meta["time_range"] = float(m.group(1)) if m else float(h)
    meta["time_unit"] = m.group(2) if m else "px"

    m = re.search(r'ScalingYUnit="([^"]*)"', header_text)
    meta["y_unit"] = m.group(1) if (m and m.group(1)) else meta["time_unit"]

    m = re.search(r'ScalingXScale=([\d.eE+\-]+)', header_text)
    meta["x_scale"] = float(m.group(1)) if m else 1.0
    m = re.search(r'ScalingXUnit="([^"]*)"', header_text)
    meta["x_unit"] = m.group(1) if (m and m.group(1)) else "px"

    m = re.search(r'MCP Gain="?([\d.]+)"?', header_text)
    meta["mcp_gain"] = float(m.group(1)) if m else np.nan

    # pntOrigCh is the image origin on the camera chip; pntBinning gives the
    # number of chip pixels represented by one stored image pixel.  They are
    # required when an old full-height scaling table is embedded in a cropped
    # or binned image.
    x_orig = y_orig = 0
    m_orig = re.search(r'pntOrigCh="(\d+),(\d+)"', header_text)
    if m_orig:
        x_orig, y_orig = int(m_orig.group(1)), int(m_orig.group(2))
    bx = by = 1
    m_bin = re.search(r'pntBinning="(\d+),(\d+)"', header_text)
    if m_bin:
        bx = max(int(m_bin.group(1)), 1)
        by = max(int(m_bin.group(2)), 1)
    meta["chip_origin"] = (x_orig, y_orig)
    meta["binning"] = (bx, by)
    meta["y_offset"] = y_orig
    meta["y_binning"] = by

    # Preserve the historical transverse-axis convention. areGRBScan is used
    # when present because this is how the previous script located the X ROI.
    x_offset = x_orig
    m = re.search(r'areGRBScan="(\d+),(\d+),(\d+),(\d+)"', header_text)
    if m:
        x_offset = int(m.group(1))
    meta["x_offset"] = x_offset
    meta["space_axis"] = (np.arange(w) + x_offset) * meta["x_scale"]
    return meta


def _scaling_pointer(header_text, axis="Y"):
    """Parse an embedded HiPic/HPD-TA scaling-table pointer.

    Old files use ``*offset`` (1024 values) or ``+offset`` (1280 values).
    Newer files use ``#offset,count``; some manuals describe the same form
    without the comma, with the final four digits storing the count.
    """
    name = f"Scaling{axis}ScalingFile"
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', header_text)
    token = m.group(1).strip() if m else ""
    if not token:
        # Conservative fallback for unquoted old headers.
        m = re.search(rf'{name}\s*=\s*([*+]\d+|#\d+(?:,\d+)?)',
                      header_text)
        token = m.group(1).strip() if m else ""
    if not token or token.lower() in {"no scaling", "none"}:
        return None

    marker = token[0]
    body = token[1:].strip()
    try:
        if marker == "*":
            return int(body), 1024, token
        if marker == "+":
            return int(body), 1280, token
        if marker == "#":
            if "," in body:
                off_s, count_s = body.split(",", 1)
            elif len(body) > 4:
                off_s, count_s = body[:-4], body[-4:]
            else:
                return None
            return int(off_s), int(count_s), token
    except ValueError:
        return None
    return None


def _map_scaling_table(table, h, y_offset=0, y_binning=1):
    """Map a chip-coordinate scaling table onto the stored image rows."""
    table = np.asarray(table, dtype=float)
    n = table.size
    h = int(h)
    y_offset = max(int(y_offset or 0), 0)
    y_binning = max(int(y_binning or 1), 1)
    if n == h:
        return table.copy(), "direct"

    def mapped(start, binning):
        stop = start + h * binning
        if stop > n:
            return None
        block = table[start:stop]
        if binning == 1:
            return block.copy()
        return block.reshape(h, binning).mean(axis=1)

    out = mapped(y_offset, y_binning)
    if out is not None:
        return out, f"chip ROI y0={y_offset}, bin={y_binning}"

    # Some exporters have already shifted the table to the ROI but retain the
    # original chip-origin metadata.  Accept that only when the remaining shape
    # gives an exact mapping.
    out = mapped(0, y_binning)
    if out is not None:
        return out, f"ROI-local table, bin={y_binning}"

    # Last deterministic case: infer an integer binning factor from table/image
    # heights.  Do not silently interpolate an incompatible calibration table.
    if h > 0 and n % h == 0:
        inferred = n // h
        out = mapped(0, inferred)
        if out is not None:
            return out, f"inferred bin={inferred}"
    return None, f"incompatible table length {n} for image height {h}"


def _read_y_lut(raw, header_text, h, y_offset=0, y_binning=1):
    """Read and map the embedded per-row time calibration table.

    Scaling values are 4-byte floats whose unit is carried separately by
    ``ScalingYUnit``.  Both ascending and descending tables are valid.
    Returns ``(lut_or_None, source, info)``.
    """
    ptr = _scaling_pointer(header_text, "Y")
    if ptr is None:
        return None, "linear", {"reason": "no embedded Y scaling table"}
    off, declared_count, token = ptr
    info = {"pointer": token, "offset": off,
            "declared_count": declared_count}
    if off <= 0 or off >= len(raw):
        info["reason"] = "invalid table offset"
        return None, "linear", info

    count = declared_count
    if off + 4 * count > len(raw):
        # Compatibility with files whose old marker claims 1024/1280 entries
        # although only the stored image-height table was appended.
        if off + 4 * h <= len(raw):
            count = h
            info["count_fallback"] = h
        else:
            info["reason"] = "scaling table extends beyond file"
            return None, "linear", info
    try:
        table = np.frombuffer(raw, dtype="<f4", count=count,
                              offset=off).astype(np.float64)
    except Exception as exc:
        info["reason"] = f"table read failed: {exc}"
        return None, "linear", info
    if not np.all(np.isfinite(table)):
        info["reason"] = "non-finite scaling value"
        return None, "linear", info
    d = np.diff(table)
    if not (np.all(d > 0) or np.all(d < 0)):
        info["reason"] = "scaling table is not strictly monotonic"
        return None, "linear", info

    lut, mapping = _map_scaling_table(table, h, y_offset, y_binning)
    info["table_count"] = int(table.size)
    info["mapping"] = mapping
    if lut is None or lut.size != h:
        info["reason"] = mapping
        return None, "linear", info
    if abs(lut[-1] - lut[0]) < 1e-12:
        info["reason"] = "mapped scaling table is constant"
        return None, "linear", info
    return lut, "lut", info


def _largest_run_fraction(P):
    """
    Fraction of the (background-subtracted) transverse flux contained in the
    single largest contiguous supra-threshold run of the column profile.

    A correctly-oriented streak image has one connected transverse blob → ~1.
    A wrongly half-swapped one has the true centre columns pushed onto the two
    image edges, splitting the blob in two → ~0.5.
    """
    y = np.clip(np.asarray(P, float) - np.median(P), 0.0, None)
    tot = float(y.sum())
    if tot <= 0 or not np.isfinite(tot):
        return 0.0
    m = (y > 0.1 * y.max()).astype(np.int8)
    if not m.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.concatenate(([0], m, [0]))))
    best = max((float(y[a:b].sum()) for a, b in zip(edges[::2], edges[1::2])),
               default=0.0)
    return best / tot


def detect_half_swap(data, margin=0.05):
    """
    Decide whether the column half-swap is needed, from the data itself.

    Returns (needed, f_asread, f_swapped). The seam-continuity test does NOT
    work here: a wrongly-stored image puts the TRUE edge columns (background,
    hence continuous) at the centre, so the centre seam looks clean either way.
    Blob connectivity is the discriminator. Ambiguous cases (|Δf| < margin,
    e.g. a legitimately edge-located blob) return False — do not swap on a coin
    flip; use the manual override.
    """
    P = np.median(np.asarray(data, float), axis=0)
    half = P.size // 2
    f_a = _largest_run_fraction(P)
    f_b = _largest_run_fraction(np.concatenate([P[half:], P[:half]]))
    return (f_b > f_a + margin), f_a, f_b


def _apply_swap(data, mode):
    """mode: 'auto' | True | False → (data, applied, f_asread, f_swapped)."""
    if mode == "auto":
        need, f_a, f_b = detect_half_swap(data)
    else:
        need, f_a, f_b = bool(mode), np.nan, np.nan
    if need:
        half = data.shape[1] // 2
        data = np.concatenate([data[:, half:], data[:, :half]], axis=1)
    return data, bool(need), f_a, f_b


def parse_hamamatsu_img(filepath, half_swap="auto"):
    """
    Hamamatsu .img reader.

    Format: 64-byte fixed header
        [0:2]   'IM'
        [2:4]   uint16  comment (ASCII header) length
        [4:6]   uint16  width
        [6:8]   uint16  height
        [8:10]  uint16  x offset
        [10:12] uint16  y offset
        [12:14] uint16  data type
    then the ASCII comment, then w*h*2 bytes of uint16 image data, then
    (optionally) the scaling tables pointed to by ScalingY/XScalingFile.

    NOTE vs streak_batch.py: the original reader assumed the file ended with the
    image data (header_size = len(raw) - w*h*2). That is wrong whenever a
    scaling table is appended — it shifts the whole image. Here the comment
    length field is used, with the old heuristic kept only as a fallback.
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    comment_len, w, h = struct.unpack_from("<HHH", raw, 2)
    if w == 0 or h == 0:
        raise ValueError(f"Bad dimensions w={w} h={h}")

    data_off = 64 + comment_len
    n_bytes = w * h * 2
    if data_off + n_bytes > len(raw):
        # fallback: legacy assumption
        data_off = len(raw) - n_bytes
        comment_len = data_off - 64
        if data_off < 64:
            raise ValueError("File too small / inconsistent header")

    header_text = raw[64:64 + comment_len].decode("ascii", errors="ignore")
    meta = _parse_header_text(header_text, h, w)

    data = np.frombuffer(raw, dtype="<u2", count=w * h, offset=data_off).reshape((h, w))
    data = data.astype(np.float32)
    data, applied, f_a, f_b = _apply_swap(data, half_swap)
    meta["half_swap"] = applied
    meta["swap_scores"] = (f_a, f_b)

    lut, src, lut_info = _read_y_lut(
        raw, header_text, h, meta.get("y_offset", 0), meta.get("y_binning", 1))
    if lut is None:
        lut = np.linspace(0.0, meta["time_range"], h)
    meta["t_lut_raw"] = np.asarray(lut, dtype=float).copy()
    meta["t_lut_info"] = lut_info
    meta["t_source"] = src
    data, meta = _prepare_time_axis_ns(data, lut, meta, src)
    return data, meta


def parse_hamamatsu_tif(filepath, half_swap="auto"):
    """HPD-TA TIFF export — same header text in TIFF tag 270, LUT appended."""
    from PIL import Image
    im = Image.open(filepath)
    header_text = im.tag_v2.get(270, "") or ""
    data = np.array(im).astype(np.float32)
    h, w = data.shape
    meta = _parse_header_text(header_text, h, w)
    data, applied, f_a, f_b = _apply_swap(data, half_swap)
    meta["half_swap"] = applied
    meta["swap_scores"] = (f_a, f_b)
    with open(filepath, "rb") as f:
        raw = f.read()
    lut, src, lut_info = _read_y_lut(
        raw, header_text, h, meta.get("y_offset", 0), meta.get("y_binning", 1))
    if lut is None:
        lut = np.linspace(0.0, meta["time_range"], h)
    meta["t_lut_raw"] = np.asarray(lut, dtype=float).copy()
    meta["t_lut_info"] = lut_info
    meta["t_source"] = src
    data, meta = _prepare_time_axis_ns(data, lut, meta, src)
    return data, meta


def load_streak(filepath, half_swap="auto"):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".img":
        return parse_hamamatsu_img(filepath, half_swap=half_swap)
    if ext in (".tif", ".tiff"):
        return parse_hamamatsu_tif(filepath, half_swap=half_swap)
    raise ValueError(f"Unsupported extension: {ext}")


# ═════════════════════════════════════════════════════════════════════════════
#  Analysis core  (GUI-independent, testable)
# ═════════════════════════════════════════════════════════════════════════════

def super_gaussian(x, A, x0, w, n, c):
    """A·exp(-2·|(x-x0)/w|^(2n)) + c ;  w = 1/e² half-width, n free."""
    z = np.abs((x - x0) / w)
    # clip exponent argument to avoid overflow in the wings for large n
    e = np.clip(2.0 * np.power(z, 2.0 * n), 0.0, 700.0)
    return A * np.exp(-e) + c


def fit_transverse(x, P, n_bounds=(0.5, 10.0), n_fixed=None):
    """
    Fit the time-integrated transverse profile.
    Returns dict(A, x0, w, n, c, chi2red, rms, ok, msg).
    """
    P = np.asarray(P, float)
    x = np.asarray(x, float)
    good = np.isfinite(P)
    xg, Pg = x[good], P[good]
    if xg.size < 10:
        return dict(ok=False, msg="not enough points")

    c0 = float(np.median(Pg))
    A0 = float(np.max(Pg) - c0)
    if A0 <= 0:
        return dict(ok=False, msg="no positive signal")

    # moment-based seed on the above-half-max support (robust to wings)
    Wt = np.clip(Pg - c0, 0, None)
    thr = 0.5 * Wt.max()
    sel = Wt > thr
    if sel.sum() < 3:
        sel = Wt > 0.1 * Wt.max()
    x0_0 = float(np.sum(xg[sel] * Wt[sel]) / np.sum(Wt[sel]))
    var = float(np.sum(Wt[sel] * (xg[sel] - x0_0) ** 2) / np.sum(Wt[sel]))
    sig0 = max(np.sqrt(max(var, 1e-6)), 1.0)
    w0 = 2.0 * sig0

    span = xg.max() - xg.min()
    if n_fixed is not None:
        p0 = [A0, x0_0, w0, c0]
        lo = [0.0, xg.min(), 1e-3 * span, -abs(c0) * 10 - 1]
        hi = [10 * A0 + 1, xg.max(), span, abs(c0) * 10 + 1 + A0]
        f = lambda xx, A, x0, w, c: super_gaussian(xx, A, x0, w, n_fixed, c)
    else:
        p0 = [A0, x0_0, w0, 1.0, c0]
        lo = [0.0, xg.min(), 1e-3 * span, n_bounds[0], -abs(c0) * 10 - 1]
        hi = [10 * A0 + 1, xg.max(), span, n_bounds[1], abs(c0) * 10 + 1 + A0]
        f = super_gaussian

    try:
        popt, pcov = curve_fit(f, xg, Pg, p0=p0, bounds=(lo, hi), maxfev=20000)
    except Exception as e:
        return dict(ok=False, msg=f"curve_fit failed: {e}")

    if n_fixed is not None:
        A, x0, w, c = popt
        n = float(n_fixed)
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(4, np.nan)
        errs = dict(A_err=perr[0], x0_err=perr[1], w_err=perr[2], n_err=0.0, c_err=perr[3])
    else:
        A, x0, w, n, c = popt
        perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(5, np.nan)
        errs = dict(A_err=perr[0], x0_err=perr[1], w_err=perr[2], n_err=perr[3], c_err=perr[4])

    resid = Pg - f(xg, *popt)
    dof = max(xg.size - len(popt), 1)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    # unweighted χ²/dof normalised by the residual variance is meaningless;
    # report reduced χ² using the fitted-baseline noise as σ estimate.
    sigma = float(np.std(Pg[np.abs(xg - x0) > 2 * w])) if np.any(np.abs(xg - x0) > 2 * w) else rms
    sigma = sigma if sigma > 0 else 1.0
    chi2red = float(np.sum((resid / sigma) ** 2) / dof)

    out = dict(ok=True, msg="", A=float(A), x0=float(x0), w=float(w),
               n=float(n), c=float(c), rms=rms, chi2red=chi2red,
               fwhm=float(2.0 * w * (np.log(2.0) / 2.0) ** (1.0 / (2.0 * n))))
    out.update({k: float(v) for k, v in errs.items()})
    return out


def _need_num(v, name, default=None):
    """Coerce to float; None → default, or a named error instead of a bare
    TypeError from deep inside numpy ('>=' not supported for NoneType...)."""
    if v is None:
        if default is None:
            raise ValueError(f"{name} is None (unset in the GUI?)")
        return float(default)
    return float(v)


def _need_pair(v, name):
    if v is None or len(v) != 2 or any(x is None for x in v):
        raise ValueError(f"{name} = {v!r} is unset/incomplete")
    return int(v[0]), int(v[1])


def _resolve_gate(row_gate, h):
    """
    Normalise a row gate to a valid (r0, r1) index pair.

    Accepts None (full height) or a 2-tuple whose entries may individually be
    None or <= 0, both meaning 'unset' → 0 / h respectively. The GUI emits
    (0, None) whenever the r1 spinbox sits at its 0 = 'full height' default,
    which is why the None must be handled inside the tuple, not only as the
    tuple itself.
    """
    if row_gate is None:
        return 0, h
    r0, r1 = row_gate
    r0 = 0 if r0 is None else int(r0)
    r1 = h if (r1 is None or int(r1) <= 0) else int(r1)
    r0 = int(np.clip(r0, 0, h - 1))
    r1 = int(np.clip(r1, 1, h))
    if r1 <= r0:
        raise ValueError(f"Row gate r1={r1} must exceed r0={r0}")
    return r0, r1


def _null_fit(prof, w_img, msg):
    """Placeholder fit for empty frames — keeps the display/lineout path alive
    without pretending a super-Gaussian was measured."""
    return dict(ok=False, msg=msg, A=0.0, x0=0.5 * w_img, w=0.25 * w_img,
                n=1.0, c=float(np.median(prof)), rms=float(np.std(prof)),
                chi2red=np.nan, fwhm=np.nan,
                A_err=np.nan, x0_err=np.nan, w_err=np.nan, n_err=np.nan,
                c_err=np.nan)


def profile_snr(prof, bg_cols):
    """
    Transverse-profile SNR.

    σ is taken from the SAME row-summed profile inside the user's off-signal
    band, so it already carries the Σ_rows √N inflation and any residual
    row-to-row background structure — i.e. it is the noise the peak must be
    judged against, not a per-pixel σ. Amplitude is peak-minus-median, which is
    biased high by ~σ on a blank frame; that is why the default threshold is
    well above 1.
    """
    c0, c1 = bg_cols
    noise = float(np.std(prof[c0:c1]))
    amp = float(np.max(prof) - np.median(prof))
    if not np.isfinite(noise) or noise <= 0:
        return np.inf, amp, noise
    return amp / noise, amp, noise


def analyze_shot(data, meta, bg_cols, half_width_px, row_gate=None,
                 dg_ns=0.0, extra_ns=0.0, n_fixed=None,
                 snr_min=10.0, x0_override=None, dg_sign=+1.0):
    """
    Full per-shot reduction. Returns dict with:
        d_bg      : background-subtracted image
        prof      : time-integrated transverse profile (over row_gate)
        fit       : dict from fit_transverse
        t_abs     : absolute time axis (ns)
        lineout   : Σ over [x0-W, x0+W] of d_bg  (counts)
        cols      : (i0, i1) integration column index range (inclusive/exclusive)
    """
    h, w = data.shape
    c0, c1 = _need_pair(bg_cols, "bg_cols")
    half_width_px = _need_num(half_width_px, "half_width_px")
    dg_ns = _need_num(dg_ns, "dg_ns", default=0.0)
    extra_ns = _need_num(extra_ns, "extra_ns", default=0.0)
    c0, c1 = int(np.clip(min(c0, c1), 0, w - 1)), int(np.clip(max(c0, c1), 1, w))
    if c1 - c0 < 3:
        raise ValueError("Background band too narrow (need ≥3 columns)")
    bg = np.median(data[:, c0:c1], axis=1)
    d = data - bg[:, None]

    r0, r1 = _resolve_gate(row_gate, h)
    if r1 - r0 < 3:
        raise ValueError("Row gate too narrow")

    xpix = np.arange(w, dtype=float)
    prof = d[r0:r1, :].sum(axis=0)

    # ── empty-frame test BEFORE the fit: a super-Gaussian will happily fit
    #    noise and return a confident, meaningless x0.
    snr, amp, noise = profile_snr(prof, (c0, c1))
    no_signal = snr < float(snr_min)

    if no_signal:
        fit = _null_fit(prof, w, f"no signal (SNR {snr:.1f} < {snr_min:g})")
    else:
        fit = fit_transverse(xpix, prof, n_fixed=n_fixed)
        if not fit.get("ok"):
            raise ValueError(f"Transverse fit failed: {fit.get('msg')}")

    xc = float(fit["x0"]) if x0_override is None else float(x0_override)
    i0 = int(np.floor(xc - half_width_px))
    i1 = int(np.ceil(xc + half_width_px)) + 1
    i0, i1 = max(i0, 0), min(i1, w)
    if i1 - i0 < 1:
        raise ValueError("Integration window outside the detector")

    lineout = d[:, i0:i1].sum(axis=1)
    # dg_sign = +1: signal/probe delay (larger DG → later within the window).
    # dg_sign = -1: sweep-trigger delay (larger DG → window later → signal
    #               appears earlier in the sweep). extra_ns follows the same
    #               sign so the manual nudge moves the same direction as DG.
    tau_abs = np.asarray(meta.get("time_axis_abs", meta["time_axis"]), float)
    t_abs = tau_abs + float(dg_sign) * (float(dg_ns) + float(extra_ns))

    return dict(d_bg=d, prof=prof, fit=fit, t_abs=t_abs,
                tau_abs=tau_abs,
                tau_rel=np.asarray(meta.get("time_axis_rel",
                                            tau_abs - tau_abs[0]), float),
                lineout=lineout, cols=(i0, i1), bg_cols=(c0, c1),
                row_gate=(r0, r1), snr=snr, amp=amp, noise=noise,
                no_signal=no_signal, x0_used=xc,
                x0_manual=(x0_override is not None))


# ── t0 auto-detection ────────────────────────────────────────────────────────

T0_CRITERIA = ["rise 50%", "rise 20%", "rise 10%", "rise custom",
               "peak", "centroid", "max slope"]


def _ascending_trace(t, y):
    """Return finite, strictly increasing ``t`` with ``y`` reordered alike."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("time and signal must be 1-D arrays of equal length")
    good = np.isfinite(t)
    t, y = t[good], y[good]
    if t.size < 2:
        return t, y
    if np.all(np.diff(t) < 0):
        return t[::-1].copy(), y[::-1].copy()
    if np.all(np.diff(t) > 0):
        return t, y
    order = np.argsort(t, kind="mergesort")
    t, y = t[order], y[order]
    keep = np.concatenate(([True], np.diff(t) > 0))
    return t[keep], y[keep]


def _smooth(y, win):
    win = int(win)
    if win < 5 or win >= len(y):
        return np.asarray(y, float)
    if win % 2 == 0:
        win += 1
    try:
        return savgol_filter(np.asarray(y, float), win, 3)
    except Exception:
        k = np.ones(win) / win
        return np.convolve(np.asarray(y, float), k, mode="same")


def detect_t0(t, y, criterion="rise 50%", smooth_win=21, cent_thresh=0.2,
              rise_frac=0.5):
    """
    Auto t0 on a background-subtracted lineout.

    'rise X%' : first LEADING-EDGE crossing of X% of the peak, linearly
                interpolated between samples; searched backwards from the peak,
                so post-peak structure cannot trigger it.
    'rise custom' : same, with the fraction taken from rise_frac (0–1). Low
                fractions sit closer to the true foot but ride on the noise, so
                t0 jitter grows as the level approaches the baseline; high ones
                are stable but biased late by the pulse's own risetime.
    'peak'    : time of the maximum of the smoothed trace (parabolic refine).
    'centroid': Σ t·y / Σ y over the contiguous region above cent_thresh·max.
    'max slope': time of the steepest positive derivative before the peak.
    """
    t, y = _ascending_trace(t, y)
    if t.size < 2:
        return np.nan, dict(ok=False, msg="not enough finite time samples")
    ys = _smooth(y, smooth_win)
    ys = ys - np.median(ys[: max(int(0.05 * len(ys)), 5)])
    ip = int(np.argmax(ys))
    ymax = ys[ip]
    if not np.isfinite(ymax) or ymax <= 0:
        return np.nan, dict(ok=False, msg="no positive peak")

    if criterion.startswith("rise"):
        if "custom" in criterion:
            frac = float(rise_frac)
        else:
            frac = float(criterion.split()[1].strip("%")) / 100.0
        frac = float(np.clip(frac, 1e-4, 0.9999))
        lvl = frac * ymax
        j = ip
        while j > 0 and ys[j] > lvl:
            j -= 1
        if j == ip:
            return float(t[ip]), dict(ok=True, msg="edge at peak", level=lvl)
        y1, y2 = ys[j], ys[j + 1]
        if y2 == y1:
            return float(t[j]), dict(ok=True, level=lvl)
        f = (lvl - y1) / (y2 - y1)
        return float(t[j] + f * (t[j + 1] - t[j])), dict(ok=True, level=lvl,
                                                         frac=frac)

    if criterion == "peak":
        if 0 < ip < len(ys) - 1:
            y0, y1, y2 = ys[ip - 1], ys[ip], ys[ip + 1]
            den = (y0 - 2 * y1 + y2)
            dd = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            dd = float(np.clip(dd, -1, 1))
            dt = t[ip + 1] - t[ip - 1]
            return float(t[ip] + 0.5 * dd * dt), dict(ok=True, level=ymax)
        return float(t[ip]), dict(ok=True, level=ymax)

    if criterion == "centroid":
        lvl = cent_thresh * ymax
        sel = ys > lvl
        # keep the contiguous block containing the peak
        j0 = ip
        while j0 > 0 and sel[j0 - 1]:
            j0 -= 1
        j1 = ip
        while j1 < len(sel) - 1 and sel[j1 + 1]:
            j1 += 1
        wgt = np.clip(ys[j0:j1 + 1], 0, None)
        if wgt.sum() <= 0:
            return np.nan, dict(ok=False, msg="null weight")
        return float(np.sum(t[j0:j1 + 1] * wgt) / wgt.sum()), dict(ok=True, level=lvl)

    if criterion == "max slope":
        dy = np.gradient(ys, t)
        k = int(np.argmax(dy[: ip + 1])) if ip > 0 else 0
        return float(t[k]), dict(ok=True, level=ys[k])

    return np.nan, dict(ok=False, msg=f"unknown criterion {criterion}")


# ── group resampling ─────────────────────────────────────────────────────────

# ── relative-margin clustering (energy matching) ────────────────────────────
CLUSTER_MODES = ["greedy (spread-capped)", "log bins (fixed edges)"]


def cluster_relative(vals, margin, mode="greedy (spread-capped)"):
    """
    Group a continuous quantity (laser energy) by RELATIVE agreement.

    margin is a half-width: a cluster may span at most ±margin about its centre,
    i.e. a full relative width of 2·margin.

    'greedy (spread-capped)' sorts the values and closes a cluster as soon as
    adding the next one would push (max−min)/mid above 2·margin. This GUARANTEES
    the within-cluster spread and cannot chain (a long ladder of near-neighbours
    can never drift into one wide cluster) — but the partition depends on where
    the first cluster starts, so it is not invariant under adding a new shot.

    'log bins (fixed edges)' assigns bin = floor(ln(E/Emin)/ln(1+2·margin)).
    Reproducible and shot-order independent, but the edges are arbitrary with
    respect to the data: two shots 0.1 % apart straddling an edge land in
    different bins.

    Returns (ids, centers, spans). ids = -1 for non-finite values.
    """
    v = np.asarray(vals, float)
    ids = np.full(v.size, -1, dtype=int)
    ok = np.isfinite(v) & (v > 0)
    if not ok.any() or margin <= 0:
        return ids, {}, {}
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(v[idx])]

    if mode.startswith("log"):
        vmin = float(v[order[0]])
        b = np.floor(np.log(v[order] / vmin) / np.log1p(2.0 * margin)).astype(int)
        for k, ui in enumerate(np.unique(b)):
            ids[order[b == ui]] = k
    else:
        cid, i = 0, 0
        while i < order.size:
            j = i
            while j + 1 < order.size:
                lo, hi = v[order[i]], v[order[j + 1]]
                if (hi - lo) <= 2.0 * margin * (0.5 * (hi + lo)):
                    j += 1
                else:
                    break
            ids[order[i:j + 1]] = cid
            cid += 1
            i = j + 1

    centers, spans = {}, {}
    for c in np.unique(ids[ids >= 0]):
        w = v[ids == c]
        centers[int(c)] = float(np.mean(w))
        mid = 0.5 * (w.max() + w.min())
        spans[int(c)] = (float(w.min()), float(w.max()),
                         float((w.max() - w.min()) / mid) if mid else 0.0)
    return ids, centers, spans


# ── per-trace normalisation ─────────────────────────────────────────────────
NORM_MODES = ["none",
              "each shot → peak",
              "each shot → area (∫dt)",
              "each shot → mean in Δt window",
              "group mean → peak"]

_TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")   # NumPy 2 rename


def normalize_trace(t, y, mode, smooth_win=21, window=None):
    """
    Returns (y_norm, factor). y_norm = y / factor.

    'peak'  uses the peak of a SMOOTHED copy, so a single noise spike cannot set
            the scale, but divides the RAW trace — the normalisation is a scalar,
            not a filter.
    'area'  integrates the whole trace, noise included; on a low-SNR shot the
            baseline contributes and the factor is biased. Prefer 'peak' unless
            you are comparing integrated yields.
    'window' averages over a Δt interval — use it when a flat-top region is the
            physically meaningful reference level.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if mode == "each shot → peak":
        f = float(np.nanmax(_smooth(y, smooth_win)))
    elif mode == "each shot → area (∫dt)":
        ta, ya = _ascending_trace(t, y)
        f = float(_TRAPZ(ya, ta))
    elif mode == "each shot → mean in Δt window":
        if window is None:
            return y, 1.0
        m = (t >= window[0]) & (t <= window[1])
        f = float(np.nanmean(y[m])) if m.any() else np.nan
    else:
        return y, 1.0
    if not np.isfinite(f) or f == 0:
        return y, np.nan          # caller flags it; do not silently divide by ~0
    return y / f, f


def common_grid(traces, mode="intersection", dt=None):
    """
    traces: list of (t, y). Returns the common Δt grid.
    mode: 'intersection' (no extrapolation anywhere) or 'union' (NaN-padded).
    dt  : if None, the finest mean sample spacing among the traces.
    """
    clean = [_ascending_trace(tt, yy) for tt, yy in traces]
    if any(tt.size < 2 for tt, _ in clean):
        raise ValueError("A trace has fewer than two valid time samples")
    t_lo = [tt[0] for tt, _ in clean]
    t_hi = [tt[-1] for tt, _ in clean]
    if mode == "union":
        lo, hi = min(t_lo), max(t_hi)
    else:
        lo, hi = max(t_lo), min(t_hi)
    if hi <= lo:
        raise ValueError("Empty common time range (intersection); use 'union' "
                         "or split the group by delay")
    if dt is None:
        dt = min(float(np.mean(np.diff(tt))) for tt, _ in clean)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"grid dt must be positive, got {dt!r}")
    npts = int(np.floor((hi - lo) / dt)) + 1
    return lo + dt * np.arange(npts)


def resample(t, y, grid):
    t, y = _ascending_trace(t, y)
    out = np.full(grid.shape, np.nan)
    if t.size < 2:
        return out
    m = (grid >= t[0]) & (grid <= t[-1])
    out[m] = np.interp(grid[m], t, y)
    return out


def group_average(traces, mode="intersection", dt=None):
    """traces: list of (t, y) → (grid, mean, std, n_eff)."""
    if not traces:
        raise ValueError("empty group")
    clean = [_ascending_trace(t, y) for t, y in traces]
    grid = common_grid(clean, mode=mode, dt=dt)
    Y = np.vstack([resample(t, y, grid) for t, y in clean])
    n_eff = np.sum(np.isfinite(Y), axis=0)
    import warnings as _w
    with np.errstate(invalid="ignore"), _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(Y, axis=0)
        std = np.nanstd(Y, axis=0, ddof=1) if Y.shape[0] > 1 else np.zeros_like(mean)
    mean[n_eff == 0] = np.nan
    std[n_eff < 2] = np.nan
    return grid, mean, std, n_eff


def relative_time_axis(t_abs, t_offset, dg_ns, t0_value, model,
                       dg_sign=1.0, group_offset=0.0):
    """
    Build the group/export Δt axis from one unambiguous convention.

    Camera-t0 mode removes the common ``extra_ns`` contribution and retains
    only the signed physical DG scan.  The previous implementation subtracted
    ``t0_cam`` directly from ``t_abs`` and therefore leaked ``extra_ns`` into
    every group trace despite the UI saying "keep DG".
    """
    t_abs = np.asarray(t_abs, float)
    if str(model).startswith("camera"):
        tau_abs = t_abs - float(t_offset)
        return (tau_abs - float(t0_value)
                + float(dg_sign) * float(dg_ns) + float(group_offset))
    return t_abs - float(t0_value) + float(group_offset)


# ═════════════════════════════════════════════════════════════════════════════
#  Shotbook
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_SHOT_COL = "Shot n°"
DEFAULT_DG_COL = "Back SOP delay Channel C  (ms)"
DEFAULT_TW_COL = "Back SOP TW (ns)"
DEFAULT_SENS_HINTS = ["mcp", "nd", "filter", "gain", "att"]

# ── delay-column unit → nanoseconds ─────────────────────────────────────────
# Convert the selected shotbook delay column to nanoseconds.  The previous
# default contradicted DEFAULT_DG_COL: the column says "(ms)" but was multiplied
# as seconds, introducing a factor-1000 timing error.
DG_UNITS = {"s  (×1e9)": 1e9, "ms (×1e6)": 1e6, "µs (×1e3)": 1e3,
            "ns (×1)": 1.0, "ps (×1e-3)": 1e-3}
DEFAULT_DG_UNIT = "ms (×1e6)"

_UNIT_HINT = [(r"\(\s*ps\s*\)", "ps"), (r"\(\s*ns\s*\)", "ns"),
              (r"\(\s*[uµ]s\s*\)", "µs"), (r"\(\s*ms\s*\)", "ms"),
              (r"\(\s*s\s*\)", "s")]


def header_unit_hint(colname):
    """Unit token in a column header, or None."""
    for rx, name in _UNIT_HINT:
        if re.search(rx, str(colname), re.I):
            return name
    return None


def dg_unit_choice(unit):
    """GUI choice matching a parsed header unit token, or None."""
    return {"s": "s  (×1e9)", "ms": "ms (×1e6)",
            "µs": "µs (×1e3)", "ns": "ns (×1)",
            "ps": "ps (×1e-3)"}.get(unit)


# ── display timescale for the Δt axis ───────────────────────────────────────
PLOT_UNITS = {"ps": 1e3, "ns": 1.0, "µs": 1e-3, "ms": 1e-6}   # ns → unit


def window_fingerprint(time_range, time_unit):
    """Return the HPD-TA sweep-window duration in ns and a display label."""
    try:
        tr = float(time_range)
    except Exception:
        return None, "uncalibrated"
    fac = _time_unit_factor_ns(time_unit)
    if fac is None or not np.isfinite(tr):
        return None, ("uncalibrated" if str(time_unit).lower() == "px"
                      else f"{tr:g} {time_unit} (unknown unit)")
    return tr * fac, f"{tr:g} {time_unit}"


def same_window(a_ns, b_ns, rel_tol=1e-3):
    """Whether two HPD-TA sweep ranges represent the same hardware setting."""
    if a_ns is None or b_ns is None:
        return False
    if not (np.isfinite(a_ns) and np.isfinite(b_ns)):
        return False
    return abs(a_ns - b_ns) <= rel_tol * max(abs(a_ns), abs(b_ns), 1e-12)


def pick_time_unit(span_ns):
    """Auto display unit from the plotted span (in ns)."""
    a = abs(float(span_ns))
    if not np.isfinite(a) or a == 0:
        return "ns"
    if a < 1.0:
        return "ps"
    if a < 1e3:
        return "ns"
    if a < 1e6:
        return "µs"
    return "ms"

# Filename → shot number. The naive r"(\d+)" grabs the leading date in names
# like "20260527_shot 8_Disp.tif" (→ 20260527, no shotbook match); anchoring on
# "shot" handles "shot 8", "shot_182" and "shot49" alike.
DEFAULT_REGEX = r"[Ss]hot[\s_\-]*(\d+)"

FILE_KINDS = {
    "*.img":                   dict(globs=["*.img"], exclude=None),
    "*.tif  (exclude _Disp)":  dict(globs=["*.tif", "*.tiff"], exclude=r"_Disp\.tiff?$"),
    "*_Disp.tif":              dict(globs=["*_Disp.tif", "*_Disp.tiff"], exclude=None),
    "*.tif  (all)":            dict(globs=["*.tif", "*.tiff"], exclude=None),
    "*.img + *.tif":           dict(globs=["*.img", "*.tif", "*.tiff"], exclude=None),
    "custom glob":             dict(globs=None, exclude=None),
}


def list_streak_files(folder, kind, custom_glob=""):
    """Files of the selected kind, sorted, with the exclusion rule applied."""
    if not folder or not os.path.isdir(folder):
        return []
    spec = FILE_KINDS.get(kind, FILE_KINDS["*.img"])
    globs = spec["globs"] or [g.strip() for g in custom_glob.split(";") if g.strip()]
    files = []
    for g in globs:
        files += glob.glob(os.path.join(folder, g))
    if spec["exclude"]:
        rx = re.compile(spec["exclude"], re.I)
        files = [f for f in files if not rx.search(os.path.basename(f))]
    return sorted(set(files))


def load_shotbook(path):
    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _to_float(v):
    """
    Excel cells are not floats. A French locale writes '1,2'; a careful operator
    writes '1.2 ms'; a duration column comes back as a Timedelta. All of those
    used to return NaN → delay silently 0. Parsing is deliberately strict about
    the WHOLE string (so '2026-05-27' fails rather than yielding 2026).
    """
    if v is None:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if np.isfinite(f) else np.nan
    if hasattr(v, "total_seconds"):          # datetime.timedelta / pd.Timedelta
        try:
            return float(v.total_seconds())
        except Exception:
            return np.nan
    t = str(v).replace("\u00a0", " ").strip()
    if not t:
        return np.nan
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"(?i)(ps|ns|[uµ]s|ms|s)$", "", t)     # trailing unit token
    t = t.replace(",", ".")
    try:
        f = float(t)
    except Exception:
        return np.nan
    return f if np.isfinite(f) else np.nan


def shot_number_from_name(name, pattern=r"(\d+)"):
    m = re.search(pattern, os.path.basename(name))
    return int(m.group(1)) if m else None


# ═════════════════════════════════════════════════════════════════════════════
#  Style (matches streak_batch.py)
# ═════════════════════════════════════════════════════════════════════════════

DARK = "#0d0d0d"
MID = "#191919"
MID2 = "#222222"
ACC = "#e87040"
DIM = "#444444"
TEXT = "#aaaaaa"
GREEN = "#4fc97e"
BLUE = "#4fa8e8"
RED = "#e05252"
YELL = "#e8c84f"

# ── Matplotlib palette, independent of the Qt widget theme ───────────────────
# Figures are rendered with these; switch with the "plots" combo in the bottom
# bar (dark for screen, light for printing / publication figures).
PLOT_THEMES = {
    "dark": dict(bg="#0d0d0d", panel="#191919", text="#aaaaaa", dim="#444444",
                 acc="#e87040", green="#4fc97e", blue="#4fa8e8",
                 yellow="#e8c84f", red="#e05252", cmap="inferno"),
    "light": dict(bg="#ffffff", panel="#f2f2f2", text="#1a1a1a", dim="#999999",
                  acc="#c1481a", green="#1f7a4a", blue="#1f5fa8",
                  yellow="#a07a00", red="#b02020", cmap="viridis"),
}
PLOT = dict(PLOT_THEMES["dark"])


# ── user-adjustable plot style (Plot style… in the bottom bar) ──────────────
STYLE = dict(
    tick_size=8.0, label_size=9.0, title_size=9.0, legend_size=7.0,
    annot_size=7.0,
    lw_mean=1.6, lw_shot=0.4, lw_ref=0.8, lw_marker=0.8, lw_prof=0.7,
    band_alpha=0.20, shot_alpha=0.35, span_alpha=0.15,
    grid=True, grid_alpha=0.5, grid_lw=0.3,
    legend=True, legend_loc="best",
    group_cmap="turbo",
    yscale_group="linear", yscale_ref="linear", yscale_review="linear",
    symlog_thresh=1.0,
    dpi=200, transparent=False, tight=True,
)

# Empty string → the automatic label (which carries the live unit / norm mode).
# "{unit}" and "{norm}" are substituted in the group labels.
LABELS = dict(group_x="", group_y="", group_title="",
              ref_x="", ref_y="", ref_title="",
              img_y="", prof_x="", prof_y="", line_x="")
LABEL_DEFAULTS = dict(LABELS)

SCALES = ["linear", "log", "symlog"]


def apply_scale(ax, axis, mode):
    """
    log masks non-positive samples. Background-subtracted lineouts ARE negative
    in the wings, so a log axis silently drops part of the trace — that is a
    display choice, not a data fix.
    """
    setter = ax.set_yscale if axis == "y" else ax.set_xscale
    if mode == "log":
        setter("log", nonpositive="mask")
    elif mode == "symlog":
        setter("symlog", linthresh=max(float(STYLE["symlog_thresh"]), 1e-12))
    else:
        setter("linear")


SAVE_FILTERS = ["PNG (*.png)", "PDF (*.pdf)", "SVG (*.svg)", "EPS (*.eps)",
                "TIFF (*.tif *.tiff)", "JPEG (*.jpg *.jpeg)"]
KNOWN_EXT = {".png": "png", ".pdf": "pdf", ".svg": "svg", ".eps": "eps",
             ".ps": "ps", ".tif": "tiff", ".tiff": "tiff",
             ".jpg": "jpg", ".jpeg": "jpg"}


def resolve_save_format(path, selected_filter):
    """
    The name filter in a save dialog is ADVISORY: pick 'PDF (*.pdf)' while the
    filename still reads 'groups.png' and Qt hands back 'groups.png'. Matplotlib
    then infers from the suffix and writes a PNG — silently, under a .png name.

    Rule: an explicit known suffix typed by the user wins (they meant it);
    otherwise the suffix comes from the chosen filter; format is passed to
    savefig explicitly so nothing is inferred twice.
    """
    root, cur = os.path.splitext(path)
    cur = cur.lower()
    if cur in KNOWN_EXT:
        return path, KNOWN_EXT[cur]
    m = re.search(r"\*(\.\w+)", selected_filter or "")
    ext = m.group(1).lower() if m else ".png"
    return root + ext, KNOWN_EXT.get(ext, "png")


def _lab(key, auto, **fmt):
    t = LABELS.get(key, "") or ""
    if not t.strip():
        return auto
    try:
        return t.format(**fmt)
    except Exception:
        return t

STYLE_DEFAULTS = dict(STYLE)

GROUP_CMAPS = ["turbo", "viridis", "plasma", "coolwarm", "cividis",
               "tab10", "tab20", "Set1", "Dark2"]
LEGEND_LOCS = ["best", "upper right", "upper left", "lower left",
               "lower right", "center right"]


def group_color(k, n):
    """Qualitative maps are indexed by position, continuous ones sampled."""
    cm = (matplotlib.colormaps[STYLE["group_cmap"]]
          if hasattr(matplotlib, "colormaps")
          else matplotlib.cm.get_cmap(STYLE["group_cmap"]))
    cols = getattr(cm, "colors", None)
    if cols is not None and len(cols) <= 20:
        return cols[k % len(cols)]
    return cm(0.1 + 0.8 * k / max(n - 1, 1))


def set_plot_theme(name):
    PLOT.clear()
    PLOT.update(PLOT_THEMES.get(name, PLOT_THEMES["dark"]))


def plot_colors():
    """Local shadowing helper: bg, panel, text, dim, acc, green, blue, yellow, red."""
    p = PLOT
    return (p["bg"], p["panel"], p["text"], p["dim"], p["acc"],
            p["green"], p["blue"], p["yellow"], p["red"])


def dark_palette():
    """
    Fusion dark palette.

    This is what fixes the black-on-black widgets: a stylesheet set on a parent
    is inherited by every descendant, including QFileDialog / QMessageBox and
    their internal views, which then got `background:#0d0d0d` with the default
    BLACK text. A QPalette + the Fusion style themes every widget consistently
    (including native-looking dialogs) without stylesheet inheritance.
    """
    from PyQt5.QtGui import QPalette
    p = QPalette()
    c = QColor
    p.setColor(QPalette.Window, c(MID))
    p.setColor(QPalette.WindowText, c("#d0d0d0"))
    p.setColor(QPalette.Base, c("#141414"))
    p.setColor(QPalette.AlternateBase, c(MID2))
    p.setColor(QPalette.ToolTipBase, c(MID2))
    p.setColor(QPalette.ToolTipText, c("#d0d0d0"))
    p.setColor(QPalette.Text, c("#d0d0d0"))
    p.setColor(QPalette.Button, c(MID2))
    p.setColor(QPalette.ButtonText, c("#d0d0d0"))
    p.setColor(QPalette.BrightText, c(RED))
    p.setColor(QPalette.Link, c(BLUE))
    p.setColor(QPalette.Highlight, c(ACC))
    p.setColor(QPalette.HighlightedText, c("#000000"))
    p.setColor(QPalette.PlaceholderText, c(DIM))
    for grp in (QPalette.Disabled,):
        p.setColor(grp, QPalette.Text, c("#666666"))
        p.setColor(grp, QPalette.WindowText, c("#666666"))
        p.setColor(grp, QPalette.ButtonText, c("#666666"))
    return p


def _lbl(text, color=TEXT, size=10, bold=False):
    l = QLabel(text)
    w = "bold;" if bold else ""
    l.setStyleSheet(f"color:{color}; font-size:{size}px; font-family:monospace; {w}")
    return l


def _hsep():
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{DIM};")
    return f


def _btn(text, color=ACC, w=None):
    b = QPushButton(text)
    if w:
        b.setFixedWidth(w)
    b.setStyleSheet(f"""
        QPushButton {{
            background:#2a2a2a; color:{color}; border:1px solid {color};
            font-size:11px; font-family:monospace; border-radius:3px; padding:4px 10px;
        }}
        QPushButton:hover {{ background:{color}; color:#000; }}
        QPushButton:disabled {{ color:#444; border-color:#333; }}
    """)
    return b


def _group(title):
    g = QGroupBox(title)
    g.setStyleSheet(f"""
        QGroupBox {{ color:{ACC}; font-family:monospace; font-size:11px;
                     border:1px solid {DIM}; border-radius:3px; margin-top:8px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left:8px; padding:0 4px; }}
    """)
    return g


def _list(multi=True):
    lw = QListWidget()
    lw.setSelectionMode(QAbstractItemView.ExtendedSelection if multi
                        else QAbstractItemView.SingleSelection)
    lw.setStyleSheet(f"""
        QListWidget {{ background:{MID}; color:{TEXT}; border:1px solid {DIM};
                       font-family:monospace; font-size:10px; }}
        QListWidget::item:selected {{ background:{ACC}; color:#000; }}
    """)
    return lw


def _tree():
    tw = QTreeWidget()
    tw.setHeaderHidden(True)
    tw.setStyleSheet(f"""
        QTreeWidget {{ background:{MID}; color:{TEXT}; border:1px solid {DIM};
                       font-family:monospace; font-size:10px; }}
        QTreeWidget::item:selected {{ background:{MID2}; }}
    """)
    return tw


def _split(side_widget, canvas, side_px=430):
    """
    Side panel + canvas in a draggable splitter instead of a fixed width.
    Fixed 260–320 px was unusable: the info/warning panels are monospace and
    routinely wider than that, so text was clipped while the plot had room.
    """
    sp = QSplitter(Qt.Horizontal)
    scroll = QScrollArea()
    scroll.setWidget(side_widget)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setMinimumWidth(300)
    sp.addWidget(scroll)
    sp.addWidget(canvas)
    sp.setStretchFactor(0, 0)
    sp.setStretchFactor(1, 1)
    sp.setSizes([side_px, 1200])
    sp.setChildrenCollapsible(False)
    sp.setStyleSheet(f"QSplitter::handle {{ background:{DIM}; width:3px; }}")
    return sp


def _spin(minv, maxv, val, dec=3, step=0.1, suffix=""):
    s = PasteSpinBox()
    s.setDecimals(dec)
    s.setRange(minv, maxv)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    s.setStyleSheet(f"""
        QDoubleSpinBox {{ background:{MID2}; color:{TEXT}; border:1px solid {DIM};
                          font-family:monospace; font-size:10px; padding:2px; }}
    """)
    return s


def _combo(items):
    c = QComboBox()
    c.addItems(items)
    c.setStyleSheet(f"""
        QComboBox {{ background:{MID2}; color:{TEXT}; border:1px solid {DIM};
                     font-family:monospace; font-size:10px; padding:2px; }}
        QComboBox QAbstractItemView {{ background:{MID}; color:{TEXT};
                                       selection-background-color:{ACC}; }}
    """)
    return c


def _edit(text=""):
    e = QLineEdit(text)
    e.setStyleSheet(f"""
        QLineEdit {{ background:{MID2}; color:{TEXT}; border:1px solid {DIM};
                     font-family:monospace; font-size:10px; padding:3px; }}
    """)
    return e


def _check(text, checked=False):
    c = QCheckBox(text)
    c.setChecked(checked)
    c.setStyleSheet(f"color:{TEXT}; font-family:monospace; font-size:10px;")
    return c


class PasteSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox with locale-tolerant paste (from streak_batch.py)."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Paste):
            txt = QApplication.clipboard().text().strip().replace(",", ".")
            m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', txt)
            if m:
                try:
                    self.setValue(max(self.minimum(), min(self.maximum(), float(m.group()))))
                    return
                except ValueError:
                    pass
        super().keyPressEvent(event)


def _autoscale(data):
    vmin = float(np.percentile(data, 50))
    vmax = float(np.percentile(data, 99.5))
    if vmax <= vmin:
        vmin, vmax = float(data.min()), float(data.max())
    return vmin, vmax


# ═════════════════════════════════════════════════════════════════════════════
#  Data model
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Shot:
    path: str
    name: str
    number: int = -1
    is_ref: bool = False
    dg_ns: float = 0.0
    meta_row: dict = field(default_factory=dict)
    # analysis products
    fit: dict = field(default_factory=dict)
    t_abs: np.ndarray = None
    lineout: np.ndarray = None
    cols: tuple = None
    bg_cols: tuple = None
    row_gate: tuple = None
    t0: float = np.nan          # reference fiducial on t_abs
    t0_cam: float = np.nan      # reference fiducial on τ_abs (applied offset removed)
    ref_offset: float = 0.0     # references only, per-ref
    window_ns: float = None      # HPD-TA Time Range, converted to ns
    window_label: str = ""       # e.g. "50 ns", read from this file's header
    win_lo: float = None         # references only: manually-assigned DG window
    win_hi: float = None         # (ns). None → fall back to nearest-DG matching.
    skipped: bool = False
    no_signal: bool = False
    snr: float = np.nan
    x0_manual: float = None      # None → use the fitted centre
    ov: dict = None              # per-shot overrides, None → follow Setup defaults
    dg_manual: float = None      # user-entered delay; wins over the shotbook
    dg_source: str = "none"      # manual | book | no-row | unparsed | no-book
    dg_raw: object = None        # the shotbook cell, verbatim
    dg_factor: float = np.nan    # unit factor applied to dg_raw → ns
    e_label: str = ""            # energy-matching cluster label
    excluded: bool = False       # kept in its group, but out of mean/±σ/overlay
    t_offset: float = 0.0        # signed(DG + extra): t_abs = τ_abs + t_offset
    x0_used: float = np.nan
    error: str = ""
    error_tb: str = ""

    @property
    def ok(self):
        return (self.lineout is not None) and (not self.skipped) and (not self.error)


# ═════════════════════════════════════════════════════════════════════════════
#  Canvases
# ═════════════════════════════════════════════════════════════════════════════

class ShotCanvas(FigureCanvas):
    """
    Image + transverse profile/fit + lineout, with an interactive
    left-drag column-band selector for the background region.
    """
    band_drawn = pyqtSignal(int, int)
    center_moved = pyqtSignal(float)

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax_img = self.fig.add_axes([0.07, 0.34, 0.55, 0.60])
        self.ax_prof = self.fig.add_axes([0.07, 0.07, 0.55, 0.21])
        self.ax_line = self.fig.add_axes([0.70, 0.34, 0.27, 0.60])
        self._style()

        self._drag0 = None
        self._draw_mode = False
        self._x0 = None            # current centre (data coords)
        self._drag_x0 = False
        self._ln_img = None
        self._ln_prof = None
        self._last_x = None        # last cursor x, so a release outside the
        self._preview = []         # axes does not silently drop the band
        self.mpl_connect("button_press_event", self._press)
        self.mpl_connect("motion_notify_event", self._motion)
        self.mpl_connect("button_release_event", self._release)
        self.placeholder("Load a shot")

    def _style(self):
        DARK, MID, TEXT, DIM, ACC, GREEN, BLUE, YELL, RED = plot_colors()
        for ax in (self.ax_img, self.ax_prof, self.ax_line):
            ax.set_facecolor(DARK)
            for sp in ax.spines.values():
                sp.set_edgecolor(DIM)
            ax.tick_params(colors=TEXT, labelsize=STYLE["tick_size"])
        self.fig.patch.set_facecolor(DARK)

    def placeholder(self, text):
        DARK, MID, TEXT, DIM, ACC, GREEN, BLUE, YELL, RED = plot_colors()
        self._preview = []
        self._ln_img = self._ln_prof = None
        for ax in (self.ax_img, self.ax_prof, self.ax_line):
            ax.cla()
        self._style()
        self.ax_img.text(0.5, 0.5, text, color=DIM, ha="center", va="center",
                         transform=self.ax_img.transAxes, family="monospace")
        self.draw_idle()

    def set_draw_mode(self, on):
        self._draw_mode = bool(on)

    def _press(self, event):
        if event.button != 1 or event.xdata is None:
            return
        if self._draw_mode:
            # image OR profile — the profile is where the wings are actually
            # visible, so refusing clicks there was half the problem
            if event.inaxes in (self.ax_img, self.ax_prof):
                self._drag0 = float(event.xdata)
                self._last_x = float(event.xdata)
            return
        # grab the centre line if the click lands near it (image or profile)
        if event.inaxes in (self.ax_img, self.ax_prof) and self._x0 is not None:
            lo, hi = self.ax_img.get_xlim()
            tol = max(0.015 * abs(hi - lo), 3.0)
            if abs(event.xdata - self._x0) <= tol:
                self._drag_x0 = True

    def _clear_preview(self):
        for art in self._preview:
            try:
                art.remove()
            except Exception:
                pass
        self._preview = []

    def _motion(self, event):
        if event.xdata is not None and event.inaxes in (self.ax_img, self.ax_prof):
            self._last_x = float(event.xdata)

        if self._drag_x0:
            if self._last_x is None:
                return
            self._x0 = self._last_x
            for ln in (self._ln_img, self._ln_prof):
                if ln is not None:
                    ln.set_xdata([self._x0, self._x0])
            self.draw_idle()
            return

        # live band preview while dragging
        if self._draw_mode and self._drag0 is not None and self._last_x is not None:
            self._clear_preview()
            a, b = sorted([self._drag0, self._last_x])
            for ax in (self.ax_img, self.ax_prof):
                self._preview.append(ax.axvspan(a, b, color=PLOT["blue"],
                                                alpha=0.30, lw=0, zorder=5))
                self._preview.append(ax.axvline(a, color=PLOT["blue"], lw=1.0, zorder=6))
                self._preview.append(ax.axvline(b, color=PLOT["blue"], lw=1.0, zorder=6))
            self.draw_idle()

    def _release(self, event):
        if self._drag_x0:
            self._drag_x0 = False
            self.center_moved.emit(float(self._x0))
            return
        if not self._draw_mode or self._drag0 is None:
            self._drag0 = None
            return
        # a release outside the axes has xdata=None; fall back to the last
        # position the cursor had INSIDE an axes rather than dropping the band
        x = event.xdata if event.xdata is not None else self._last_x
        if x is None:
            self._drag0 = None
            self._clear_preview()
            self.draw_idle()
            return
        a, b = sorted([self._drag0, float(x)])
        self._drag0 = None
        self._clear_preview()
        if b - a >= 3:
            self.band_drawn.emit(int(round(a)), int(round(b)))
        else:
            self.draw_idle()

    def render(self, data, meta, res, title="", cmap=None, show_window=True):
        DARK, MID, TEXT, DIM, ACC, GREEN, BLUE, YELL, RED = plot_colors()
        cmap = cmap or PLOT["cmap"]
        self._preview = []
        for ax in (self.ax_img, self.ax_prof, self.ax_line):
            ax.cla()
        self._style()

        h, w = data.shape
        t = res["t_abs"]
        vmin, vmax = _autoscale(res["d_bg"])
        self.ax_img.imshow(res["d_bg"], aspect="auto", cmap=cmap,
                           vmin=vmin, vmax=vmax, origin="lower",
                           extent=[0, w, t[0], t[-1]])
        self.ax_img.set_ylabel(_lab("img_y", "t_abs (ns)"), color=TEXT,
                               fontsize=STYLE["label_size"])
        self.ax_img.set_title(title, color=ACC, fontsize=STYLE["title_size"],
                              family="monospace")

        c0, c1 = res["bg_cols"]
        self.ax_img.axvspan(c0, c1, color=BLUE, alpha=STYLE["span_alpha"])
        self.ax_img.axvline(c0, color=BLUE, lw=STYLE["lw_marker"])
        self.ax_img.axvline(c1, color=BLUE, lw=STYLE["lw_marker"])

        if show_window:
            i0, i1 = res["cols"]
            self.ax_img.axvspan(i0, i1, color=GREEN, alpha=STYLE["span_alpha"])
            self.ax_img.axvline(i0, color=GREEN, lw=STYLE["lw_marker"])
            self.ax_img.axvline(i1, color=GREEN, lw=STYLE["lw_marker"])
        self._x0 = float(res.get("x0_used", res["fit"]["x0"]))
        self._ln_img = self.ax_img.axvline(self._x0, color=ACC,
                                           lw=STYLE["lw_marker"] + 0.6, ls="--")
        if res.get("no_signal"):
            self.ax_img.text(0.5, 0.94, f"NO SIGNAL  (SNR {res['snr']:.1f})",
                             transform=self.ax_img.transAxes, color=RED,
                             ha="center", fontsize=STYLE["annot_size"] + 4,
                             family="monospace",
                             fontweight="bold")

        r0, r1 = res["row_gate"]
        if (r0, r1) != (0, h):
            self.ax_img.axhline(t[r0], color=YELL, lw=STYLE["lw_marker"], ls=":")
            self.ax_img.axhline(t[min(r1, h - 1)], color=YELL,
                                lw=STYLE["lw_marker"], ls=":")

        # transverse profile + fit
        x = np.arange(w)
        f = res["fit"]
        self.ax_prof.plot(x, res["prof"], color=TEXT, lw=STYLE["lw_prof"])
        if f.get("ok", True):
            self.ax_prof.plot(x, super_gaussian(x, f["A"], f["x0"], f["w"],
                                                f["n"], f["c"]),
                              color=ACC, lw=STYLE["lw_mean"])
        if res.get("x0_manual"):
            # fitted centre stays visible as a reference when overridden
            self.ax_prof.axvline(f["x0"], color=DIM, lw=STYLE["lw_marker"], ls=":")
            self.ax_img.axvline(f["x0"], color=DIM, lw=STYLE["lw_marker"], ls=":")
        self._ln_prof = self.ax_prof.axvline(self._x0, color=ACC,
                                             lw=STYLE["lw_marker"] + 0.6, ls="--")
        if show_window:
            i0, i1 = res["cols"]
            self.ax_prof.axvspan(i0, i1, color=GREEN, alpha=STYLE["span_alpha"])
        self.ax_prof.axvspan(c0, c1, color=BLUE, alpha=STYLE["span_alpha"])
        self.ax_prof.set_xlim(0, w)
        self.ax_prof.set_xlabel(_lab("prof_x", "transverse (px)"), color=TEXT,
                                fontsize=STYLE["label_size"])
        self.ax_prof.set_ylabel(_lab("prof_y", "Σ_t counts"), color=TEXT,
                                fontsize=STYLE["label_size"])
        apply_scale(self.ax_prof, "y", STYLE["yscale_review"])
        tag = "  [x0 MANUAL — drag]" if res.get("x0_manual") else ""
        self.ax_prof.text(
            0.02, 0.92,
            f"x0={self._x0:.1f}" + (f"±{f.get('x0_err', np.nan):.1f}"
                                    if not res.get("x0_manual") else "") +
            f"  w={f['w']:.1f}  n={f['n']:.2f}  FWHM={f['fwhm']:.1f} px  "
            f"χ²ᵣ={f['chi2red']:.2f}  SNR={res.get('snr', np.nan):.1f}{tag}",
            transform=self.ax_prof.transAxes,
            color=(RED if res.get("no_signal") else GREEN),
            fontsize=STYLE["annot_size"],
            family="monospace", va="top")

        # lineout
        self.ax_line.plot(res["lineout"], t, color=GREEN, lw=STYLE["lw_ref"])
        self.ax_line.set_xlabel(_lab("line_x", "Σ_x counts"), color=TEXT,
                                fontsize=STYLE["label_size"])
        # time is the VERTICAL axis here, so the signal axis is x
        apply_scale(self.ax_line, "x", STYLE["yscale_review"])
        self.ax_line.set_ylim(t[0], t[-1])
        if STYLE["grid"]:
            self.ax_line.grid(color=DIM, lw=STYLE["grid_lw"],
                              alpha=STYLE["grid_alpha"])
        self.draw_idle()


class PlotCanvas(FigureCanvas):
    """Generic single-axes canvas (references / groups)."""

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_axes([0.09, 0.11, 0.88, 0.84])
        self._style()

    def _style(self):
        DARK, MID, TEXT, DIM, ACC, GREEN, BLUE, YELL, RED = plot_colors()
        self.ax.set_facecolor(DARK)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(DIM)
        self.ax.tick_params(colors=TEXT, labelsize=STYLE["tick_size"])
        self.fig.patch.set_facecolor(DARK)

    def clear(self, msg=None):
        DARK, MID, TEXT, DIM, ACC, GREEN, BLUE, YELL, RED = plot_colors()
        self.ax.cla()
        self._style()
        if msg:
            self.ax.text(0.5, 0.5, msg, color=DIM, ha="center", va="center",
                         transform=self.ax.transAxes, family="monospace")
        self.draw_idle()


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Setup
# ═════════════════════════════════════════════════════════════════════════════

class SetupTab(QWidget):
    process_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self.folder = ""
        self.ref_folder = ""
        self.sb_path = ""

        L = QVBoxLayout(self)
        L.setSpacing(8)

        # ── paths
        g = _group("Data")
        gl = QGridLayout(g)
        self.folder_lbl = _lbl("—", size=9)
        self.ref_lbl = _lbl("—", size=9)
        self.sb_lbl = _lbl("—", size=9)
        b1 = _btn("Shot folder…", w=130)
        b2 = _btn("References…", BLUE, w=130)
        b3 = _btn("Shotbook (.xlsx)…", w=130)
        b1.clicked.connect(self._pick_folder)
        b2.clicked.connect(self._pick_refs)
        b3.clicked.connect(self._pick_sb)
        gl.addWidget(b1, 0, 0); gl.addWidget(self.folder_lbl, 0, 1)
        gl.addWidget(b2, 1, 0); gl.addWidget(self.ref_lbl, 1, 1)
        gl.addWidget(b3, 2, 0); gl.addWidget(self.sb_lbl, 2, 1)
        gl.addWidget(_lbl("file type"), 3, 0)
        hk = QHBoxLayout()
        self.cb_kind = _combo(list(FILE_KINDS.keys()))
        self.cb_kind.currentTextChanged.connect(
            lambda t: self.edit_glob.setEnabled(t == "custom glob"))
        self.edit_glob = _edit("*.img; *.tif")
        self.edit_glob.setEnabled(False)
        hk.addWidget(self.cb_kind, 2); hk.addWidget(self.edit_glob, 1)
        wk = QWidget(); wk.setLayout(hk)
        gl.addWidget(wk, 3, 1)

        gl.addWidget(_lbl("filename → shot n° regex"), 4, 0)
        hr = QHBoxLayout()
        self.regex_edit = _edit(DEFAULT_REGEX)
        b_prev = _btn("Preview", BLUE, w=80)
        b_prev.clicked.connect(self._preview)
        hr.addWidget(self.regex_edit, 1); hr.addWidget(b_prev)
        wrx = QWidget(); wrx.setLayout(hr)
        gl.addWidget(wrx, 4, 1)
        gl.setColumnStretch(1, 1)
        L.addWidget(g)

        # ── shotbook columns
        g2 = _group("Shotbook columns")
        g2l = QGridLayout(g2)
        g2l.addWidget(_lbl("shot n° column"), 0, 0)
        self.cb_shot = _combo([])
        g2l.addWidget(self.cb_shot, 0, 1)
        g2l.addWidget(_lbl("delay column"), 1, 0)
        self.cb_dg = _combo([])
        self.cb_dg.currentTextChanged.connect(self._dg_hint)
        g2l.addWidget(self.cb_dg, 1, 1)
        g2l.addWidget(_lbl("delay unit → ns", YELL), 2, 0)
        hu = QHBoxLayout()
        self.cb_dgunit = _combo(list(DG_UNITS.keys()))
        self.cb_dgunit.setCurrentText(DEFAULT_DG_UNIT)
        self.cb_dgunit.currentTextChanged.connect(lambda _: self._dg_hint())
        self.lbl_dghint = _lbl("", YELL, 9)
        hu.addWidget(self.cb_dgunit, 1); hu.addWidget(self.lbl_dghint, 2)
        wu = QWidget(); wu.setLayout(hu)
        g2l.addWidget(wu, 2, 1)
        g2l.addWidget(_lbl("keep columns (→ summary)"), 3, 0, Qt.AlignTop)
        self.lst_keep = _list()
        self.lst_keep.setMinimumHeight(120)
        g2l.addWidget(self.lst_keep, 3, 1)
        g2l.addWidget(_lbl("group by"), 4, 0, Qt.AlignTop)
        self.lst_group = _list()
        self.lst_group.setMinimumHeight(120)
        g2l.addWidget(self.lst_group, 4, 1)
        g2l.addWidget(_lbl("sensitivity columns\n(MCP / ND — flagged if mixed)",
                           color=YELL), 5, 0, Qt.AlignTop)
        self.lst_sens = _list()
        self.lst_sens.setMinimumHeight(70)
        g2l.addWidget(self.lst_sens, 5, 1)
        g2l.setColumnStretch(1, 1)
        L.addWidget(g2)

        # ── analysis defaults
        g3 = _group("Analysis defaults")
        g3l = QGridLayout(g3)
        g3l.addWidget(_lbl("bg band  c0 / c1 (px)"), 0, 0)
        hb = QHBoxLayout()
        self.sp_bg0 = _spin(0, 4096, 50, dec=0, step=1)
        self.sp_bg1 = _spin(0, 4096, 250, dec=0, step=1)
        hb.addWidget(self.sp_bg0); hb.addWidget(self.sp_bg1)
        wbg = QWidget(); wbg.setLayout(hb)
        g3l.addWidget(wbg, 0, 1)

        g3l.addWidget(_lbl("lineout half-width W (px)", color=GREEN), 1, 0)
        self.sp_W = _spin(0.5, 500, 10, dec=1, step=0.5)
        g3l.addWidget(self.sp_W, 1, 1)

        g3l.addWidget(_lbl("row gate r0 / r1 (px, fit only)"), 2, 0)
        hb2 = QHBoxLayout()
        self.sp_r0 = _spin(0, 8192, 0, dec=0, step=1)
        self.sp_r1 = _spin(0, 8192, 0, dec=0, step=1)   # 0 → full height
        hb2.addWidget(self.sp_r0); hb2.addWidget(self.sp_r1)
        wr = QWidget(); wr.setLayout(hb2)
        g3l.addWidget(wr, 2, 1)

        self.chk_nfree = _check("free super-Gaussian order n", True)
        g3l.addWidget(self.chk_nfree, 3, 0)
        self.sp_nfix = _spin(0.5, 10, 1.0, dec=2, step=0.1)
        self.sp_nfix.setEnabled(False)
        self.chk_nfree.toggled.connect(lambda v: self.sp_nfix.setEnabled(not v))
        g3l.addWidget(self.sp_nfix, 3, 1)

        g3l.addWidget(_lbl("extra delay (all shots)", color=YELL), 4, 0)
        self.sp_extra = _spin(-1e6, 1e6, 0.0, dec=4, step=0.01, suffix=" ns")
        g3l.addWidget(self.sp_extra, 4, 1)

        g3l.addWidget(_lbl("no-signal SNR threshold", YELL), 5, 0)
        hs = QHBoxLayout()
        self.sp_snr = _spin(0.0, 1e4, 10.0, dec=1, step=1.0)
        self.chk_autoskip = _check("auto-skip", True)
        hs.addWidget(self.sp_snr); hs.addWidget(self.chk_autoskip)
        wsn = QWidget(); wsn.setLayout(hs)
        g3l.addWidget(wsn, 5, 1)

        g3l.addWidget(_lbl("delay sign in t_abs", YELL), 6, 0)
        self.cb_dgsign = _combo(["+  signal/probe delay (τ + DG)",
                                 "−  sweep-trigger delay (τ − DG)"])
        self.cb_dgsign.setToolTip(
            "Sweep/SOP trigger delay: a larger delay moves the streak WINDOW "
            "later, so a fixed signal appears EARLIER in the sweep → choose −.\n"
            "Delay applied to the light itself → choose +.")
        g3l.addWidget(self.cb_dgsign, 6, 1)

        g3l.addWidget(_lbl("column half-swap"), 7, 0)
        self.cb_swap = _combo(["auto (detect)", "force on", "force off"])
        g3l.addWidget(self.cb_swap, 7, 1)
        g3l.setColumnStretch(1, 1)
        L.addWidget(g3)

        hb3 = QHBoxLayout()
        self.btn_scan = _btn("Scan files", BLUE, w=110)
        self.btn_proc = _btn("Process all →", GREEN, w=140)
        self.btn_scan.clicked.connect(self._scan)
        self.btn_proc.clicked.connect(self.process_requested.emit)
        self.btn_proc.setEnabled(False)
        hb3.addWidget(self.btn_scan); hb3.addWidget(self.btn_proc); hb3.addStretch()
        L.addLayout(hb3)

        self.status = _lbl("Select a shot folder.", size=9)
        L.addWidget(self.status)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFixedHeight(150)
        self.preview.setStyleSheet(f"background:{MID}; color:{TEXT}; border:1px solid {DIM};"
                                   f"font-family:monospace; font-size:9px;")
        L.addWidget(self.preview)
        L.addStretch()

    def _preview(self):
        """file → parsed shot n° → shotbook match. Answers 'is my regex right?'
        before 200 shots get silently mis-numbered."""
        pat = self.regex_edit.text() or DEFAULT_REGEX
        try:
            re.compile(pat)
        except re.error as e:
            self.preview.setText(f"bad regex: {e}")
            return
        files = list_streak_files(self.folder, self.cb_kind.currentText(),
                                  self.edit_glob.text())
        refs = list_streak_files(self.ref_folder, self.cb_kind.currentText(),
                                 self.edit_glob.text()) if self.ref_folder else []
        idx, dgs = set(), {}
        if self.df is not None and self.cb_shot.currentText():
            dc = self.cb_dg.currentText()
            fac = DG_UNITS.get(self.cb_dgunit.currentText(), DG_UNITS[DEFAULT_DG_UNIT])
            for _, row in self.df.iterrows():
                fv = _to_float(row.get(self.cb_shot.currentText(), np.nan))
                if np.isfinite(fv):
                    idx.add(int(fv))
                    raw = row.get(dc, None) if dc else None
                    dgs[int(fv)] = (raw, _to_float(raw) * fac if dc else np.nan)
        matched, unmatched = [], []
        for f in files + refs:
            n = shot_number_from_name(f, pat)
            (matched if n is not None else unmatched).append((f, n))

        lines = [f"{len(files)} shots + {len(refs)} refs   "
                 f"({len(matched)} regex-matched, {len(unmatched)} unmatched)\n"]

        lines.append(f"— first {min(5, len(matched))} regex matches —")
        for f, n in matched[:5]:
            tag = "REF " if f in refs else "    "
            if not idx:
                hit = "(no shotbook)"
            elif n not in idx:
                hit = "NOT IN SHOTBOOK → DG = 0"
            else:
                raw, dg = dgs.get(n, (None, np.nan))
                hit = (f"DG = {dg:.4f} ns  (cell {raw!r})" if np.isfinite(dg)
                       else f"⚠ DG UNPARSED → 0   (cell {raw!r})")
            lines.append(f"{tag}{os.path.basename(f):38s} → {str(n):>10s}  {hit}")
        if len(matched) > 5:
            lines.append(f"… {len(matched) - 5} more matched (not shown)")

        if unmatched:
            lines.append(f"\n— all {len(unmatched)} regex NON-matches —")
            for f, n in unmatched:
                tag = "REF " if f in refs else "    "
                lines.append(f"{tag}{os.path.basename(f):38s} → "
                              f"NO REGEX MATCH  (pattern {pat!r} not found in filename)")

        self.preview.setText("\n".join(lines))

    # ── pickers
    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Shot folder")
        if d:
            self.folder = d
            self.folder_lbl.setText(d)
            auto = os.path.join(d, "References")
            if os.path.isdir(auto) and not self.ref_folder:
                self.ref_folder = auto
                self.ref_lbl.setText(auto + "  (auto)")

    def _pick_refs(self):
        d = QFileDialog.getExistingDirectory(self, "References folder")
        if d:
            self.ref_folder = d
            self.ref_lbl.setText(d)

    def _pick_sb(self):
        p, _ = QFileDialog.getOpenFileName(self, "Shotbook", "", "Excel (*.xlsx *.xls)")
        if not p:
            return
        try:
            self.df = load_shotbook(p)
        except Exception as e:
            QMessageBox.critical(self, "Shotbook", f"{e}")
            return
        self.sb_path = p
        self.sb_lbl.setText(os.path.basename(p) + f"  [{len(self.df)} rows]")
        self._populate_cols()

    def _dg_hint(self, *_):
        """Show the shotbook header unit against the selected conversion."""
        col = self.cb_dg.currentText()
        hint = header_unit_hint(col)
        chosen = self.cb_dgunit.currentText().split()[0]
        if hint is None:
            self.lbl_dghint.setText("")
        elif hint == chosen:
            self.lbl_dghint.setText(f"header says ({hint}) ✓")
        else:
            self.lbl_dghint.setText(f"⚠ header says ({hint}), using {chosen}")

    def _populate_cols(self):
        cols = list(self.df.columns)
        for cb, default in ((self.cb_shot, DEFAULT_SHOT_COL), (self.cb_dg, DEFAULT_DG_COL)):
            cb.clear(); cb.addItems(cols)
            hit = [c for c in cols if c.strip() == default.strip()]
            if hit:
                cb.setCurrentText(hit[0])
        # The column name is the only explicit unit metadata available for the
        # shotbook delay.  Start from it instead of reproducing the old ×1e9
        # hard-code when the selected column explicitly says "(ms)".
        hinted = dg_unit_choice(header_unit_hint(self.cb_dg.currentText()))
        if hinted is not None:
            self.cb_dgunit.setCurrentText(hinted)
        self._dg_hint()
        for lw in (self.lst_keep, self.lst_group, self.lst_sens):
            lw.clear()
            for c in cols:
                lw.addItem(QListWidgetItem(c))
        # preselect sensitivity-like columns
        for i in range(self.lst_sens.count()):
            name = self.lst_sens.item(i).text().lower()
            if any(k in name for k in DEFAULT_SENS_HINTS):
                self.lst_sens.item(i).setSelected(True)

    def _scan(self):
        if not self.folder:
            QMessageBox.warning(self, "Scan", "Pick a shot folder first.")
            return
        pat = self.regex_edit.text() or DEFAULT_REGEX
        try:
            re.compile(pat)
        except re.error as e:
            QMessageBox.critical(self, "Regex", f"bad regex: {e}")
            return
        shots = self._collect(self.folder, is_ref=False, pattern=pat)
        refs = self._collect(self.ref_folder, is_ref=True, pattern=pat) if self.ref_folder else []
        self._scanned = (shots, refs)
        msg = f"{len(shots)} shots, {len(refs)} references found."
        if getattr(self, "_unmatched", 0):
            msg += f"  ⚠ {self._unmatched} filename(s) unmatched by the regex → shot n° = -1"
        self._unmatched = 0
        self.status.setText(msg)
        self._preview()
        self.btn_proc.setEnabled(bool(shots))

    def _collect(self, folder, is_ref, pattern):
        out = []
        files = list_streak_files(folder, self.cb_kind.currentText(),
                                  self.edit_glob.text())
        refdir = os.path.abspath(self.ref_folder) if self.ref_folder else None
        unmatched = 0
        for f in files:
            if (not is_ref) and refdir and os.path.abspath(os.path.dirname(f)) == refdir:
                continue
            n = shot_number_from_name(f, pattern)
            if n is None:
                unmatched += 1
            out.append(Shot(path=f, name=os.path.basename(f),
                            number=n if n is not None else -1, is_ref=is_ref))
        self._unmatched = getattr(self, "_unmatched", 0) + unmatched
        return out

    def config(self):
        r1 = int(self.sp_r1.value())
        return dict(
            folder=self.folder, ref_folder=self.ref_folder, shotbook=self.sb_path,
            regex=self.regex_edit.text(),
            file_kind=self.cb_kind.currentText(),
            custom_glob=self.edit_glob.text(),
            shot_col=self.cb_shot.currentText(), dg_col=self.cb_dg.currentText(),
            dg_unit=self.cb_dgunit.currentText(),
            keep_cols=[i.text() for i in self.lst_keep.selectedItems()],
            group_cols=[i.text() for i in self.lst_group.selectedItems()],
            sens_cols=[i.text() for i in self.lst_sens.selectedItems()],
            bg_cols=(int(self.sp_bg0.value()), int(self.sp_bg1.value())),
            half_width=float(self.sp_W.value()),
            row_gate=(int(self.sp_r0.value()), r1 if r1 > 0 else None),
            n_fixed=None if self.chk_nfree.isChecked() else float(self.sp_nfix.value()),
            extra_ns=float(self.sp_extra.value()),
            dg_sign=(-1.0 if self.cb_dgsign.currentIndex() == 1 else 1.0),
            snr_min=float(self.sp_snr.value()),
            auto_skip=self.chk_autoskip.isChecked(),
            half_swap={"auto (detect)": "auto", "force on": True,
                       "force off": False}[self.cb_swap.currentText()],
        )

    def apply_config(self, c):
        self.folder = c.get("folder", ""); self.folder_lbl.setText(self.folder or "—")
        self.ref_folder = c.get("ref_folder", ""); self.ref_lbl.setText(self.ref_folder or "—")
        self.regex_edit.setText(c.get("regex", DEFAULT_REGEX))
        self.cb_kind.setCurrentText(c.get("file_kind", "*.img"))
        self.edit_glob.setText(c.get("custom_glob", "*.img; *.tif"))
        if c.get("shotbook") and os.path.exists(c["shotbook"]):
            self.df = load_shotbook(c["shotbook"])
            self.sb_path = c["shotbook"]
            self.sb_lbl.setText(os.path.basename(self.sb_path))
            self._populate_cols()
            self.cb_shot.setCurrentText(c.get("shot_col", ""))
            self.cb_dg.setCurrentText(c.get("dg_col", ""))
            self.cb_dgunit.setCurrentText(c.get("dg_unit", DEFAULT_DG_UNIT))
            for lw, key in ((self.lst_keep, "keep_cols"), (self.lst_group, "group_cols"),
                            (self.lst_sens, "sens_cols")):
                want = set(c.get(key, []))
                for i in range(lw.count()):
                    lw.item(i).setSelected(lw.item(i).text() in want)
        self.sp_bg0.setValue(c.get("bg_cols", (50, 250))[0])
        self.sp_bg1.setValue(c.get("bg_cols", (50, 250))[1])
        self.sp_W.setValue(c.get("half_width", 10))
        rg = c.get("row_gate", (0, None))
        self.sp_r0.setValue(rg[0] or 0)
        self.sp_r1.setValue(rg[1] or 0)
        nf = c.get("n_fixed")
        self.chk_nfree.setChecked(nf is None)
        if nf is not None:
            self.sp_nfix.setValue(nf)
        self.sp_extra.setValue(c.get("extra_ns", 0.0))
        self.cb_dgsign.setCurrentIndex(1 if c.get("dg_sign", 1.0) < 0 else 0)
        self.sp_snr.setValue(c.get("snr_min", 10.0))
        self.chk_autoskip.setChecked(c.get("auto_skip", True))
        self.cb_swap.setCurrentText({"auto": "auto (detect)", True: "force on",
                                     False: "force off"}.get(c.get("half_swap", "auto"),
                                                             "auto (detect)"))


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Review (per-shot)
# ═════════════════════════════════════════════════════════════════════════════

class ReviewTab(QWidget):
    reanalyze = pyqtSignal(object, dict)   # (shot, overrides)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.shots = []
        self.cur = None
        self._cache = None
        self._dgsign = 1.0

        L = QHBoxLayout(self)
        side = QVBoxLayout()
        side.addWidget(_lbl("SHOTS", ACC, 11, True))
        self.lst = _list(multi=False)
        self.lst.setFixedWidth(240)
        self.lst.currentRowChanged.connect(self._select)
        side.addWidget(self.lst)

        g = _group("Overrides (this shot)")
        gl = QGridLayout(g)
        gl.addWidget(_lbl("bg c0/c1"), 0, 0)
        hb = QHBoxLayout()
        self.sp_bg0 = _spin(0, 4096, 50, dec=0, step=1)
        self.sp_bg1 = _spin(0, 4096, 250, dec=0, step=1)
        hb.addWidget(self.sp_bg0); hb.addWidget(self.sp_bg1)
        w = QWidget(); w.setLayout(hb); gl.addWidget(w, 0, 1)
        gl.addWidget(_lbl("W (px)", GREEN), 1, 0)
        self.sp_W = _spin(0.5, 500, 10, dec=1, step=0.5)
        gl.addWidget(self.sp_W, 1, 1)
        gl.addWidget(_lbl("row gate"), 2, 0)
        hb2 = QHBoxLayout()
        self.sp_r0 = _spin(0, 8192, 0, dec=0, step=1)
        self.sp_r1 = _spin(0, 8192, 0, dec=0, step=1)
        hb2.addWidget(self.sp_r0); hb2.addWidget(self.sp_r1)
        w2 = QWidget(); w2.setLayout(hb2); gl.addWidget(w2, 2, 1)
        gl.addWidget(_lbl("centre x0 (px)", ACC), 3, 0)
        hx = QHBoxLayout()
        self.chk_x0 = _check("manual")
        self.sp_x0 = _spin(0, 8192, 0, dec=2, step=1.0)
        self.sp_x0.setEnabled(False)
        self.chk_x0.toggled.connect(self._x0_mode)
        self.sp_x0.valueChanged.connect(self._x0_typed)
        hx.addWidget(self.chk_x0); hx.addWidget(self.sp_x0)
        wx = QWidget(); wx.setLayout(hx)
        gl.addWidget(wx, 3, 1)
        self.btn_x0_reset = _btn("x0 → back to fit")
        self.btn_x0_reset.clicked.connect(lambda: self.chk_x0.setChecked(False))
        gl.addWidget(self.btn_x0_reset, 4, 0, 1, 2)

        self.btn_draw = _btn("Draw bg band  (drag on image or profile)", BLUE)
        self.btn_draw.setCheckable(True)
        self.btn_draw.setToolTip("Left-drag across an off-signal column range "
                                 "on the image or the profile.\nStays armed "
                                 "until unticked; untick to drag x0.")
        self.btn_draw.toggled.connect(self._arm_draw)
        gl.addWidget(self.btn_draw, 5, 0, 1, 2)
        self.btn_refit = _btn("Re-fit", GREEN)
        self.btn_refit.clicked.connect(self._refit)
        gl.addWidget(self.btn_refit, 6, 0, 1, 2)
        self.btn_apply_all = _btn("Apply bg/W to all")
        gl.addWidget(self.btn_apply_all, 7, 0, 1, 2)
        self.btn_skip = _btn("Toggle skip", RED)
        self.btn_skip.clicked.connect(self._toggle_skip)
        gl.addWidget(self.btn_skip, 8, 0, 1, 2)
        side.addWidget(g)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setMinimumHeight(190)
        self.info.setLineWrapMode(QTextEdit.NoWrap)
        self.info.setStyleSheet(f"background:{MID}; color:{TEXT}; border:1px solid {DIM};"
                                f"font-family:monospace; font-size:9px;")
        side.addWidget(self.info)
        side.addStretch()

        sw = QWidget(); sw.setLayout(side)

        self.canvas = ShotCanvas(self)
        self.canvas.band_drawn.connect(self._band)
        self.canvas.center_moved.connect(self._center_dragged)
        L.addWidget(_split(sw, self.canvas, side_px=400))

    def set_shots(self, shots):
        self.shots = shots
        self.lst.clear()
        for s in shots:
            it = QListWidgetItem(self._label(s))
            self.lst.addItem(it)
        self._recolor()
        if shots:
            self.lst.setCurrentRow(0)

    def _label(self, s):
        tag = "REF " if s.is_ref else "    "
        if s.error:
            st = "ERR  "
        elif s.no_signal:
            st = "NOSIG"
        elif s.skipped:
            st = "SKIP "
        elif s.ok:
            st = "OK   "
        else:
            st = "…    "
        mark = "*" if s.x0_manual is not None else " "
        return f"{tag}{st}{mark}{s.name}"

    def _recolor(self):
        for i, s in enumerate(self.shots):
            it = self.lst.item(i)
            it.setText(self._label(s))
            if s.error:
                col = RED
            elif s.no_signal:
                col = YELL
            elif s.skipped:
                col = DIM
            else:
                col = BLUE if s.is_ref else (GREEN if s.ok else TEXT)
            it.setForeground(QColor(col))

    def _select(self, row):
        if row < 0 or row >= len(self.shots):
            return
        s = self.shots[row]
        self.cur = s
        if s.bg_cols:
            self.sp_bg0.setValue(s.bg_cols[0]); self.sp_bg1.setValue(s.bg_cols[1])
        if s.cols and s.fit:
            self.sp_W.setValue(0.5 * (s.cols[1] - 1 - s.cols[0]))
        if s.row_gate:
            self.sp_r0.setValue(s.row_gate[0]); self.sp_r1.setValue(s.row_gate[1])
        self.chk_x0.blockSignals(True); self.sp_x0.blockSignals(True)
        self.chk_x0.setChecked(s.x0_manual is not None)
        self.sp_x0.setEnabled(s.x0_manual is not None)
        self.sp_x0.setValue(s.x0_manual if s.x0_manual is not None
                            else (s.fit.get("x0", 0.0) if s.fit else 0.0))
        self.chk_x0.blockSignals(False); self.sp_x0.blockSignals(False)
        self.refresh()

    def _arm_draw(self, on):
        self.canvas.set_draw_mode(on)
        self.btn_draw.setText("Draw bg band  ▶ ARMED" if on
                              else "Draw bg band  (drag on image or profile)")

    def _center_dragged(self, x):
        """Centre line released → adopt as the manual x0 for THIS shot only."""
        if self.cur is None:
            return
        self.cur.x0_manual = float(x)
        for wdg, val in ((self.chk_x0, True), (self.sp_x0, float(x))):
            wdg.blockSignals(True)
            (wdg.setChecked if wdg is self.chk_x0 else wdg.setValue)(val)
            wdg.blockSignals(False)
        self.sp_x0.setEnabled(True)
        self.refresh()

    def _x0_mode(self, on):
        self.sp_x0.setEnabled(on)
        if self.cur is None:
            return
        self.cur.x0_manual = float(self.sp_x0.value()) if on else None
        self.refresh()

    def _x0_typed(self, v):
        if self.cur is None or not self.chk_x0.isChecked():
            return
        self.cur.x0_manual = float(v)
        self.refresh()

    def _band(self, a, b):
        # stays armed so a bad band can be redrawn immediately; untick the
        # button (or hit Re-fit) when done — x0 dragging needs it unticked
        self.sp_bg0.setValue(a); self.sp_bg1.setValue(b)
        self._refit()

    def overrides(self):
        r1 = int(self.sp_r1.value())
        return dict(bg_cols=(int(self.sp_bg0.value()), int(self.sp_bg1.value())),
                    half_width=float(self.sp_W.value()),
                    row_gate=(int(self.sp_r0.value()), r1 if r1 > 0 else None))

    def _refit(self):
        if self.cur is None:
            return
        self.reanalyze.emit(self.cur, self.overrides())

    def _toggle_skip(self):
        if self.cur is None:
            return
        self.cur.skipped = not self.cur.skipped
        self._recolor()

    def show_result(self, shot, data, meta, res):
        self._cache = (shot, data, meta, res)
        self.canvas.render(data, meta, res, title=shot.name)
        f = res["fit"]
        txt = [f"file      {shot.name}",
               f"ref       {shot.is_ref}",
               f"shot n°   {shot.number}",
               f"DG        {shot.dg_ns:.4f} ns   [{shot.dg_source}]"
               + (f"  cell={shot.dg_raw!r} × {shot.dg_factor:g}"
                  if shot.dg_source == "book" and np.isfinite(shot.dg_factor)
                  else ""),
               f"map       t_abs = τ_abs {'+' if self._dgsign > 0 else '−'} (DG + extra)",
               f"t_abs     [{res['t_abs'][0]:.4f}, {res['t_abs'][-1]:.4f}] ns"
               + ("   ← NOT shifted: DG=0" if shot.dg_ns == 0 else ""),
               f"τ_abs     [{meta['time_axis_abs'][0]:.4f}, "
               f"{meta['time_axis_abs'][-1]:.4f}] ns  "
               f"(origin preserved; span {np.ptp(meta['time_axis_abs']):.4f} ns)",
               f"t axis    {meta['t_source']}  input unit={meta.get('time_unit_input')}"
               + ("  [ASSUMED ns]" if meta.get("time_unit_assumed_ns") else "")
               + ("  rows reversed → increasing time" if meta.get("time_reversed") else ""),
               f"LUT map   {meta.get('t_lut_info', {}).get('pointer', 'none')}  "
               f"{meta.get('t_lut_info', {}).get('mapping', meta.get('t_lut_info', {}).get('reason', ''))}",
               f"sweep     {shot.window_label or 'uncalibrated'}",
               f"half-swap {'APPLIED' if meta.get('half_swap') else 'not applied'}"
               + (f"  (blob frac {meta['swap_scores'][0]:.2f} as-read /"
                  f" {meta['swap_scores'][1]:.2f} swapped)"
                  if np.isfinite(meta.get('swap_scores', (np.nan,))[0]) else "  (forced)"),
               f"dt mean   {meta['dt_mean']*1e3:.4f} ps",
               f"SNR       {res.get('snr', np.nan):.1f}"
               + ("   ← NO SIGNAL" if res.get("no_signal") else ""),
               f"x0 used   {res.get('x0_used', np.nan):.2f} px"
               + ("  (MANUAL)" if res.get("x0_manual") else "  (fit)"),
               f"x0 fit    {f['x0']:.2f} ± {f.get('x0_err', np.nan):.2f} px",
               f"w (1/e²)  {f['w']:.2f} px",
               f"n         {f['n']:.3f}",
               f"FWHM      {f['fwhm']:.2f} px",
               f"pedestal  {f['c']:.1f}",
               f"χ²ᵣ       {f['chi2red']:.2f}",
               f"cols      {res['cols'][0]}–{res['cols'][1]}",
               f"bg cols   {res['bg_cols'][0]}–{res['bg_cols'][1]}"]
        if not np.isnan(shot.t0):
            txt.append(f"t0        {shot.t0:.4f} ns")
        self.info.setText("\n".join(txt))
        self._recolor()

    def refresh(self):
        if self.cur is None:
            return
        self.reanalyze.emit(self.cur, self.overrides())

    def rerender(self):
        """Redraw the cached result (e.g. after a plot-theme change)."""
        if self._cache is None:
            self.canvas.placeholder("Load a shot")
            return
        shot, data, meta, res = self._cache
        self.canvas.render(data, meta, res, title=shot.name)


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 3 — References
# ═════════════════════════════════════════════════════════════════════════════

REF_XMODES = ["absolute  t_abs = τ_abs + signed DG",
              "camera LUT  τ_abs  (DG removed)",
              "aligned on t₀"]


class RefTab(QWidget):
    changed = pyqtSignal()
    dg_edited = pyqtSignal(object, float)     # (Shot, new DG in ns)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.refs = []
        self._updating = False

        L = QHBoxLayout(self)
        side = QVBoxLayout()
        side.addWidget(_lbl("REFERENCE t₀", ACC, 11, True))

        g = _group("Auto-detection")
        gl = QGridLayout(g)
        gl.addWidget(_lbl("criterion"), 0, 0)
        self.cb_crit = _combo(T0_CRITERIA)
        self.cb_crit.currentTextChanged.connect(self._crit_changed)
        gl.addWidget(self.cb_crit, 0, 1)
        gl.addWidget(_lbl("custom rise level", ACC), 1, 0)
        self.sp_rise = _spin(0.01, 99.99, 50.0, dec=2, step=1.0, suffix=" % of peak")
        self.sp_rise.setEnabled(False)
        self.sp_rise.valueChanged.connect(lambda _: self.changed.emit())
        gl.addWidget(self.sp_rise, 1, 1)
        gl.addWidget(_lbl("smoothing (px)"), 2, 0)
        self.sp_sm = _spin(0, 501, 21, dec=0, step=2)
        self.sp_sm.valueChanged.connect(lambda _: self.changed.emit())
        gl.addWidget(self.sp_sm, 2, 1)
        gl.addWidget(_lbl("centroid thresh"), 3, 0)
        self.sp_ct = _spin(0.01, 0.99, 0.2, dec=2, step=0.05)
        self.sp_ct.valueChanged.connect(lambda _: self.changed.emit())
        gl.addWidget(self.sp_ct, 3, 1)
        gl.addWidget(_lbl("level is a fraction of the SMOOTHED peak, found by\n"
                          "walking back from it. Lower = nearer the true foot\n"
                          "but noisier; higher = stabler but biased late by the\n"
                          "risetime. Check the spread in the t₀ column.",
                          DIM, 9), 4, 0, 1, 2)
        side.addWidget(g)

        g2 = _group("Offsets")
        g2l = QGridLayout(g2)
        g2l.addWidget(_lbl("global ref offset", YELL), 0, 0)
        self.sp_glob = _spin(-1e6, 1e6, 0.0, dec=4, step=0.01, suffix=" ns")
        self.sp_glob.valueChanged.connect(lambda _: self.changed.emit())
        g2l.addWidget(self.sp_glob, 0, 1)
        g2l.addWidget(_lbl("per-ref offset"), 1, 0)
        self.sp_per = _spin(-1e6, 1e6, 0.0, dec=4, step=0.01, suffix=" ns")
        self.sp_per.valueChanged.connect(self._set_per)
        g2l.addWidget(self.sp_per, 1, 1)
        side.addWidget(g2)

        g3 = _group("Display")
        g3l = QGridLayout(g3)
        g3l.addWidget(_lbl("x axis", ACC), 0, 0)
        self.cb_xmode = _combo(REF_XMODES)
        self.cb_xmode.currentTextChanged.connect(lambda _: self._redraw())
        g3l.addWidget(self.cb_xmode, 0, 1)
        self.chk_rnorm = _check("normalise each to its peak", False)
        self.chk_rnorm.toggled.connect(lambda _: self._redraw())
        g3l.addWidget(self.chk_rnorm, 1, 0, 1, 2)
        g3l.addWidget(_lbl("'camera LUT τ_abs' removes DG + extra but preserves\n"
                           "the calibrated LUT origin. References recorded on\n"
                           "different sweep ranges are not interchangeable: the\n"
                           "internal trigger→sweep delay depends on Time Range.",
                           DIM, 9), 2, 0, 1, 2)
        side.addWidget(g3)

        self.btn_redo = _btn("Re-detect all", GREEN)
        self.btn_redo.clicked.connect(lambda: self.changed.emit())
        side.addWidget(self.btn_redo)

        side.addWidget(_lbl("DG (ns) is editable — double-click.\n"
                            "References are usually absent from the\n"
                            "shotbook; a value entered here outranks it.\n"
                            "'sweep' is read from the HPD-TA Time Range header;\n"
                            "shots first match references on the SAME sweep.\n"
                            "'win lo/hi' (ns, DG scale) then assigns a reference\n"
                            "to an explicit time window: any shot whose DG\n"
                            "falls inside [lo, hi] uses THIS reference,\n"
                            "regardless of which is nearer in DG. Leave both\n"
                            "blank to fall back to nearest-DG matching.",
                            YELL, 9))
        self.tbl = QTableWidget(0, 8)
        self.tbl.setHorizontalHeaderLabels(
            ["ref", "DG (ns)", "sweep", "win lo (ns)", "win hi (ns)",
             "src", "t₀ raw", "t₀ eff"])
        self.tbl.setStyleSheet(f"""
            QTableWidget {{ background:{MID}; color:{TEXT}; border:1px solid {DIM};
                            font-family:monospace; font-size:9px; gridline-color:{DIM}; }}
            QHeaderView::section {{ background:{MID2}; color:{ACC}; border:0;
                                    font-family:monospace; font-size:9px; padding:3px; }}
        """)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.currentCellChanged.connect(lambda r, *a: self._sel(r))
        self.tbl.itemChanged.connect(self._cell_edited)
        side.addWidget(self.tbl, 1)

        sw = QWidget(); sw.setLayout(side)
        self.canvas = PlotCanvas(self)
        L.addWidget(_split(sw, self.canvas, side_px=460))
        self.canvas.clear("No references")

    def _crit_changed(self, txt):
        self.sp_rise.setEnabled("custom" in txt)
        self.changed.emit()

    def _cell_edited(self, item):
        if self._updating:
            return
        col = item.column()
        r = item.row()
        if not (0 <= r < len(self.refs)):
            return
        if col == 1:
            try:
                v = float(item.text().strip().replace(",", "."))
            except ValueError:
                self.update_view(self.refs)          # revert the cell
                return
            self.dg_edited.emit(self.refs[r], v)
        elif col in (3, 4):
            txt = item.text().strip().replace(",", ".")
            if txt == "":
                val = None
            else:
                try:
                    val = float(txt)
                except ValueError:
                    self.update_view(self.refs)       # revert the cell
                    return
            if col == 3:
                self.refs[r].win_lo = val
            else:
                self.refs[r].win_hi = val
            self.update_view(self.refs)
            self.changed.emit()

    def _sel(self, r):
        if 0 <= r < len(self.refs):
            self.sp_per.blockSignals(True)
            self.sp_per.setValue(self.refs[r].ref_offset)
            self.sp_per.blockSignals(False)

    def _set_per(self, v):
        r = self.tbl.currentRow()
        if 0 <= r < len(self.refs):
            self.refs[r].ref_offset = float(v)
            self.changed.emit()

    def params(self):
        return dict(criterion=self.cb_crit.currentText(),
                    smooth=int(self.sp_sm.value()),
                    cent_thresh=float(self.sp_ct.value()),
                    global_offset=float(self.sp_glob.value()),
                    rise_frac=float(self.sp_rise.value()) / 100.0,
                    rise_frac_pct=float(self.sp_rise.value()),
                    xmode=self.cb_xmode.currentText(),
                    norm_peak=self.chk_rnorm.isChecked())

    def _redraw(self):
        self.update_view(self.refs)

    def update_view(self, refs):
        self.refs = refs
        go = self.sp_glob.value()
        self._updating = True
        self.tbl.setRowCount(len(refs))
        for i, r in enumerate(refs):
            eff = r.t0 + go + r.ref_offset
            src = "manual" if r.dg_manual is not None else (
                "book" if r.meta_row else "none")
            has_win = (r.win_lo is not None) and (r.win_hi is not None)
            vals = [r.name, f"{r.dg_ns:.4f}", r.window_label or "?",
                    "" if r.win_lo is None else f"{r.win_lo:.4f}",
                    "" if r.win_hi is None else f"{r.win_hi:.4f}",
                    src, f"{r.t0:.4f}", f"{eff:.4f}"]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(str(v))
                if j in (1, 3, 4):
                    it.setFlags(it.flags() | Qt.ItemIsEditable)
                    it.setForeground(QColor(ACC if (j == 1 and r.dg_manual is not None)
                                            or (j in (3, 4) and has_win) else TEXT))
                else:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                    col = BLUE if j == 0 else TEXT
                    if (j == 2 and r.window_ns is None) or (j == 5 and src == "none"):
                        col = YELL
                    it.setForeground(QColor(col))
                self.tbl.setItem(i, j, it)
        self._updating = False
        ax = self.canvas.ax
        ax.cla(); self.canvas._style()
        PBG, PPANEL, PTEXT, PDIM, PACC = (PLOT['bg'], PLOT['panel'],
                                          PLOT['text'], PLOT['dim'], PLOT['acc'])
        mode = self.cb_xmode.currentText()
        do_norm = self.chk_rnorm.isChecked()

        # ── build the x arrays first so the display unit can adapt ──────────
        curves = []
        for r in refs:
            if r.lineout is None:
                continue
            eff = r.t0 + go + r.ref_offset
            if mode.startswith("camera"):
                # Remove only the applied delay offset; keep the LUT origin.
                x, mark = r.t_abs - r.t_offset, eff - r.t_offset
            elif mode.startswith("aligned"):
                x, mark = r.t_abs - eff, 0.0
            else:
                x, mark = r.t_abs, eff
            y = np.asarray(r.lineout, float)
            if do_norm:
                pk = float(np.nanmax(_smooth(y, int(self.sp_sm.value()))))
                y = y / pk if np.isfinite(pk) and pk > 0 else y
            curves.append((r, x, y, mark))

        if curves:
            lo = min(float(np.nanmin(x)) for _, x, _, _ in curves)
            hi = max(float(np.nanmax(x)) for _, x, _, _ in curves)
            uname = pick_time_unit(max(abs(lo), abs(hi), hi - lo))
        else:
            uname = "ns"
        ufac = PLOT_UNITS[uname]

        drawn_mark = set()
        for r, x, y, mark in curves:
            ax.plot(x * ufac, y, lw=STYLE["lw_ref"],
                    label=f"{r.name} (DG={r.dg_ns:.3f} ns)")
            key = round(float(mark), 12)
            if key not in drawn_mark:          # 'aligned' puts them all on 0
                ax.axvline(mark * ufac, lw=STYLE["lw_marker"], ls="--", color=PACC)
                drawn_mark.add(key)

        auto_x = {0: f"t_abs ({uname})",
                  1: f"τ_abs = t_abs − signed(DG + extra) ({uname})",
                  2: f"t − t₀ ({uname})"}[self.cb_xmode.currentIndex()]
        ax.set_xlabel(_lab("ref_x", auto_x, unit=uname), color=PTEXT,
                      fontsize=STYLE["label_size"])
        ax.set_ylabel(_lab("ref_y", "Σ_x counts / peak" if do_norm
                           else "Σ_x counts", unit=uname),
                      color=PTEXT, fontsize=STYLE["label_size"])
        if LABELS["ref_title"].strip():
            ax.set_title(LABELS["ref_title"], color=PACC,
                         fontsize=STYLE["title_size"])
        apply_scale(ax, "y", STYLE["yscale_ref"])
        if STYLE["grid"]:
            ax.grid(color=PDIM, lw=STYLE["grid_lw"], alpha=STYLE["grid_alpha"])
        if refs and STYLE["legend"]:
            ax.legend(fontsize=STYLE["legend_size"], facecolor=PPANEL,
                      edgecolor=PDIM, labelcolor=PTEXT, loc=STYLE["legend_loc"])
        self.canvas.draw_idle()


# ═════════════════════════════════════════════════════════════════════════════
#  Tab 4 — Groups
# ═════════════════════════════════════════════════════════════════════════════

DT_MODELS = ["camera t₀ + keep signed DG   (τ_abs − t₀,cam) ± DG",
             "absolute t₀   t_abs − t₀,abs"]


class GroupTab(QWidget):
    replot = pyqtSignal()
    regroup = pyqtSignal()       # changes the group KEYS, not just the drawing

    def __init__(self, parent=None):
        super().__init__(parent)
        self.groups = {}
        self.group_offsets = {}   # label -> extra Δt shift, ns (manual, per group)
        self._updating = False

        L = QHBoxLayout(self)
        side = QVBoxLayout()
        side.addWidget(_lbl("GROUPS", ACC, 11, True))

        g = _group("Delay handling")
        gl = QGridLayout(g)
        self.rb_interp = _check("interpolate onto common Δt grid", True)
        gl.addWidget(self.rb_interp, 0, 0, 1, 2)
        gl.addWidget(_lbl("same-delay tol."), 1, 0)
        self.sp_tol = _spin(0.0, 1e4, 0.001, dec=4, step=0.001, suffix=" ns")
        gl.addWidget(self.sp_tol, 1, 1)
        gl.addWidget(_lbl("Δt zero", ACC), 2, 0)
        self.cb_t0mode = _combo(DT_MODELS)
        self.cb_t0mode.currentTextChanged.connect(lambda _: self.regroup.emit())
        gl.addWidget(self.cb_t0mode, 2, 1)
        gl.addWidget(_lbl("grid range"), 3, 0)
        self.cb_mode = _combo(["intersection", "union"])
        gl.addWidget(self.cb_mode, 3, 1)
        gl.addWidget(_lbl("grid dt (0 → finest)"), 4, 0)
        self.sp_dt = _spin(0.0, 100, 0.0, dec=5, step=0.001, suffix=" ns")
        gl.addWidget(self.sp_dt, 4, 1)
        gl.addWidget(_lbl("Δt display unit", ACC), 5, 0)
        self.cb_tunit = _combo(["auto"] + list(PLOT_UNITS.keys()))
        self.cb_tunit.currentTextChanged.connect(lambda _: self.replot.emit())
        gl.addWidget(self.cb_tunit, 5, 1)
        gl.addWidget(_lbl("x-range"), 6, 0)
        self.cb_xr = _combo(["full", "zoom to signal"])
        self.cb_xr.currentTextChanged.connect(lambda _: self.replot.emit())
        gl.addWidget(self.cb_xr, 6, 1)
        gl.addWidget(_lbl("'camera t₀ + keep signed DG' realises\n"
                          "Δt = (τ_abs − t₀,cam) ± DG. The common extra delay\n"
                          "is removed, while the delay scan survives.\n"
                          "'absolute t₀' subtracts the DG-loaded\n"
                          "fiducial and the delay cancels per matched ref.",
                          DIM, 9), 7, 0, 1, 2)
        side.addWidget(g)

        ge = _group("Energy matching")
        gel = QGridLayout(ge)
        gel.addWidget(_lbl("column", ACC), 0, 0)
        self.cb_ecol = _combo(["(none)"])
        self.cb_ecol.currentTextChanged.connect(lambda _: self.regroup.emit())
        gel.addWidget(self.cb_ecol, 0, 1)
        gel.addWidget(_lbl("relative margin ±"), 1, 0)
        self.sp_emargin = _spin(0.0, 100.0, 5.0, dec=2, step=0.5, suffix=" %")
        self.sp_emargin.valueChanged.connect(lambda _: self.regroup.emit())
        gel.addWidget(self.sp_emargin, 1, 1)
        gel.addWidget(_lbl("method"), 2, 0)
        self.cb_emode = _combo(CLUSTER_MODES)
        self.cb_emode.currentTextChanged.connect(lambda _: self.regroup.emit())
        gel.addWidget(self.cb_emode, 2, 1)
        self.lbl_ecl = _lbl("", GREEN, 9)
        self.lbl_ecl.setWordWrap(True)
        gel.addWidget(self.lbl_ecl, 3, 0, 1, 2)
        side.addWidget(ge)

        gn = _group("Normalisation")
        gnl = QGridLayout(gn)
        gnl.addWidget(_lbl("mode", ACC), 0, 0)
        self.cb_norm = _combo(NORM_MODES)
        self.cb_norm.currentTextChanged.connect(self._norm_changed)
        gnl.addWidget(self.cb_norm, 0, 1)
        gnl.addWidget(_lbl("Δt window"), 1, 0)
        hw = QHBoxLayout()
        self.sp_nw0 = _spin(-1e6, 1e6, -0.5, dec=4, step=0.05, suffix=" ns")
        self.sp_nw1 = _spin(-1e6, 1e6, 0.5, dec=4, step=0.05, suffix=" ns")
        for w_ in (self.sp_nw0, self.sp_nw1):
            w_.setEnabled(False)
            w_.valueChanged.connect(lambda _: self.replot.emit())
            hw.addWidget(w_)
        wnw = QWidget(); wnw.setLayout(hw)
        gnl.addWidget(wnw, 1, 1)
        self.lbl_norm = _lbl("", YELL, 9)
        self.lbl_norm.setWordWrap(True)
        gnl.addWidget(self.lbl_norm, 2, 0, 1, 2)
        side.addWidget(gn)

        self.chk_show_shots = _check("show individual shots", True)
        self.chk_show_band = _check("show ±1σ band", True)
        for c in (self.chk_show_shots, self.chk_show_band):
            c.toggled.connect(lambda _: self.replot.emit())
            side.addWidget(c)
        for wdg in (self.rb_interp, self.sp_tol, self.cb_mode, self.sp_dt):
            sig = wdg.toggled if isinstance(wdg, QCheckBox) else (
                wdg.currentTextChanged if isinstance(wdg, QComboBox) else wdg.valueChanged)
            sig.connect(lambda *_: self.replot.emit())

        hb_rebuild = QHBoxLayout()
        self.btn_rebuild = _btn("Rebuild groups", GREEN)
        hb_rebuild.addWidget(self.btn_rebuild)
        self.btn_reload_refs = _btn("Reload refs", BLUE)
        self.btn_reload_refs.setToolTip(
            "Re-run t0 detection on the reference lineouts (Reference tab "
            "criterion/smoothing/offsets), then rebuild groups. Use this if "
            "reference settings changed but the reference tab was not "
            "revisited — group Δt=0 depends on t0_ref, which is otherwise "
            "only refreshed from there.")
        hb_rebuild.addWidget(self.btn_reload_refs)
        wrebuild = QWidget(); wrebuild.setLayout(hb_rebuild)
        side.addWidget(wrebuild)

        side.addWidget(_lbl("tick a group to plot it · expand it to include or\n"
                            "exclude individual shots from the mean and ±1σ\n"
                            "(buttons act on the selected group's shots)", TEXT, 9))
        hb_sel = QHBoxLayout()
        for txt, fn in (("all", lambda: self._set_all(True)),
                        ("none", lambda: self._set_all(False)),
                        ("invert", self._invert)):
            b_ = _btn(txt, BLUE, w=58)
            b_.clicked.connect(fn)
            hb_sel.addWidget(b_)
        hb_sel.addStretch()
        wsel = QWidget(); wsel.setLayout(hb_sel)
        side.addWidget(wsel)

        hb_search = QHBoxLayout()
        self.ed_search = QLineEdit()
        self.ed_search.setPlaceholderText("find shot (name, substring)…")
        self.ed_search.returnPressed.connect(self._search_shot)
        hb_search.addWidget(self.ed_search)
        b_search = _btn("find", BLUE, w=52)
        b_search.clicked.connect(self._search_shot)
        hb_search.addWidget(b_search)
        wsearch = QWidget(); wsearch.setLayout(hb_search)
        side.addWidget(wsearch)
        self.lbl_search = _lbl("", TEXT, 9)
        self.lbl_search.setWordWrap(True)
        side.addWidget(self.lbl_search)

        self.tree = _tree()
        self.tree.setMinimumHeight(220)
        side.addWidget(self.tree, 1)
        # parent tick = group visible; child tick = shot included in mean/±σ
        self.tree.itemChanged.connect(self._item_toggled)
        self.tree.currentItemChanged.connect(self._sync_delay_spin)

        hb_delay = QHBoxLayout()
        hb_delay.addWidget(_lbl("group delay"))
        self.sp_gdelay = _spin(-1e6, 1e6, 0.0, dec=4, step=0.01, suffix=" ns")
        self.sp_gdelay.setEnabled(False)
        hb_delay.addWidget(self.sp_gdelay)
        b_gdelay_apply = _btn("apply", GREEN, w=52)
        b_gdelay_apply.clicked.connect(self._apply_group_delay)
        hb_delay.addWidget(b_gdelay_apply)
        b_gdelay_clear = _btn("clear", BLUE, w=52)
        b_gdelay_clear.clicked.connect(self._clear_group_delay)
        hb_delay.addWidget(b_gdelay_clear)
        wdelay = QWidget(); wdelay.setLayout(hb_delay)
        side.addWidget(wdelay)
        side.addWidget(_lbl("select a group (or a shot in it) above, set the\n"
                            "offset, 'apply' — it shifts that group's whole Δt\n"
                            "axis (all its shots), on top of t0/DG.", DIM, 9))

        self.warn = _lbl("", YELL, 9)
        self.warn.setWordWrap(True)
        side.addWidget(self.warn)

        sw = QWidget(); sw.setLayout(side)
        self.canvas = PlotCanvas(self)
        L.addWidget(_split(sw, self.canvas, side_px=470))
        self.canvas.clear("No groups")

    def _norm_changed(self, txt):
        win = txt.endswith("window")
        self.sp_nw0.setEnabled(win)
        self.sp_nw1.setEnabled(win)
        notes = {
            "none": "",
            "each shot → peak":
                "±1σ now measures SHAPE scatter only; amplitude jitter (and any "
                "mixed MCP/ND) is divided out. Integrated yields are no longer "
                "comparable between groups.",
            "each shot → area (∫dt)":
                "Area includes the baseline over the whole Δt range — biased on "
                "low-SNR shots. Compares shape at fixed integrated signal.",
            "each shot → mean in Δt window":
                "Level in the window is forced to 1 for every shot; pick a window "
                "where the signal is physically flat.",
            "group mean → peak":
                "Display only — applied AFTER averaging, so ±1σ still carries the "
                "shot-to-shot amplitude jitter.",
        }
        self.lbl_norm.setText(notes.get(txt, ""))
        self.replot.emit()

    def params(self):
        return dict(interpolate=self.rb_interp.isChecked(),
                    tol=float(self.sp_tol.value()),
                    t0_model=self.cb_t0mode.currentText(),
                    mode=self.cb_mode.currentText(),
                    dt=(None if self.sp_dt.value() <= 0 else float(self.sp_dt.value())),
                    time_unit=self.cb_tunit.currentText(),
                    xrange=self.cb_xr.currentText(),
                    show_shots=self.chk_show_shots.isChecked(),
                    show_band=self.chk_show_band.isChecked(),
                    energy_col=self.cb_ecol.currentText(),
                    energy_margin=float(self.sp_emargin.value()) / 100.0,
                    energy_mode=self.cb_emode.currentText(),
                    normalise=self.cb_norm.currentText(),
                    norm_window=(float(self.sp_nw0.value()),
                                 float(self.sp_nw1.value())),
                    group_offsets=dict(self.group_offsets))

    def _tops(self):
        return [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]

    def _item_toggled(self, item, col=0):
        if self._updating:
            return
        shot = item.data(0, Qt.UserRole)
        if shot is not None:                       # a shot row
            shot.excluded = (item.checkState(0) != Qt.Checked)
            self._relabel(item.parent())
        self.replot.emit()

    def _relabel(self, top):
        if top is None:
            return
        lbl = top.data(0, Qt.UserRole + 1)
        n_tot = top.childCount()
        n_on = sum(1 for i in range(n_tot)
                   if top.child(i).checkState(0) == Qt.Checked)
        off = self.group_offsets.get(lbl, 0.0)
        tag = f"   [{n_on}/{n_tot}]" + (f"   Δgroup={off:+.4f} ns" if off else "")
        self._updating = True
        top.setText(0, f"{lbl}{tag}")
        self._updating = False

    def _set_all(self, on):
        """Applies to whatever level is in focus: a selected shot's siblings if
        a shot is selected, otherwise the groups."""
        self._updating = True
        cur = self.tree.currentItem()
        if cur is not None and cur.parent() is not None:
            top = cur.parent()
            for i in range(top.childCount()):
                c = top.child(i)
                c.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
                sh = c.data(0, Qt.UserRole)
                if sh is not None:
                    sh.excluded = not on
            self._updating = False
            self._relabel(top)
        else:
            for t in self._tops():
                t.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
            self._updating = False
        self.replot.emit()

    def _invert(self):
        self._updating = True
        cur = self.tree.currentItem()
        if cur is not None and cur.parent() is not None:
            top = cur.parent()
            for i in range(top.childCount()):
                c = top.child(i)
                on = c.checkState(0) != Qt.Checked
                c.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
                sh = c.data(0, Qt.UserRole)
                if sh is not None:
                    sh.excluded = not on
            self._updating = False
            self._relabel(top)
        else:
            for t in self._tops():
                t.setCheckState(0, Qt.Unchecked
                                if t.checkState(0) == Qt.Checked else Qt.Checked)
            self._updating = False
        self.replot.emit()

    def set_groups(self, items, warnings):
        """items: list of (label, [Shot]) — groups keep their tick across a
        rebuild, new ones default to on."""
        prev = {t.data(0, Qt.UserRole + 1): t.checkState(0) for t in self._tops()}
        expanded = {t.data(0, Qt.UserRole + 1): t.isExpanded() for t in self._tops()}
        live_labels = {lbl for lbl, _ in items}
        self.group_offsets = {k: v for k, v in self.group_offsets.items()
                              if k in live_labels}
        self._updating = True
        self.tree.clear()
        for lbl, shots in items:
            top = QTreeWidgetItem([lbl])
            top.setFlags(top.flags() | Qt.ItemIsUserCheckable)
            top.setCheckState(0, prev.get(lbl, Qt.Checked))
            top.setData(0, Qt.UserRole, None)
            top.setData(0, Qt.UserRole + 1, lbl)
            for sh in shots:
                tag = (f"{sh.name}   DG={sh.dg_ns:.3f} ns   "
                       f"sweep={sh.window_label or 'uncalibrated'}   "
                       f"SNR={sh.snr:.0f}" if np.isfinite(sh.snr) else sh.name)
                c = QTreeWidgetItem([tag])
                c.setFlags(c.flags() | Qt.ItemIsUserCheckable)
                c.setCheckState(0, Qt.Unchecked if sh.excluded else Qt.Checked)
                c.setData(0, Qt.UserRole, sh)
                c.setForeground(0, QColor(TEXT))
                top.addChild(c)
            self.tree.addTopLevelItem(top)
            top.setExpanded(expanded.get(lbl, False))
        self._updating = False
        for t in self._tops():
            self._relabel(t)
        self.warn.setText("\n".join(warnings))
        self._sync_delay_spin()

    def selected(self):
        return [t.data(0, Qt.UserRole + 1) for t in self._tops()
                if t.checkState(0) == Qt.Checked]

    def set_item_color(self, label, hexcol):
        """Tint the group row with the colour its curve is drawn in."""
        self._updating = True
        for t in self._tops():
            if t.data(0, Qt.UserRole + 1) == label:
                t.setForeground(0, QColor(hexcol))
                break
        self._updating = False

    def _search_shot(self):
        """
        Find which group(s) a shot belongs to, by (case-insensitive) substring
        match on Shot.name. Exact match wins if present (disambiguates e.g.
        "042" matching both "shot_042" and "shot_0042"); otherwise all
        substring hits are reported. The first hit is selected/scrolled to
        and its group expanded; every matching row is briefly highlighted.
        """
        q = self.ed_search.text().strip().lower()
        self._updating = True
        for t in self._tops():
            for i in range(t.childCount()):
                t.child(i).setBackground(0, QColor(MID))
        self._updating = False

        if not q:
            self.lbl_search.setText("")
            return

        hits = []  # (group_label, shot_name, tree_item, is_exact)
        for t in self._tops():
            lbl = t.data(0, Qt.UserRole + 1)
            for i in range(t.childCount()):
                c = t.child(i)
                sh = c.data(0, Qt.UserRole)
                if sh is not None and q in sh.name.lower():
                    hits.append((lbl, sh.name, c, sh.name.lower() == q))

        if not hits:
            self.lbl_search.setText(f"no shot matching '{q}'")
            return

        exact = [h for h in hits if h[3]]
        chosen = exact if exact else hits

        self._updating = True
        for lbl, name, item, is_exact in hits:
            item.setBackground(0, QColor(GREEN if is_exact else ACC))
        self._updating = False

        first_item = chosen[0][2]
        first_item.parent().setExpanded(True)
        self.tree.setCurrentItem(first_item)
        self.tree.scrollToItem(first_item)

        if len(hits) == 1:
            self.lbl_search.setText(f"'{hits[0][1]}' → group: {hits[0][0]}")
        else:
            groups = sorted(set(h[0] for h in hits))
            self.lbl_search.setText(
                f"{len(hits)} shots match '{q}' across {len(groups)} group(s): "
                + ", ".join(groups))

    def _current_group_label(self):
        """Group label for whatever is selected: a top-level item directly,
        or the parent group if a shot row is selected."""
        cur = self.tree.currentItem()
        if cur is None:
            return None
        top = cur if cur.parent() is None else cur.parent()
        return top.data(0, Qt.UserRole + 1)

    def _sync_delay_spin(self, *_):
        lbl = self._current_group_label()
        self.sp_gdelay.blockSignals(True)
        if lbl is None:
            self.sp_gdelay.setValue(0.0)
            self.sp_gdelay.setEnabled(False)
        else:
            self.sp_gdelay.setEnabled(True)
            self.sp_gdelay.setValue(self.group_offsets.get(lbl, 0.0))
        self.sp_gdelay.blockSignals(False)

    def _apply_group_delay(self):
        lbl = self._current_group_label()
        if lbl is None:
            return
        v = float(self.sp_gdelay.value())
        if v:
            self.group_offsets[lbl] = v
        else:
            self.group_offsets.pop(lbl, None)
        for t in self._tops():
            if t.data(0, Qt.UserRole + 1) == lbl:
                self._relabel(t)
                break
        self.replot.emit()

    def _clear_group_delay(self):
        lbl = self._current_group_label()
        if lbl is None:
            return
        self.group_offsets.pop(lbl, None)
        self.sp_gdelay.blockSignals(True)
        self.sp_gdelay.setValue(0.0)
        self.sp_gdelay.blockSignals(False)
        for t in self._tops():
            if t.data(0, Qt.UserRole + 1) == lbl:
                self._relabel(t)
                break
        self.replot.emit()

    def group_delay(self, label):
        return self.group_offsets.get(label, 0.0)

    def set_columns(self, cols):
        keep = self.cb_ecol.currentText()
        self.cb_ecol.blockSignals(True)
        self.cb_ecol.clear()
        self.cb_ecol.addItems(["(none)"] + list(cols))
        if keep in (["(none)"] + list(cols)):
            self.cb_ecol.setCurrentText(keep)
        else:
            # preselect an energy-looking column, but never silently group by it
            for c in cols:
                if re.search(r"(?i)\benerg|\bE\s*\(|joule|\bJ\b", str(c)):
                    self.cb_ecol.setCurrentIndex(0)
                    break
        self.cb_ecol.blockSignals(False)

    def set_selection(self, labels):
        want = set(labels)
        tops = self._tops()
        if not any(t.data(0, Qt.UserRole + 1) in want for t in tops):
            return                      # labels no longer exist → keep all ticked
        self._updating = True
        for t in tops:
            t.setCheckState(0, Qt.Checked
                            if t.data(0, Qt.UserRole + 1) in want else Qt.Unchecked)
        self._updating = False
        self.replot.emit()


class StyleDialog(QDialog):
    """Modeless: edits apply to the live figures immediately."""

    def __init__(self, parent, on_change):
        super().__init__(parent)
        self.setWindowTitle("Plot style")
        self.on_change = on_change
        self._busy = False
        self.w = {}
        self.resize(430, 900)

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        L = QVBoxLayout(inner)
        g = _group("Fonts (pt)")
        gl = QGridLayout(g)
        for r, (k, lab, lo, hi) in enumerate([
                ("tick_size", "ticks", 3, 40), ("label_size", "axis labels", 3, 40),
                ("title_size", "title", 3, 40), ("legend_size", "legend", 3, 40),
                ("annot_size", "annotations", 3, 40)]):
            gl.addWidget(_lbl(lab), r, 0)
            self.w[k] = _spin(lo, hi, STYLE[k], dec=1, step=0.5)
            self.w[k].valueChanged.connect(lambda v, kk=k: self._set(kk, v))
            gl.addWidget(self.w[k], r, 1)
        L.addWidget(g)

        g2 = _group("Lines / shading")
        g2l = QGridLayout(g2)
        for r, (k, lab, lo, hi, st) in enumerate([
                ("lw_mean", "group mean", 0.1, 10, 0.1),
                ("lw_shot", "individual shots", 0.0, 10, 0.1),
                ("lw_ref", "reference / lineout", 0.1, 10, 0.1),
                ("lw_prof", "transverse profile", 0.1, 10, 0.1),
                ("lw_marker", "markers / guides", 0.1, 10, 0.1),
                ("band_alpha", "±1σ band alpha", 0.0, 1.0, 0.05),
                ("shot_alpha", "shot alpha", 0.0, 1.0, 0.05),
                ("span_alpha", "bg/window span alpha", 0.0, 1.0, 0.05)]):
            g2l.addWidget(_lbl(lab), r, 0)
            self.w[k] = _spin(lo, hi, STYLE[k], dec=2, step=st)
            self.w[k].valueChanged.connect(lambda v, kk=k: self._set(kk, v))
            g2l.addWidget(self.w[k], r, 1)
        L.addWidget(g2)

        g3 = _group("Colours / decorations")
        g3l = QGridLayout(g3)
        g3l.addWidget(_lbl("group colormap"), 0, 0)
        self.w["group_cmap"] = _combo(GROUP_CMAPS)
        self.w["group_cmap"].setCurrentText(STYLE["group_cmap"])
        self.w["group_cmap"].currentTextChanged.connect(
            lambda v: self._set("group_cmap", v))
        g3l.addWidget(self.w["group_cmap"], 0, 1)
        g3l.addWidget(_lbl("qualitative maps (tab10, Set1…)\nindex by group; "
                           "continuous ones\nsample the ramp", DIM, 9), 1, 0, 1, 2)
        self.w["grid"] = _check("grid", STYLE["grid"])
        self.w["grid"].toggled.connect(lambda v: self._set("grid", v))
        g3l.addWidget(self.w["grid"], 2, 0)
        self.w["legend"] = _check("legend", STYLE["legend"])
        self.w["legend"].toggled.connect(lambda v: self._set("legend", v))
        g3l.addWidget(self.w["legend"], 2, 1)
        g3l.addWidget(_lbl("legend location"), 3, 0)
        self.w["legend_loc"] = _combo(LEGEND_LOCS)
        self.w["legend_loc"].setCurrentText(STYLE["legend_loc"])
        self.w["legend_loc"].currentTextChanged.connect(
            lambda v: self._set("legend_loc", v))
        g3l.addWidget(self.w["legend_loc"], 3, 1)
        for r, (k, lab, lo, hi, st) in enumerate([
                ("grid_alpha", "grid alpha", 0.0, 1.0, 0.05),
                ("grid_lw", "grid width", 0.05, 5, 0.05)], start=4):
            g3l.addWidget(_lbl(lab), r, 0)
            self.w[k] = _spin(lo, hi, STYLE[k], dec=2, step=st)
            self.w[k].valueChanged.connect(lambda v, kk=k: self._set(kk, v))
            g3l.addWidget(self.w[k], r, 1)
        L.addWidget(g3)

        g5 = _group("Axis scale")
        g5l = QGridLayout(g5)
        for r, (k, lab) in enumerate([("yscale_group", "Groups — y"),
                                      ("yscale_ref", "References — y"),
                                      ("yscale_review", "Review — signal axis")]):
            g5l.addWidget(_lbl(lab), r, 0)
            self.w[k] = _combo(SCALES)
            self.w[k].setCurrentText(STYLE[k])
            self.w[k].currentTextChanged.connect(lambda v, kk=k: self._set(kk, v))
            g5l.addWidget(self.w[k], r, 1)
        g5l.addWidget(_lbl("symlog linthresh"), 3, 0)
        self.w["symlog_thresh"] = _spin(1e-9, 1e9, STYLE["symlog_thresh"],
                                        dec=4, step=1.0)
        self.w["symlog_thresh"].valueChanged.connect(
            lambda v: self._set("symlog_thresh", v))
        g5l.addWidget(self.w["symlog_thresh"], 3, 1)
        g5l.addWidget(_lbl("log masks ≤0 samples — background-subtracted\n"
                           "lineouts are negative in the wings, so part of\n"
                           "the trace silently disappears. symlog keeps them.",
                           YELL, 9), 4, 0, 1, 2)
        L.addWidget(g5)

        g6 = _group("Axis labels  (blank → automatic)")
        g6l = QGridLayout(g6)
        self.lab_w = {}
        for r, (k, lab, ph) in enumerate([
                ("group_x", "Groups x", "Δt = t_abs − t₀,ref ({unit})"),
                ("group_y", "Groups y", "Σ_x counts / {norm}"),
                ("group_title", "Groups title", "(none)"),
                ("ref_x", "References x", "t_abs (ns)"),
                ("ref_y", "References y", "Σ_x counts"),
                ("ref_title", "References title", "(none)"),
                ("img_y", "Review image y", "t_abs (ns)"),
                ("prof_x", "Review profile x", "transverse (px)"),
                ("prof_y", "Review profile y", "Σ_t counts"),
                ("line_x", "Review lineout x", "Σ_x counts")]):
            g6l.addWidget(_lbl(lab), r, 0)
            e = _edit(LABELS[k])
            e.setPlaceholderText(ph)
            e.textChanged.connect(lambda v, kk=k: self._set_label(kk, v))
            self.lab_w[k] = e
            g6l.addWidget(e, r, 1)
        g6l.addWidget(_lbl("{unit} → ps/ns/µs · {norm} → normalisation mode",
                           DIM, 9), 10, 0, 1, 2)
        L.addWidget(g6)

        g4 = _group("Export")
        g4l = QGridLayout(g4)
        g4l.addWidget(_lbl("dpi (raster only)"), 0, 0)
        self.w["dpi"] = _spin(50, 1200, STYLE["dpi"], dec=0, step=50)
        self.w["dpi"].valueChanged.connect(lambda v: self._set("dpi", int(v)))
        g4l.addWidget(self.w["dpi"], 0, 1)
        self.w["transparent"] = _check("transparent background", STYLE["transparent"])
        self.w["transparent"].toggled.connect(lambda v: self._set("transparent", v))
        g4l.addWidget(self.w["transparent"], 1, 0)
        self.w["tight"] = _check("tight bounding box", STYLE["tight"])
        self.w["tight"].toggled.connect(lambda v: self._set("tight", v))
        g4l.addWidget(self.w["tight"], 1, 1)
        L.addWidget(g4)

        L.addStretch()
        hb = QHBoxLayout()
        b_def = _btn("Reset defaults", YELL)
        b_def.clicked.connect(self._reset)
        b_close = _btn("Close", GREEN)
        b_close.clicked.connect(self.close)
        hb.addWidget(b_def); hb.addStretch(); hb.addWidget(b_close)
        outer.addLayout(hb)

    def _set(self, key, val):
        if self._busy:
            return
        STYLE[key] = val
        self.on_change()

    def _set_label(self, key, val):
        if self._busy:
            return
        LABELS[key] = val
        self.on_change()

    def _reset(self):
        self._busy = True
        STYLE.update(STYLE_DEFAULTS)
        LABELS.update(LABEL_DEFAULTS)
        for k, e in self.lab_w.items():
            e.setText(LABELS[k])
        for k, wdg in self.w.items():
            if isinstance(wdg, QCheckBox):
                wdg.setChecked(bool(STYLE[k]))
            elif isinstance(wdg, QComboBox):
                wdg.setCurrentText(str(STYLE[k]))
            else:
                wdg.setValue(float(STYLE[k]))
        self._busy = False
        self.on_change()


# ═════════════════════════════════════════════════════════════════════════════
#  Main window
# ═════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Streak — transverse super-Gaussian / lineout / grouping")
        self.resize(1650, 980)

        self.cfg = {}
        self.shots = []
        self.refs = []
        self.group_map = {}     # group label → [Shot]
        self.group_curves = {}  # group label → (grid, mean, std, n_eff)
        self.group_refs = {}    # group label → {(ref_name, match_kind): n_shots}

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border:1px solid {DIM}; }}
            QTabBar::tab {{ background:{MID}; color:{TEXT}; padding:6px 14px;
                            font-family:monospace; font-size:10px; border:1px solid {DIM}; }}
            QTabBar::tab:selected {{ background:{MID2}; color:{ACC}; }}
        """)
        self.setup = SetupTab()
        self.review = ReviewTab()
        self.reft = RefTab()
        self.groupt = GroupTab()
        self.tabs.addTab(self.setup, "1 · Setup")
        self.tabs.addTab(self.review, "2 · Review")
        self.tabs.addTab(self.reft, "3 · References")
        self.tabs.addTab(self.groupt, "4 · Groups")

        central = QWidget()
        v = QVBoxLayout(central)
        v.addWidget(self.tabs, 1)

        bar = QHBoxLayout()
        self.prog = QProgressBar()
        self.prog.setStyleSheet(f"""
            QProgressBar {{ background:{MID}; border:1px solid {DIM}; color:{TEXT};
                            font-family:monospace; font-size:9px; height:16px; }}
            QProgressBar::chunk {{ background:{ACC}; }}
        """)
        self.status = _lbl("ready", size=9)
        self.cb_theme = _combo(["dark plots", "light plots"])
        self.cb_theme.setFixedWidth(110)
        self.cb_theme.currentTextChanged.connect(self._set_theme)
        self.cb_cmap = _combo(["inferno", "viridis", "magma", "plasma",
                               "turbo", "gray", "cividis"])
        self.cb_cmap.setFixedWidth(90)
        self.cb_cmap.currentTextChanged.connect(self._set_cmap)
        self.cb_figfmt = _combo(["png", "pdf", "svg", "eps", "tif"])
        self.cb_figfmt.setFixedWidth(62)
        self.cb_figfmt.setToolTip("Figure format used by Export…")
        b_style = _btn("Plot style…", w=95)
        b_style.clicked.connect(self._open_style)
        b_png = _btn("Save plot…", GREEN, w=95)
        b_png.setToolTip("Save the figure on the current tab (Ctrl+P)")
        b_png.clicked.connect(self._save_plot)
        b_save = _btn("Export…", GREEN, w=100)
        b_sess = _btn("Save session", w=110)
        b_load = _btn("Load session", BLUE, w=110)
        b_save.clicked.connect(self._export)
        b_sess.clicked.connect(self._save_session)
        b_load.clicked.connect(self._load_session)
        bar.addWidget(self.prog, 1)
        bar.addWidget(self.status, 2)
        bar.addWidget(self.cb_theme); bar.addWidget(self.cb_cmap)
        bar.addWidget(self.cb_figfmt); bar.addWidget(b_style); bar.addWidget(b_png)
        bar.addWidget(b_save); bar.addWidget(b_sess); bar.addWidget(b_load)
        v.addLayout(bar)
        self.setCentralWidget(central)

        from PyQt5.QtWidgets import QShortcut
        QShortcut(QKeySequence("Ctrl+P"), self, activated=self._save_plot)

        self.setup.process_requested.connect(self._process_all)
        self.review.reanalyze.connect(self._reanalyze_one)
        self.review.btn_apply_all.clicked.connect(self._apply_all)
        self.reft.changed.connect(self._refresh_refs)
        self.reft.dg_edited.connect(self._ref_dg_edited)
        self.groupt.btn_rebuild.clicked.connect(self._build_groups)
        self.groupt.btn_reload_refs.clicked.connect(self._refresh_refs)
        self.groupt.replot.connect(self._plot_groups)
        self.groupt.regroup.connect(self._build_groups)

    # ── style / single-figure save
    def _open_style(self):
        if getattr(self, "_styledlg", None) is None:
            self._styledlg = StyleDialog(self, self._redraw_all)
        self._styledlg.show()
        self._styledlg.raise_()

    def _current_canvas(self):
        return {1: self.review.canvas, 2: self.reft.canvas,
                3: self.groupt.canvas}.get(self.tabs.currentIndex())

    def _save_plot(self):
        canv = self._current_canvas()
        if canv is None:
            QMessageBox.information(self, "Save plot",
                                    "Switch to Review, References or Groups — "
                                    "the Setup tab has no figure.")
            return
        name = {1: "review", 2: "references", 3: "groups"}[self.tabs.currentIndex()]
        # An instance dialog, not getSaveFileName(): the filter must REWRITE the
        # filename's suffix as it is chosen. Otherwise the prefilled "…​.png"
        # survives a switch to the PDF filter and matplotlib writes a PNG.
        dlg = QFileDialog(self, "Save plot")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)  # filterSelected fires
        dlg.setNameFilters(SAVE_FILTERS)
        dlg.setDefaultSuffix("png")
        dlg.selectFile(f"{name}.png")

        def _sync(flt):
            m = re.search(r"\*(\.\w+)", flt or "")
            if not m:
                return
            ext = m.group(1)
            sel_ = dlg.selectedFiles()
            base = (os.path.splitext(os.path.basename(sel_[0]))[0]
                    if sel_ and sel_[0] else name)
            dlg.setDefaultSuffix(ext.lstrip("."))
            dlg.selectFile(base + ext)

        dlg.filterSelected.connect(_sync)
        if dlg.exec_() != QFileDialog.Accepted:
            return
        path = dlg.selectedFiles()[0]
        path, fmt = resolve_save_format(path, dlg.selectedNameFilter())
        try:
            canv.fig.savefig(
                path, format=fmt, dpi=int(STYLE["dpi"]),
                facecolor=("none" if STYLE["transparent"] else PLOT["bg"]),
                transparent=bool(STYLE["transparent"]),
                bbox_inches=("tight" if STYLE["tight"] else None))
        except Exception as e:
            QMessageBox.critical(self, "Save plot", f"{e}")
            return
        vector = fmt in ("pdf", "svg", "eps", "ps")
        self.status.setText(f"plot → {path}  [{fmt}]"
                            + ("  (vector — dpi ignored)" if vector
                               else f"  {STYLE['dpi']} dpi")
                            + (", transparent" if STYLE["transparent"] else ""))

    # ── theming
    def _set_theme(self, txt):
        set_plot_theme("light" if txt.startswith("light") else "dark")
        self.cb_cmap.blockSignals(True)
        self.cb_cmap.setCurrentText(PLOT["cmap"])
        self.cb_cmap.blockSignals(False)
        self._redraw_all()

    def _set_cmap(self, name):
        PLOT["cmap"] = name
        self._redraw_all()

    def _redraw_all(self):
        self.review.rerender()
        self.reft.update_view(self.refs)
        if self.group_map:
            self._plot_groups()

    # ── caching reader
    def _read(self, shot):
        key = (shot.path, self.cfg.get("half_swap", "auto"))
        if getattr(self, "_rk", None) == key:
            return self._rd, self._rm
        data, meta = load_streak(shot.path, half_swap=self.cfg.get("half_swap", "auto"))
        self._rk, self._rd, self._rm = key, data, meta
        return data, meta

    # ── batch
    def _process_all(self):
        self.cfg = self.setup.config()
        shots, refs = getattr(self.setup, "_scanned", ([], []))
        if not shots:
            QMessageBox.warning(self, "Process", "Run 'Scan files' first.")
            return
        self._capture_live()                    # keep what is already on screen
        self._apply_pending_pre(refs + shots)   # overrides, manual x0, manual DG
        self._attach_shotbook(shots + refs)     # ... then the book, which yields to them
        allsh = refs + shots
        self.prog.setMaximum(len(allsh))
        for i, s in enumerate(allsh):
            self.prog.setValue(i + 1)
            QApplication.processEvents()
            self._analyze(s, self.cfg)
        self.shots = [s for s in allsh if not s.is_ref]
        self.refs = [s for s in allsh if s.is_ref]
        self._apply_pending_post(allsh)         # skip flags + ref offsets
        self.review.set_shots(allsh)
        self._detect_t0()
        self._build_groups()
        self._restore_ui()
        n_ok = sum(s.ok for s in allsh)
        n_err = sum(bool(s.error) for s in allsh)
        n_ns = sum(s.no_signal for s in allsh)
        self.status.setText(f"processed {n_ok}/{len(allsh)} ok, {n_err} errors, "
                            f"{n_ns} no-signal (SNR < {self.cfg.get('snr_min', 10):g})")
        self.tabs.setCurrentIndex(1)

    def _attach_shotbook(self, shots):
        df = self.setup.df
        if df is None:
            for s in shots:
                if s.dg_manual is not None:
                    s.dg_ns, s.dg_source = float(s.dg_manual), "manual"
                else:
                    s.dg_ns, s.dg_source = 0.0, "no-book"
            self._dg_report(shots)
            return
        sc, dc = self.cfg["shot_col"], self.cfg["dg_col"]
        idx = {}
        for _, row in df.iterrows():
            n = _to_float(row.get(sc, np.nan))
            if np.isfinite(n):
                idx[int(n)] = row
        for s in shots:
            row = idx.get(s.number)
            s.meta_row = {c: row[c] for c in df.columns} if row is not None else {}
            # a manually entered delay is user input and outranks the shotbook —
            # references are typically absent from the book, so falling back to
            # 0 here is what was wiping them on every Scan.
            if s.dg_manual is not None:
                s.dg_ns, s.dg_source = float(s.dg_manual), "manual"
                continue
            if row is None:
                s.dg_ns, s.dg_source, s.dg_raw = 0.0, "no-row", None
                continue
            s.dg_raw = row.get(dc, None)
            val = _to_float(s.dg_raw)
            fac = DG_UNITS.get(self.cfg.get("dg_unit", DEFAULT_DG_UNIT),
                               DG_UNITS[DEFAULT_DG_UNIT])
            s.dg_factor = float(fac)
            if not np.isfinite(val):
                s.dg_ns, s.dg_source = 0.0, "unparsed"
            else:
                s.dg_ns, s.dg_source = val * fac, "book"
        self._dg_report(shots)

    def _dg_report(self, shots):
        from collections import Counter
        c = Counter(s.dg_source for s in shots)
        bad = [s for s in shots if s.dg_source in ("unparsed", "no-row", "no-book")]
        msg = "DG: " + ", ".join(f"{v} {k}" for k, v in sorted(c.items()))
        if bad:
            ex = bad[0]
            msg += (f"  ⚠ {len(bad)} shot(s) fell back to DG=0 "
                    f"(e.g. {ex.name}: {ex.dg_source}"
                    + (f", cell={ex.dg_raw!r}" if ex.dg_raw is not None else "") + ")")
        self.status.setText(msg)
        return c

    @staticmethod
    def _same(a, b):
        try:
            if a is None or b is None:
                return a is b
            if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
                return [None if x is None else float(x) for x in a] == \
                       [None if x is None else float(x) for x in b]
            return float(a) == float(b)
        except Exception:
            return a == b

    def _analyze(self, s, cfg, overrides=None):
        # overrides=None → fall back to whatever this shot already carries
        # (that is what makes a restored session reproduce itself on re-process)
        o = dict(overrides) if overrides is not None else dict(s.ov or {})
        if overrides is not None:
            diff = {k: v for k, v in overrides.items()
                    if not self._same(v, cfg.get(k))}
            s.ov = dict(overrides) if diff else None
        try:
            data, meta = self._read(s)
            s.window_ns, s.window_label = window_fingerprint(
                meta.get("time_range", np.nan), meta.get("time_unit"))
            res = analyze_shot(
                data, meta,
                bg_cols=o.get("bg_cols", cfg["bg_cols"]),
                half_width_px=o.get("half_width", cfg["half_width"]),
                row_gate=o.get("row_gate", cfg["row_gate"]),
                dg_ns=s.dg_ns, extra_ns=cfg["extra_ns"],
                n_fixed=cfg["n_fixed"],
                snr_min=cfg.get("snr_min", 10.0),
                x0_override=s.x0_manual,
                dg_sign=cfg.get("dg_sign", 1.0))
            s.t_offset = cfg.get("dg_sign", 1.0) * (float(s.dg_ns)
                                                     + float(cfg["extra_ns"]))
            s.snr = res["snr"]
            s.no_signal = res["no_signal"]
            s.x0_used = res["x0_used"]
            # auto-skip only ever SETS the flag: an explicit un-skip by the user
            # survives a re-fit, otherwise the toggle would be unusable.
            if s.no_signal and cfg.get("auto_skip", True):
                s.skipped = True
            s.fit = res["fit"]
            s.t_abs = res["t_abs"]
            s.lineout = res["lineout"]
            s.cols = res["cols"]
            s.bg_cols = res["bg_cols"]
            s.row_gate = res["row_gate"]
            s.error = ""
            return data, meta, res
        except Exception as e:
            s.error = f"{type(e).__name__}: {e}"
            s.error_tb = traceback.format_exc()
            s.lineout = None
            print(s.error_tb, file=sys.stderr)
            return None, None, None

    def _reanalyze_one(self, shot, overrides):
        if not self.cfg:
            self.cfg = self.setup.config()
        data, meta, res = self._analyze(shot, self.cfg, overrides)
        self.review._dgsign = self.cfg.get("dg_sign", 1.0)
        if res is None:
            self.review.canvas.placeholder(f"{shot.name}\n{shot.error}")
            self.review.info.setText(f"ERROR  {shot.error}\n\n{shot.error_tb}")
            self.review._recolor()
            self.status.setText(f"{shot.name}: {shot.error}")
            return
        if shot.is_ref:
            self._detect_t0(only=shot)
        self.review.show_result(shot, data, meta, res)

    def _apply_all(self):
        o = self.review.overrides()
        self.cfg.update(o)
        self.setup.sp_bg0.setValue(o["bg_cols"][0])
        self.setup.sp_bg1.setValue(o["bg_cols"][1])
        self.setup.sp_W.setValue(o["half_width"])
        allsh = self.refs + self.shots
        self.prog.setMaximum(len(allsh))
        for i, s in enumerate(allsh):
            self.prog.setValue(i + 1)
            QApplication.processEvents()
            self._analyze(s, self.cfg, o)
        self._detect_t0()
        self._build_groups()
        self.review._recolor()
        self.review.refresh()
        self.status.setText("overrides applied to all shots")

    # ── references
    def _detect_t0(self, only=None):
        p = self.reft.params()
        targets = [only] if only is not None else self.refs
        for r in targets:
            if r is None or r.lineout is None:
                continue
            kw = dict(criterion=p["criterion"], smooth_win=p["smooth"],
                      cent_thresh=p["cent_thresh"], rise_frac=p["rise_frac"])
            t0, info = detect_t0(r.t_abs, r.lineout, **kw)
            r.t0 = t0
            # Camera-frame fiducial: detect on τ_abs = t_abs − signed(DG+extra).
            # Group camera mode later restores only signed DG, never extra_ns.
            t0c, _ = detect_t0(r.t_abs - r.t_offset, r.lineout, **kw)
            r.t0_cam = t0c
        self.reft.update_view(self.refs)

    def _ref_dg_edited(self, shot, value):
        """DG changes t_abs = τ_abs + signed(DG + extra), so the reference
        must be re-reduced, not merely re-labelled."""
        shot.dg_manual = float(value)
        shot.dg_ns = float(value)
        if not self.cfg:
            self.cfg = self.setup.config()
        self._analyze(shot, self.cfg)
        self._detect_t0()
        self._build_groups()
        sg = self.cfg.get("dg_sign", 1.0)
        self.status.setText(f"{shot.name}: DG set manually to {value:.4f} ns "
                            f"(t_abs = τ_abs {'+' if sg > 0 else '−'} (DG + extra))")

    def _refresh_refs(self):
        self._detect_t0()
        self._build_groups()

    def _t0_for(self, shot, model=None):
        """
        Return the effective reference fiducial for *shot*.

        Selection is intentionally two-stage:
          1. Restrict references to the same HPD-TA sweep window (Time Range),
             because the camera trigger-to-sweep delay is range-dependent.
          2. Inside that sweep-compatible pool, use explicit DG windows when
             provided, otherwise nearest DG.

        If no same-window reference exists, the nearest-DG fallback is retained
        only so the GUI can still display something; the match kind is
        ``no-window-match`` and the Groups tab raises a prominent warning.
        """
        gp_model = model or self.groupt.params()["t0_model"]
        key = "t0_cam" if gp_model.startswith("camera") else "t0"
        cand = [r for r in self.refs
                if np.isfinite(getattr(r, key, np.nan))]
        if not cand:
            return 0.0, None, "none"
        go = self.reft.params()["global_offset"]

        same = [r for r in cand if same_window(r.window_ns, shot.window_ns)]
        sweep_ok = bool(same)
        pool = same if sweep_ok else cand

        windowed = [r for r in pool
                    if r.win_lo is not None and r.win_hi is not None]
        inside = [r for r in windowed
                  if min(r.win_lo, r.win_hi) <= shot.dg_ns
                  <= max(r.win_lo, r.win_hi)]
        if inside:
            best = min(inside, key=lambda r: abs(r.dg_ns - shot.dg_ns))
            if not sweep_ok:
                kind = "no-window-match"
            else:
                kind = "window" if len(inside) == 1 else "window-ambiguous"
            return getattr(best, key) + go + best.ref_offset, best, kind

        unwindowed = [r for r in pool if r not in windowed]
        fallback_pool = unwindowed if unwindowed else pool
        best = min(fallback_pool, key=lambda r: abs(r.dg_ns - shot.dg_ns))
        kind = "nearest-dg" if sweep_ok else "no-window-match"
        return getattr(best, key) + go + best.ref_offset, best, kind

    def _relative_time_for(self, shot, model=None, group_offset=0.0):
        """One timing path shared by group plots and exported per-shot CSVs."""
        gp_model = model or self.groupt.params()["t0_model"]
        t0, ref, kind = self._t0_for(shot, model=gp_model)
        t = relative_time_axis(
            shot.t_abs, shot.t_offset, shot.dg_ns, t0, gp_model,
            dg_sign=self.cfg.get("dg_sign", 1.0),
            group_offset=group_offset)
        return t, t0, ref, kind

    # ── groups
    def _group_label(self, s, cols):
        parts = [f"{c}={s.meta_row.get(c, 'NA')}" for c in cols]
        if s.e_label:
            parts.append(s.e_label)
        return " | ".join(parts) if parts else "all shots"

    def _assign_energy_clusters(self, gp):
        """Cluster the energy column by relative margin, over ALL usable shots
        so the bins are identical across groups and remain comparable."""
        ecol = gp["energy_col"]
        oks = [s for s in self.shots if s.ok]
        for s in self.shots:
            s.e_label = ""
        if not ecol or ecol == "(none)" or not oks:
            self.groupt.lbl_ecl.setText("")
            return
        vals = np.array([_to_float(s.meta_row.get(ecol, np.nan)) for s in oks])
        ids, centers, spans = cluster_relative(vals, gp["energy_margin"],
                                               gp["energy_mode"])
        m = gp["energy_margin"] * 100.0
        for s, i in zip(oks, ids):
            s.e_label = (f"{ecol}≈{centers[int(i)]:.4g}±{m:g}%" if i >= 0
                         else f"{ecol}=NA")
        n_na = int(np.sum(ids < 0))
        lines = [f"{len(centers)} energy cluster(s) from {len(oks) - n_na} shots"
                 + (f", {n_na} with no/invalid {ecol}" if n_na else "")]
        for c in sorted(centers, key=lambda k: centers[k]):
            lo, hi, rel = spans[c]
            n = int(np.sum(ids == c))
            lines.append(f"  {centers[c]:.4g}  n={n}  [{lo:.4g}, {hi:.4g}]  "
                         f"spread ±{100 * rel / 2:.2f}%")
        self.groupt.lbl_ecl.setText("\n".join(lines))

    def _build_groups(self):
        if not self.shots:
            return
        gp = self.groupt.params()
        if self.setup.df is not None:
            self.groupt.set_columns(list(self.setup.df.columns))
        self._assign_energy_clusters(gp)
        cols = self.cfg.get("group_cols", [])
        sens = self.cfg.get("sens_cols", [])
        gmap = {}
        for s in self.shots:
            if not s.ok:
                continue
            lbl = self._group_label(s, cols)
            if not gp["interpolate"]:
                tol = max(gp["tol"], 1e-12)
                lbl += f" | DG≈{round(s.dg_ns / tol) * tol:.4f}ns"
            gmap.setdefault(lbl, []).append(s)

        warnings = []
        ecol = gp["energy_col"]
        for lbl, ss in gmap.items():
            if ecol and ecol != "(none)":
                ev = np.array([_to_float(s.meta_row.get(ecol, np.nan)) for s in ss])
                ev = ev[np.isfinite(ev)]
                if ev.size and ev.mean() != 0:
                    rel = (ev.max() - ev.min()) / (0.5 * (ev.max() + ev.min()))
                    if rel > 2 * gp["energy_margin"] + 1e-12:
                        warnings.append(f"⚠ '{lbl}': energy spread ±{100*rel/2:.2f}% "
                                        f"exceeds the ±{100*gp['energy_margin']:g}% "
                                        f"margin")
                if any(not np.isfinite(_to_float(s.meta_row.get(ecol, np.nan)))
                       for s in ss):
                    warnings.append(f"⚠ '{lbl}': contains shots with no valid "
                                    f"{ecol} — they are pooled under '=NA'")
            for c in sens:
                vals = {str(s.meta_row.get(c, "NA")) for s in ss}
                if len(vals) > 1:
                    warnings.append(f"⚠ '{lbl}': mixed {c} = {sorted(vals)} "
                                    f"— amplitudes are NOT comparable")
            dgs = np.asarray([s.dg_ns for s in ss], dtype=float)
            # np.ptp(), not dgs.ptp(): the ndarray method was removed in NumPy 2.0
            span = float(np.ptp(dgs)) if dgs.size else 0.0
            if gp["interpolate"] and span > gp["tol"]:
                warnings.append(f"ℹ '{lbl}': ΔDG span {span:.4f} ns "
                                f"> tol → interpolated onto a common grid")
        self.group_map = gmap
        self._compute_group_curves()
        for lbl, refcount in self.group_refs.items():
            names_used = sorted({name for (name, kind) in refcount
                                 if name != "(none)"})
            n_ambiguous = sum(n for (name, kind), n in refcount.items()
                              if kind == "window-ambiguous")
            n_nearest = sum(n for (name, kind), n in refcount.items()
                            if kind == "nearest-dg")
            n_window = sum(n for (name, kind), n in refcount.items()
                          if kind == "window")
            n_nowin = sum(n for (name, kind), n in refcount.items()
                          if kind == "no-window-match")
            if n_nowin:
                shots_here = self.group_map.get(lbl, [])
                sweep_labels = sorted({s.window_label or "uncalibrated"
                                       for s in shots_here})
                warnings.append(
                    f"⚠ '{lbl}': {n_nowin} shot(s) have no reference on the "
                    f"same HPD-TA sweep window ({', '.join(sweep_labels)}). "
                    f"Nearest-DG across a different Time Range was used only "
                    f"as a visible fallback; Δt is not trustworthy. Record or "
                    f"select a reference acquired on the same sweep setting.")
            elif len(names_used) > 1:
                detail = ", ".join(
                    f"{name}×{sum(n for (nm, k), n in refcount.items() if nm == name)}"
                    for name in names_used)
                warnings.append(
                    f"⚠ '{lbl}': shots matched to different references "
                    f"({detail}) — Δt=0 is not the same fiducial for every "
                    f"shot in this group; assign explicit DG windows on the "
                    f"Reference tab so each shot's DG lands in exactly one "
                    f"window")
            elif not names_used:
                warnings.append(f"⚠ '{lbl}': no reference matched any shot "
                                f"(t0_ref=0 used) — Δt is uncalibrated")
            elif n_ambiguous:
                warnings.append(
                    f"⚠ '{lbl}': {n_ambiguous} shot(s) fall inside more than "
                    f"one reference's assigned DG window (overlapping "
                    f"windows) — nearest-DG used as tiebreak; narrow the "
                    f"window bounds on the Reference tab")
            elif n_window and n_nearest:
                warnings.append(
                    f"ℹ '{lbl}': {n_window} shot(s) matched by assigned "
                    f"window, {n_nearest} fell outside all assigned windows "
                    f"and used nearest-DG instead — same reference either "
                    f"way, but check the window bounds cover this group's DG "
                    f"if that is not intended")
        self.groupt.set_groups(
            [(k, sorted(gmap[k], key=lambda s: (s.number, s.name)))
             for k in sorted(gmap.keys())], warnings)
        self._plot_groups()

    def _compute_group_curves(self):
        """
        Per-shot normalisation must happen BEFORE the average, otherwise the
        mean is dominated by the brightest shot and ±1σ measures amplitude
        jitter instead of shape. 'group mean → peak' is the exception: it is a
        display rescale applied afterwards, in _plot_groups.
        """
        gp = self.groupt.params()
        mode = gp["normalise"]
        self.group_curves = {}
        self.group_traces = {}
        self.norm_factors = {}
        self.group_refs = {}      # lbl -> {(ref_name, match_kind): n_shots}
        bad = []
        for lbl, ss in self.group_map.items():
            used = [s for s in ss if not s.excluded]
            traces, facs = [], {}
            refcount = {}
            goff = self.groupt.group_delay(lbl)
            for s in used:
                t, t0, ref, kind = self._relative_time_for(
                    s, model=gp["t0_model"], group_offset=goff)
                rname = ref.name if ref is not None else "(none)"
                key = (rname, kind)
                refcount[key] = refcount.get(key, 0) + 1
                y, f = normalize_trace(t, s.lineout, mode,
                                       smooth_win=self.reft.params()["smooth"],
                                       window=gp["norm_window"])
                if not np.isfinite(f):
                    bad.append(s.name)
                facs[s.name] = f
                traces.append((t, y))
            self.group_traces[lbl] = list(zip(used, traces))
            self.norm_factors[lbl] = facs
            self.group_refs[lbl] = refcount
            if not traces:
                self.group_curves[lbl] = None      # every shot unticked
                continue
            try:
                self.group_curves[lbl] = group_average(traces, mode=gp["mode"],
                                                       dt=gp["dt"])
            except Exception as e:
                self.group_curves[lbl] = None
                self.status.setText(f"group '{lbl}': {e}")
        if bad:
            self.status.setText(f"⚠ normalisation factor ~0 or NaN on "
                                f"{len(bad)} shot(s): {', '.join(bad[:4])}"
                                f"{' …' if len(bad) > 4 else ''} — left unscaled")

    def _plot_groups(self):
        if not self.group_map:
            return
        self._compute_group_curves()
        gp = self.groupt.params()
        DARK, MID, TEXT, DIM, ACC, GREEN, BLUE, YELL, RED = plot_colors()
        ax = self.groupt.canvas.ax
        ax.cla(); self.groupt.canvas._style()
        sel = self.groupt.selected()
        if not sel:
            ax.text(0.5, 0.5, "no groups ticked", color=DIM, ha="center",
                    va="center", transform=ax.transAxes, family="monospace")
            self.groupt.canvas.draw_idle()
            return

        # ── pick the display timescale from what is actually on screen ──────
        lo, hi = np.inf, -np.inf
        for lbl in sel:
            gc = self.group_curves.get(lbl)
            if gc is None:
                continue
            lo, hi = min(lo, gc[0][0]), max(hi, gc[0][-1])
        if not np.isfinite(lo):
            lo, hi = 0.0, 1.0
        uname = gp["time_unit"]
        if uname == "auto":
            uname = pick_time_unit(max(abs(lo), abs(hi), hi - lo))
        fac = PLOT_UNITS[uname]

        zoom_lo, zoom_hi = np.inf, -np.inf
        for k, lbl in enumerate(sorted(sel)):
            gc = self.group_curves.get(lbl)
            if gc is None:
                continue
            grid, mean, std, n_eff = gc
            col = group_color(k, len(sel))
            # group-level rescale is display-only and applies after averaging
            gnorm = 1.0
            if gp["normalise"] == "group mean → peak":
                pk = np.nanmax(mean)
                gnorm = float(pk) if np.isfinite(pk) and pk > 0 else 1.0
            norm = gnorm
            if gp["show_shots"]:
                for s, (t, y) in self.group_traces.get(lbl, []):
                    ax.plot(t * fac, y / norm, color=col,
                            lw=STYLE["lw_shot"], alpha=STYLE["shot_alpha"])
            n_used = len(self.group_traces.get(lbl, []))
            n_tot = len(self.group_map[lbl])
            refcount = self.group_refs.get(lbl, {})
            names_used = sorted({name for (name, kind) in refcount
                                 if name != "(none)"})
            if len(names_used) == 1:
                name = names_used[0]
                kinds = {kind for (nm, kind) in refcount if nm == name}
                tag = ("win" if kinds == {"window"} else
                       "amb⚠" if "window-ambiguous" in kinds else
                       "sweep⚠" if "no-window-match" in kinds else
                       "near" if kinds == {"nearest-dg"} else "mix")
                reftag = f"  ref={name}({tag})"
            elif len(names_used) > 1:
                reftag = "  ref=MIXED⚠"
            else:
                reftag = "  ref=none⚠"
            ax.plot(grid * fac, mean / norm, color=col, lw=STYLE["lw_mean"],
                    label=f"{lbl}  (n={n_used}" +
                          (f"/{n_tot}" if n_used != n_tot else "") + ")" + reftag)
            self.groupt.set_item_color(lbl, matplotlib.colors.to_hex(col))
            if gp["show_band"]:
                ax.fill_between(grid * fac, (mean - std) / norm,
                                (mean + std) / norm, color=col,
                                alpha=STYLE["band_alpha"], lw=0)
            # signal extent for the optional zoom: >5 % of this group's peak
            with np.errstate(invalid="ignore"):
                m = np.isfinite(mean) & (mean > 0.05 * np.nanmax(mean))
            if m.any():
                zoom_lo = min(zoom_lo, grid[m][0])
                zoom_hi = max(zoom_hi, grid[m][-1])

        if gp["xrange"] == "zoom to signal" and np.isfinite(zoom_lo):
            pad = 0.1 * max(zoom_hi - zoom_lo, 1e-9)
            ax.set_xlim((zoom_lo - pad) * fac, (zoom_hi + pad) * fac)
        xauto = (f"Δt = (τ_abs − t₀,cam) "
                 f"{'+' if self.cfg.get('dg_sign', 1.0) > 0 else '−'} DG ({uname})"
                 if gp["t0_model"].startswith("camera") else
                 f"Δt = t_abs − t₀,abs ({uname})")
        ax.set_xlabel(_lab("group_x", xauto,
                           unit=uname, norm=gp["normalise"]),
                      color=TEXT, fontsize=STYLE["label_size"])
        ylab = {"none": "Σ_x counts",
                "each shot → peak": "Σ_x counts / shot peak",
                "each shot → area (∫dt)": "Σ_x counts / shot area  (ns⁻¹)",
                "each shot → mean in Δt window":
                    f"Σ_x counts / mean over [{gp['norm_window'][0]:g}, "
                    f"{gp['norm_window'][1]:g}] ns",
                "group mean → peak": "Σ_x counts / group peak"}
        ax.set_ylabel(_lab("group_y", ylab.get(gp["normalise"], "Σ_x counts"),
                           unit=uname, norm=gp["normalise"]),
                      color=TEXT, fontsize=STYLE["label_size"])
        if LABELS["group_title"].strip():
            ax.set_title(_lab("group_title", "", unit=uname, norm=gp["normalise"]),
                         color=ACC, fontsize=STYLE["title_size"])
        apply_scale(ax, "y", STYLE["yscale_group"])
        if gp["normalise"] == "each shot → mean in Δt window":
            for xw in gp["norm_window"]:
                ax.axvline(xw * fac, color=YELL, lw=STYLE["lw_marker"], ls=":")
        ax.axvline(0, color=DIM, lw=STYLE["lw_marker"], ls="--")
        if STYLE["grid"]:
            ax.grid(color=DIM, lw=STYLE["grid_lw"], alpha=STYLE["grid_alpha"])
        if sel and STYLE["legend"]:
            ax.legend(fontsize=STYLE["legend_size"], facecolor=MID,
                      edgecolor=DIM, labelcolor=TEXT, loc=STYLE["legend_loc"])
        self.groupt.canvas.draw_idle()

    def _restore_ui(self):
        pend = getattr(self, "_pending", None)
        if not pend:
            return
        sel = pend.get("group_selection") or []
        if sel:
            self.groupt.set_selection(sel)
        ui = pend.get("ui", {})
        key = ui.get("review_shot")
        if key:
            for i, s in enumerate(self.refs + self.shots):
                if self._shot_key(s) == key:
                    self.review.lst.setCurrentRow(i)
                    break
        if ui.get("tab") is not None:
            self.tabs.setCurrentIndex(int(ui["tab"]))
        n_ov = sum(1 for s in self.refs + self.shots if s.ov)
        n_x0 = sum(1 for s in self.refs + self.shots if s.x0_manual is not None)
        n_sk = sum(1 for s in self.refs + self.shots if s.skipped)
        self.status.setText(self.status.text() +
                            f"  |  session: {n_ov} overrides, {n_x0} manual x0, "
                            f"{n_sk} skipped restored")
        self._pending = None

    # ── export
    def _export(self):
        if not self.shots:
            QMessageBox.warning(self, "Export", "Nothing to export.")
            return
        out = QFileDialog.getExistingDirectory(self, "Output directory")
        if not out:
            return
        os.makedirs(os.path.join(out, "shots"), exist_ok=True)
        os.makedirs(os.path.join(out, "groups"), exist_ok=True)
        keep = self.cfg.get("keep_cols", [])
        rows = []
        allsh = self.refs + self.shots
        self.prog.setMaximum(len(allsh) + len(self.group_map))
        k = 0
        for s in allsh:
            k += 1; self.prog.setValue(k); QApplication.processEvents()
            if not s.ok:
                continue
            dt_axis, t0, ref, kind = self._relative_time_for(s)
            df = pd.DataFrame({"t_abs_ns": s.t_abs,
                               "dt_ns": dt_axis,
                               "lineout_counts": s.lineout})
            tag = f"{'ref_' if s.is_ref else ''}shot_{s.number if s.number >= 0 else s.name}"
            path = os.path.join(out, "shots", f"{tag}.csv")
            with open(path, "w") as f:
                f.write(f"# file={s.name}\n# DG_ns={s.dg_ns}\n"
                        f"# SNR={s.snr:.3f} no_signal={s.no_signal}\n"
                        f"# x0_used={s.x0_used:.4f} manual={s.x0_manual is not None}\n"
                        f"# extra_ns={self.cfg['extra_ns']}\n"
                        f"# t0_model={self.groupt.params()['t0_model']}\n"
                        f"# sweep_window={s.window_label or 'uncalibrated'}\n"
                        f"# t0_ref={t0}  ref={ref.name if ref else 'none'}  "
                        f"ref_sweep={ref.window_label if ref else 'none'}  "
                        f"match={kind}\n"
                        f"# fit A={s.fit['A']:.6g} x0={s.fit['x0']:.4f} "
                        f"w={s.fit['w']:.4f} n={s.fit['n']:.4f} c={s.fit['c']:.6g} "
                        f"FWHM={s.fit['fwhm']:.4f} chi2red={s.fit['chi2red']:.4f}\n"
                        f"# cols={s.cols} bg_cols={s.bg_cols} row_gate={s.row_gate}\n")
                df.to_csv(f, index=False)
            r = dict(file=s.name, shot=s.number, is_ref=s.is_ref, dg_ns=s.dg_ns,
                     dg_sign=self.cfg.get("dg_sign", 1.0),
                     extra_ns=self.cfg.get("extra_ns", 0.0),
                     t0_model=self.groupt.params()["t0_model"],
                     sweep_window=s.window_label or "uncalibrated",
                     sweep_window_ns=s.window_ns,
                     t0_ref=t0, ref=(ref.name if ref else "none"),
                     ref_sweep_window=(ref.window_label if ref else "none"),
                     ref_match=kind,
                     snr=s.snr, no_signal=s.no_signal,
                     x0_used=s.x0_used, x0_manual=(s.x0_manual is not None),
                     x0=s.fit["x0"], x0_err=s.fit.get("x0_err", np.nan),
                     w=s.fit["w"], n=s.fit["n"], fwhm_px=s.fit["fwhm"],
                     A=s.fit["A"], c=s.fit["c"], chi2red=s.fit["chi2red"],
                     col0=s.cols[0], col1=s.cols[1],
                     e_cluster=s.e_label, excluded=s.excluded,
                     group=self._group_label(s, self.cfg.get("group_cols", [])))
            for c in keep:
                r[c] = s.meta_row.get(c, np.nan)
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(out, "summary.csv"), index=False)

        for lbl, gc in self.group_curves.items():
            k += 1; self.prog.setValue(k); QApplication.processEvents()
            if gc is None:
                continue
            grid, mean, std, n_eff = gc
            safe = re.sub(r"[^\w\-=.]+", "_", lbl)
            gp = self.groupt.params()
            fpath = os.path.join(out, "groups", f"{safe}.csv")
            with open(fpath, "w") as fh:
                used = [sh.name for sh, _ in self.group_traces.get(lbl, [])]
                drop = [sh.name for sh in self.group_map.get(lbl, [])
                        if sh.excluded]
                fh.write(f"# group={lbl}\n# shots_used={used}\n"
                         f"# shots_excluded={drop}\n"
                         f"# timing_model={gp['t0_model']}\n"
                         f"# group_time_offset_ns={gp['group_offsets'].get(lbl, 0.0)}\n"
                         f"# normalisation={gp['normalise']}\n"
                         f"# grid={gp['mode']} dt={gp['dt']}\n")
                for nm, fv in self.norm_factors.get(lbl, {}).items():
                    fh.write(f"# norm_factor {nm} = {fv:.6g}\n")
                pd.DataFrame({"dt_ns": grid, "mean": mean,
                              "std": std, "n_shots": n_eff}).to_csv(fh, index=False)

        ext = self.cb_figfmt.currentText()
        for canv, fn in ((self.groupt.canvas, os.path.join(out, "groups", "overlay")),
                         (self.reft.canvas, os.path.join(out, "references"))):
            canv.fig.savefig(f"{fn}.{ext}", format=KNOWN_EXT.get("." + ext, ext),
                             dpi=int(STYLE["dpi"]),
                             facecolor=("none" if STYLE["transparent"] else PLOT["bg"]),
                             transparent=bool(STYLE["transparent"]),
                             bbox_inches=("tight" if STYLE["tight"] else None))
        self._write_session(os.path.join(out, "session.json"))
        self.status.setText(f"exported → {out}")

    def _shot_key(self, s):
        return ("ref:" if s.is_ref else "shot:") + s.name

    def _shot_state(self):
        """Everything about a shot the USER decided; derived quantities (fit,
        SNR, no_signal) are deliberately not stored — they are recomputed."""
        out = {}
        for s in self.refs + self.shots:
            st = dict(skipped=bool(s.skipped), excluded=bool(s.excluded))
            if s.x0_manual is not None:
                st["x0_manual"] = float(s.x0_manual)
            if s.dg_manual is not None:
                st["dg_manual"] = float(s.dg_manual)
            if s.is_ref and s.ref_offset:
                st["ref_offset"] = float(s.ref_offset)
            if s.is_ref and s.win_lo is not None:
                st["win_lo"] = float(s.win_lo)
            if s.is_ref and s.win_hi is not None:
                st["win_hi"] = float(s.win_hi)
            if s.ov:
                st["overrides"] = dict(s.ov)
            out[self._shot_key(s)] = st
        return out

    def _capture_live(self):
        """
        Scan rebuilds every Shot from disk, so anything held on the old objects
        would be dropped. Fold the live state into _pending first: it is merged
        ON TOP of a loaded session, i.e. what is on screen wins over the file.
        """
        pend = dict(getattr(self, "_pending", None) or {})
        if self.refs or self.shots:
            merged = dict(pend.get("shots", {}))
            merged.update(self._shot_state())
            pend["shots"] = merged
            offs = dict(pend.get("ref_offsets", {}))
            offs.update({r.name: float(r.ref_offset) for r in self.refs})
            pend["ref_offsets"] = offs
        sel = self.groupt.selected()
        if sel:
            pend["group_selection"] = sel
        self._pending = pend

    def _session_dict(self):
        cur = getattr(self.review, "cur", None)
        return dict(
            version=2,
            created=__import__("datetime").datetime.now().isoformat(timespec="seconds"),
            setup=self.setup.config(),
            refs=self.reft.params(),
            ref_offsets={r.name: float(r.ref_offset) for r in self.refs},
            groups=self.groupt.params(),
            group_selection=self.groupt.selected(),
            dg_sign=self.cfg.get("dg_sign", 1.0),
            plots=dict(theme=self.cb_theme.currentText(),
                       cmap=self.cb_cmap.currentText(),
                       style=dict(STYLE), labels=dict(LABELS),
                       fig_format=self.cb_figfmt.currentText()),
            ui=dict(tab=self.tabs.currentIndex(),
                    review_shot=(self._shot_key(cur) if cur else None)),
            shots=self._shot_state(),
        )

    def _write_session(self, path):
        def default(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            return str(o)
        with open(path, "w") as f:
            json.dump(self._session_dict(), f, indent=2, default=default)

    def _save_session(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save session", "session.json",
                                           "JSON (*.json)")
        if p:
            self._write_session(p)
            d = self._session_dict()
            self.status.setText(
                f"session → {p}  ({len(d['shots'])} shot states, "
                f"{sum(1 for v in d['shots'].values() if v.get('overrides'))} overrides, "
                f"{sum(1 for v in d['shots'].values() if v.get('skipped'))} skipped)")

    def _load_session(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load session", "", "JSON (*.json)")
        if not p:
            return
        try:
            with open(p) as f:
                sess = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Session", f"{e}")
            return

        self.setup.apply_config(sess.get("setup", {}))

        rp = sess.get("refs", {})
        self.reft.sp_rise.setEnabled("custom" in rp.get("criterion", ""))
        for wdg, key, dflt in ((self.reft.cb_crit, "criterion", "rise 50%"),
                               (self.reft.sp_sm, "smooth", 21),
                               (self.reft.sp_ct, "cent_thresh", 0.2),
                               (self.reft.sp_glob, "global_offset", 0.0),
                               (self.reft.sp_rise, "rise_frac_pct", 50.0),
                               (self.reft.cb_xmode, "xmode", REF_XMODES[0]),
                               (self.reft.chk_rnorm, "norm_peak", False)):
            wdg.blockSignals(True)
            if isinstance(wdg, QCheckBox):
                wdg.setChecked(bool(rp.get(key, dflt)))
            elif isinstance(wdg, QComboBox):
                wdg.setCurrentText(rp.get(key, dflt))
            else:
                wdg.setValue(rp.get(key, dflt))
            wdg.blockSignals(False)

        gp = sess.get("groups", {})
        for wdg, key, dflt in ((self.groupt.rb_interp, "interpolate", True),
                               (self.groupt.sp_tol, "tol", 0.001),
                               (self.groupt.cb_mode, "mode", "intersection"),
                               (self.groupt.sp_dt, "dt", 0.0),
                               (self.groupt.cb_t0mode, "t0_model", DT_MODELS[0]),
                               (self.groupt.cb_ecol, "energy_col", "(none)"),
                               (self.groupt.cb_emode, "energy_mode", CLUSTER_MODES[0]),
                               (self.groupt.cb_tunit, "time_unit", "auto"),
                               (self.groupt.cb_xr, "xrange", "full"),
                               (self.groupt.chk_show_shots, "show_shots", True),
                               (self.groupt.chk_show_band, "show_band", True)):
            v = gp.get(key, dflt)
            wdg.blockSignals(True)
            if isinstance(wdg, QCheckBox):
                wdg.setChecked(bool(v))
            elif isinstance(wdg, QComboBox):
                wdg.setCurrentText(v)
            else:
                wdg.setValue(float(v or 0.0))
            wdg.blockSignals(False)

        self.groupt.sp_emargin.blockSignals(True)
        self.groupt.sp_emargin.setValue(100.0 * float(gp.get("energy_margin", 0.05)))
        self.groupt.sp_emargin.blockSignals(False)

        self.groupt.group_offsets = {str(k): float(v) for k, v
                                     in gp.get("group_offsets", {}).items()}

        nm = gp.get("normalise", "none")
        if isinstance(nm, bool):                 # v2 sessions stored a checkbox
            nm = "group mean → peak" if nm else "none"
        self.groupt.cb_norm.blockSignals(True)
        self.groupt.cb_norm.setCurrentText(nm if nm in NORM_MODES else "none")
        self.groupt.cb_norm.blockSignals(False)
        self.groupt._norm_changed(self.groupt.cb_norm.currentText())
        nw = gp.get("norm_window", (-0.5, 0.5))
        self.groupt.sp_nw0.blockSignals(True); self.groupt.sp_nw1.blockSignals(True)
        self.groupt.sp_nw0.setValue(float(nw[0])); self.groupt.sp_nw1.setValue(float(nw[1]))
        self.groupt.sp_nw0.blockSignals(False); self.groupt.sp_nw1.blockSignals(False)

        pl = sess.get("plots", {})
        self.cb_theme.blockSignals(True)
        self.cb_theme.setCurrentText(pl.get("theme", "dark plots"))
        self.cb_theme.blockSignals(False)
        set_plot_theme("light" if pl.get("theme", "").startswith("light") else "dark")
        self.cb_cmap.blockSignals(True)
        self.cb_cmap.setCurrentText(pl.get("cmap", PLOT["cmap"]))
        self.cb_cmap.blockSignals(False)
        PLOT["cmap"] = pl.get("cmap", PLOT["cmap"])
        st = pl.get("style") or {}
        STYLE.update({k: v for k, v in st.items() if k in STYLE_DEFAULTS})
        lb = pl.get("labels") or {}
        LABELS.update({k: str(v) for k, v in lb.items() if k in LABEL_DEFAULTS})
        self.cb_figfmt.blockSignals(True)
        self.cb_figfmt.setCurrentText(pl.get("fig_format", "png"))
        self.cb_figfmt.blockSignals(False)
        dlg = getattr(self, "_styledlg", None)
        if dlg is not None:
            dlg.close()          # its widgets hold the OLD values; rebuild it
            self._styledlg = None

        self._pending = sess
        n = len(sess.get("shots", {}))
        ans = QMessageBox.question(
            self, "Session",
            f"Session restored ({n} shot states).\n\n"
            "Re-scan the folders and re-process now?\n"
            "(No = settings only; press Scan + Process all yourself.)",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if ans == QMessageBox.Yes:
            self.setup._scan()
            self._process_all()
        else:
            self.status.setText(f"session loaded from {p} "
                                f"— Scan + Process all to apply shot states")

    # ── applying restored per-shot state ───────────────────────────────────
    def _apply_pending_pre(self, allsh):
        """Before analysis: overrides + manual x0 (they change the result)."""
        pend = getattr(self, "_pending", None)
        if not pend:
            return
        st = pend.get("shots", {})
        for s in allsh:
            d = st.get(self._shot_key(s))
            if not d:
                continue
            if d.get("x0_manual") is not None:
                s.x0_manual = float(d["x0_manual"])
            if d.get("dg_manual") is not None:
                s.dg_manual = float(d["dg_manual"])
            if d.get("ref_offset") is not None:
                s.ref_offset = float(d["ref_offset"])
            if d.get("win_lo") is not None:
                s.win_lo = float(d["win_lo"])
            if d.get("win_hi") is not None:
                s.win_hi = float(d["win_hi"])
            ov = d.get("overrides")
            if ov:
                o = dict(ov)
                if "bg_cols" in o:
                    o["bg_cols"] = tuple(int(x) for x in o["bg_cols"])
                if "row_gate" in o:
                    o["row_gate"] = tuple(None if x is None else int(x)
                                          for x in o["row_gate"])
                if "half_width" in o:
                    o["half_width"] = float(o["half_width"])
                s.ov = o

    def _apply_pending_post(self, allsh):
        """After analysis: skip flags (must win over auto-skip), ref offsets,
        group selection, tab/selection state."""
        pend = getattr(self, "_pending", None)
        if not pend:
            return
        st = pend.get("shots", {})
        for s in allsh:
            d = st.get(self._shot_key(s))
            if d is not None and "skipped" in d:
                s.skipped = bool(d["skipped"])
            if d is not None and "excluded" in d:
                s.excluded = bool(d["excluded"])
        offs = pend.get("ref_offsets", {})
        for r in self.refs:
            if r.name in offs:
                r.ref_offset = float(offs[r.name])


def _install_excepthook():
    """
    Qt swallows exceptions raised inside slots (they only reach stderr, and on
    Windows there is often no console). Surface them instead.
    """
    def hook(etype, value, tb):
        txt = "".join(traceback.format_exception(etype, value, tb))
        print(txt, file=sys.stderr)
        try:
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle("Unhandled exception")
            box.setText(f"{etype.__name__}: {value}")
            box.setDetailedText(txt)
            box.exec_()
        except Exception:
            pass
    sys.excepthook = hook


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(dark_palette())
    app.setFont(QFont("monospace", 9))
    _install_excepthook()
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
