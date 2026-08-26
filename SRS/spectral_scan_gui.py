"""
Hamamatsu streak camera — spectral scan GUI (hybrid: auto + manual override)
------------------------------------------------------------------------------
Workflow:
1. Select scan folder (scan_folder/<shot_id>/<shot_id>.img) and shotbook
   (.xlsx, columns: shot-id e.g. "Shot n°", and a scan-parameter column).
2. Click "Run batch" — every shot is auto-analyzed with spectral_scan_core:
   axis calibration (LUT-aware), background subtraction, artifact-row
   screening, 2-peak detection, adaptive centroid tracking.
3. Shots that auto-detected cleanly are marked OK; shots where <2 peaks
   were found, an artifact row was flagged, or a large fraction of the
   trace failed the SNR gate are marked NEEDS REVIEW and sorted to the
   top of the shot list.
4. Click a shot to inspect: top panel = 2D streak (time x wavelength)
   with the two tracked traces overlaid; bottom panel = time-integrated
   spectrum with the two peak windows shaded.
5. To override: select which peak (1 or 2) you're editing, then
   click-drag on the spectrum panel to redefine that peak's window.
   Click "Re-run this shot" to retrack with the manual window — this
   only replaces auto-detection, the tracking algorithm (adaptive
   centroid, artifact screening, non-crossing constraint) is unchanged.
6. All CSV/PNG outputs are written incrementally as each shot is run.

Run: python spectral_scan_gui.py
"""

import sys
import os
import json

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
    QFrame, QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QGroupBox, QDoubleSpinBox, QLineEdit, QMessageBox, QSplitter,
    QButtonGroup, QRadioButton, QTabWidget, QComboBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import spectral_scan_core as ssc

# ─────────────────────────────────────────────────────────────────────────────
# Style (matches the original streak_batch.py dark theme)
# ─────────────────────────────────────────────────────────────────────────────

DARK, MID, MID2 = "#0d0d0d", "#191919", "#222222"
ACC, DIM, TEXT, GREEN, BLUE = "#e87040", "#444444", "#aaaaaa", "#4fc97e", "#4fa8e8"
RED = "#e84f4f"


def _lbl(text, color=TEXT, size=10, bold=False):
    l = QLabel(text)
    w = "bold;" if bold else ""
    l.setStyleSheet(f"color:{color}; font-size:{size}px; font-family:monospace; {w}")
    return l


def _hsep():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet(f"color:{DIM};"); return f


def _btn(text, color=ACC, w=None):
    b = QPushButton(text)
    if w: b.setFixedWidth(w)
    b.setStyleSheet(f"""
        QPushButton {{ background:#2a2a2a; color:{color}; border:1px solid {color};
            font-size:11px; font-family:monospace; border-radius:3px; padding:4px 10px; }}
        QPushButton:hover {{ background:{color}; color:#000; }}
        QPushButton:disabled {{ color:#444; border-color:#333; }}
    """)
    return b


def _line_edit(text=""):
    e = QLineEdit(text)
    e.setStyleSheet("""
        QLineEdit { background:#252525; color:#fff; border:1px solid #555;
            padding:3px 6px; font-size:11px; font-family:monospace; border-radius:3px; }
    """)
    return e


def _spin(lo, hi, val, decimals=2, step=1.0):
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setDecimals(decimals); s.setSingleStep(step); s.setValue(val)
    s.setStyleSheet("""
        QDoubleSpinBox { background:#252525; color:#fff; border:1px solid #555;
            padding:3px 6px; font-size:11px; font-family:monospace; border-radius:3px; }
    """)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Canvas: streak image (top) + integrated spectrum with editable windows (bottom)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(facecolor=DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax_streak = self.fig.add_axes([0.10, 0.42, 0.82, 0.53])
        self.ax_spec   = self.fig.add_axes([0.10, 0.10, 0.82, 0.26])
        self._style_axes()

        self.result = None
        self.active_peak = "peak1"       # which window the user is currently editing
        self._drag_start = None
        self._pending_window = {}        # label -> (left_nm, right_nm) while dragging

        self.mpl_connect("button_press_event",   self._on_press)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("motion_notify_event",  self._on_motion)

        self.window_edited_cb = None     # callback(label, left_nm, right_nm)
        self._draw_placeholder()

    def _style_axes(self):
        for ax in [self.ax_streak, self.ax_spec]:
            ax.set_facecolor(DARK)
            for sp in ax.spines.values():
                sp.set_edgecolor(DIM)
            ax.tick_params(colors=TEXT, labelsize=8)
        self.fig.patch.set_facecolor(DARK)

    def _draw_placeholder(self):
        self.ax_streak.cla(); self.ax_spec.cla(); self._style_axes()
        self.ax_streak.text(0.5, 0.5, "Run batch or select a shot", transform=self.ax_streak.transAxes,
                             ha="center", va="center", color=DIM, fontsize=11, fontfamily="monospace")
        self.ax_streak.set_xticks([]); self.ax_streak.set_yticks([])
        self.ax_spec.set_xticks([]); self.ax_spec.set_yticks([])
        self.draw()

    def load_result(self, result):
        self.result = result
        self._pending_window = {}
        self._render()

    def set_active_peak(self, label):
        self.active_peak = label

    def _render(self):
        self.ax_streak.cla(); self.ax_spec.cla(); self._style_axes()
        r = self.result
        if r is None:
            self._draw_placeholder(); return

        lam = r["lambda_axis"]; t = r["time_axis"]
        vmax = np.percentile(r["data_bg"], 99.5)
        if vmax <= 0:
            vmax = r["data_bg"].max() or 1.0
        self.ax_streak.imshow(r["data_bg"], aspect="auto", origin="lower", cmap="inferno",
                               extent=[lam[0], lam[-1], t[0], t[-1]], vmin=0, vmax=vmax)
        colors = {"peak1": ACC, "peak2": BLUE}
        for label, tr in r["traces"].items():
            self.ax_streak.plot(tr["lambda_nm"], t, color=colors.get(label, GREEN), lw=1.1, label=label)
        self.ax_streak.set_ylabel("Time (ns)", color=TEXT, fontsize=8)
        self.ax_streak.legend(loc="upper right", fontsize=7, facecolor=MID2, labelcolor=TEXT)
        self.ax_streak.set_title(os.path.basename(r["img_path"]), color=ACC, fontsize=9,
                                  fontfamily="monospace")

        self.ax_spec.plot(lam, r["spectrum"], color="#ccc", lw=1)
        self.ax_spec.set_xlabel("Wavelength (nm)", color=TEXT, fontsize=8)
        self.ax_spec.set_ylabel("I(\u03bb)", color=TEXT, fontsize=8)
        for w in r["windows"]:
            c = colors.get(w.label, GREEN)
            l_nm, r_nm = self._window_bounds_nm(w)
            self.ax_spec.axvspan(l_nm, r_nm, color=c, alpha=0.18)
            self.ax_spec.axvline(w.center_nm, color=c, ls="--", lw=1)
            self.ax_streak.axvspan(l_nm, r_nm, color=c, alpha=0.06)

        self.draw_idle()

    def _window_bounds_nm(self, w):
        lam = self.result["lambda_axis"]
        if w.label in self._pending_window:
            return self._pending_window[w.label]
        return lam[w.left_idx], lam[w.right_idx]

    # ── click-drag to redefine the active peak's window on the spectrum axis ──
    def _on_press(self, event):
        if event.inaxes is not self.ax_spec or self.result is None:
            return
        self._drag_start = event.xdata

    def _on_motion(self, event):
        if self._drag_start is None or event.inaxes is not self.ax_spec:
            return
        lo, hi = sorted([self._drag_start, event.xdata])
        self._pending_window[self.active_peak] = (lo, hi)
        self._render()

    def _on_release(self, event):
        if self._drag_start is None:
            return
        if event.inaxes is self.ax_spec and event.xdata is not None:
            lo, hi = sorted([self._drag_start, event.xdata])
            if hi - lo > 1.0:  # ignore accidental clicks (<1nm drag)
                self._pending_window[self.active_peak] = (lo, hi)
                if self.window_edited_cb:
                    self.window_edited_cb(self.active_peak, lo, hi)
        self._drag_start = None


# ─────────────────────────────────────────────────────────────────────────────
# Setup panel
# ─────────────────────────────────────────────────────────────────────────────

def _combo(items):
    from PyQt5.QtWidgets import QComboBox
    c = QComboBox()
    c.addItems(items)
    c.setStyleSheet("""
        QComboBox { background:#252525; color:#fff; border:1px solid #555;
            padding:3px 6px; font-size:11px; font-family:monospace; border-radius:3px; }
        QComboBox QAbstractItemView { background:#252525; color:#fff; selection-background-color:#e87040; }
    """)
    return c


QUANTITY_OPTIONS = {
    "Summary: mean \u03bb vs parameter": ("summary", None, None),
    "peak1 \u03bb(t)": ("trace", "peak1", "lambda_nm"),
    "peak2 \u03bb(t)": ("trace", "peak2", "lambda_nm"),
    "peak1 intensity(t)": ("trace", "peak1", "intensity"),
    "peak2 intensity(t)": ("trace", "peak2", "intensity"),
}


class ComparisonPanel(QWidget):
    """
    Cross-shot comparison: pick any shotbook column to compare against,
    and which quantity to plot. This is the primary view once auto peak
    detection is trusted — no per-shot interaction needed.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{MID}; border-radius:4px;")
        lo = QVBoxLayout(self); lo.setContentsMargins(8, 6, 8, 6)
        lo.addWidget(_lbl("SCAN COMPARISON", GREEN, 12, bold=True))
        lo.addWidget(_hsep())

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(_lbl("Compare against:", size=10))
        self.param_combo = _combo([])
        self.param_combo.currentTextChanged.connect(self._redraw)
        ctrl_row.addWidget(self.param_combo, stretch=1)
        ctrl_row.addWidget(_lbl("Quantity:", size=10))
        self.quantity_combo = _combo(list(QUANTITY_OPTIONS.keys()))
        self.quantity_combo.currentTextChanged.connect(self._redraw)
        ctrl_row.addWidget(self.quantity_combo, stretch=1)
        lo.addLayout(ctrl_row)

        self.fig = Figure(facecolor=DARK)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_axes([0.10, 0.12, 0.82, 0.80])
        self._style_ax()
        lo.addWidget(self.canvas, stretch=1)

        self.info_lbl = _lbl("Run a batch to populate comparison plots.", DIM, 9)
        lo.addWidget(self.info_lbl)

        self._results = {}
        self._shotbook_path = None
        self._id_col = None
        self._shotbook_columns = []

    def _style_ax(self):
        self.ax.set_facecolor(DARK)
        for sp in self.ax.spines.values():
            sp.set_edgecolor(DIM)
        self.ax.tick_params(colors=TEXT, labelsize=8)
        self.fig.patch.set_facecolor(DARK)

    def set_data(self, results, shotbook_path, id_col, default_param_col):
        self._results = results
        self._shotbook_path = shotbook_path
        self._id_col = id_col
        try:
            df = pd.read_excel(shotbook_path, sheet_name=0)
            self._shotbook_columns = [c.strip() for c in df.columns if c != id_col]
        except Exception as e:
            self.info_lbl.setText(f"Could not re-read shotbook columns: {e}")
            self._shotbook_columns = [default_param_col]

        self.param_combo.blockSignals(True)
        self.param_combo.clear()
        self.param_combo.addItems(self._shotbook_columns)
        if default_param_col in self._shotbook_columns:
            self.param_combo.setCurrentText(default_param_col)
        self.param_combo.blockSignals(False)
        self._redraw()

    def _redraw(self):
        if not self._results or not self.param_combo.count():
            return
        param_col = self.param_combo.currentText()
        try:
            lookup = ssc.load_shotbook(self._shotbook_path, self._id_col, [param_col])
            param_values = {sid: lookup[sid][param_col] for sid in self._results if sid in lookup}
        except Exception as e:
            self.info_lbl.setText(f"Could not load column '{param_col}': {e}")
            return

        self.ax.cla(); self._style_ax()
        mode, peak_label, quantity = QUANTITY_OPTIONS[self.quantity_combo.currentText()]
        if mode == "summary":
            ssc.plot_summary_vs_param(self._results, param_values, param_col, ax=self.ax)
        else:
            ssc.plot_time_traces_by_param(self._results, param_values, param_col,
                                           peak_label=peak_label, quantity=quantity, ax=self.ax)
        for txt in [self.ax.title, self.ax.xaxis.label, self.ax.yaxis.label]:
            txt.set_color(TEXT)
        leg = self.ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(MID2)
            for t in leg.get_texts():
                t.set_color(TEXT)
        self.canvas.draw_idle()
        self.info_lbl.setText(f"{len(param_values)} / {len(self._results)} shots matched to '{param_col}'.")


class SetupPanel(QWidget):
    def __init__(self, on_run, parent=None):
        super().__init__(parent)
        self.on_run = on_run
        self.setStyleSheet(f"background:{MID}; border-radius:4px;")
        lo = QVBoxLayout(self); lo.setContentsMargins(10, 10, 10, 10); lo.setSpacing(8)

        lo.addWidget(_lbl("SETUP", ACC, 14, bold=True))
        lo.addWidget(_hsep())

        flo = QHBoxLayout()
        self.folder_lbl = _lbl("No scan folder selected", TEXT, 10); self.folder_lbl.setWordWrap(True)
        btn_f = _btn("Select scan folder", w=150); btn_f.clicked.connect(self._pick_folder)
        flo.addWidget(self.folder_lbl, stretch=1); flo.addWidget(btn_f)
        lo.addLayout(flo)

        clo = QHBoxLayout()
        self.csv_lbl = _lbl("No shotbook selected", TEXT, 10); self.csv_lbl.setWordWrap(True)
        btn_c = _btn("Select shotbook (.xlsx)", w=150); btn_c.clicked.connect(self._pick_csv)
        clo.addWidget(self.csv_lbl, stretch=1); clo.addWidget(btn_c)
        lo.addLayout(clo)

        lo.addWidget(_hsep())
        lo.addWidget(_lbl("Shotbook shot-id column:", size=10))
        self.id_col = _line_edit("Shot n\u00b0"); lo.addWidget(self.id_col)
        lo.addWidget(_lbl("Shotbook scan-parameter column:", size=10))
        self.param_col = _line_edit(""); lo.addWidget(self.param_col)

        lo.addWidget(_hsep())
        lo.addWidget(_lbl("OPTIONAL: absolute-scale + time alignment", GREEN, 10, bold=True))
        lo.addWidget(_lbl("Leave blank to skip any of these.", DIM, 9))
        lo.addWidget(_lbl("Gain column (linear multiplication factor):", size=9))
        self.gain_col = _line_edit(""); lo.addWidget(self.gain_col)
        lo.addWidget(_lbl("ND filter column (optical density):", size=9))
        self.nd_col = _line_edit(""); lo.addWidget(self.nd_col)
        lo.addWidget(_lbl("Delay column (ns) \u2014 aligns time axes across shots:", size=9))
        self.delay_col = _line_edit(""); lo.addWidget(self.delay_col)

        lo.addWidget(_hsep())
        lo.addWidget(_lbl("ALGORITHM PARAMETERS", GREEN, 11, bold=True))

        grid = QVBoxLayout()
        def row(label, widget):
            r = QHBoxLayout(); r.addWidget(_lbl(label, size=10), stretch=1); r.addWidget(widget)
            grid.addLayout(r)

        self.bg_pct = _spin(0, 50, 10.0, 1, 1.0); row("Background pct:", self.bg_pct)
        self.smooth_px = _spin(1, 51, 5, 0, 1); row("Smoothing (px):", self.smooth_px)
        self.prom_frac = _spin(0.01, 1.0, 0.05, 2, 0.01); row("Prominence frac:", self.prom_frac)
        self.min_sep_nm = _spin(1, 200, 10.0, 1, 1.0); row("Min peak sep (nm):", self.min_sep_nm)
        self.snr_thresh = _spin(0.5, 20, 3.0, 1, 0.5); row("SNR threshold:", self.snr_thresh)
        lo.addLayout(grid)

        lo.addStretch()
        self.btn_run = _btn("Run batch \u2192", GREEN, 140)
        self.btn_run.clicked.connect(self._run)
        lo.addWidget(self.btn_run, alignment=Qt.AlignRight)

        self._folder = ""; self._csv = ""

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select scan folder")
        if d: self._folder = d; self.folder_lbl.setText(d)

    def _pick_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select shotbook", "", "Excel (*.xlsx *.xls)")
        if p: self._csv = p; self.csv_lbl.setText(os.path.basename(p))

    def _run(self):
        if not self._folder or not self._csv:
            QMessageBox.warning(self, "Missing", "Select both a scan folder and a shotbook."); return
        if not self.param_col.text().strip():
            QMessageBox.warning(self, "Missing", "Enter the shotbook column to use as the scan parameter."); return
        kwargs = dict(bg_pct=self.bg_pct.value(), smooth_px=int(self.smooth_px.value()),
                      prominence_frac=self.prom_frac.value(), min_separation_nm=self.min_sep_nm.value(),
                      snr_thresh=self.snr_thresh.value())
        extra_cols = dict(gain_col=self.gain_col.text().strip() or None,
                           nd_col=self.nd_col.text().strip() or None,
                           delay_col=self.delay_col.text().strip() or None)
        self.on_run(self._folder, self._csv, self.id_col.text(), self.param_col.text(), extra_cols, kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Streak Spectral Scan Analyzer")
        self.resize(1500, 900)
        self.setStyleSheet(f"background:{DARK}; color:#ccc;")

        self._results = {}       # shot_id -> result dict
        self._shots = {}         # shot_id -> img_path
        self._params = {}        # shot_id -> scan param value
        self._kwargs = {}
        self._out_dir = ""
        self._current_shot = None

        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(6, 6, 6, 6); root.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background: #2a2a2a; width: 3px; }")

        self.setup_panel = SetupPanel(self._on_run_batch)
        self.setup_panel.setFixedWidth(300)
        splitter.addWidget(self.setup_panel)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{ border: 1px solid {DIM}; background:{MID}; }}
            QTabBar::tab {{ background:#222; color:{TEXT}; padding:6px 14px;
                font-family:monospace; font-size:11px; }}
            QTabBar::tab:selected {{ background:{ACC}; color:#000; }}
        """)

        # ── Tab 1 (primary): scan comparison ──
        self.comparison_panel = ComparisonPanel()
        tabs.addTab(self.comparison_panel, "Scan comparison")

        # ── Tab 2 (secondary): per-shot review, for the rare shot flagged NEEDS REVIEW ──
        review_tab = QWidget()
        rtlo = QHBoxLayout(review_tab); rtlo.setContentsMargins(0, 0, 0, 0); rtlo.setSpacing(6)

        list_panel = QWidget(); list_panel.setStyleSheet(f"background:{MID}; border-radius:4px;")
        llo = QVBoxLayout(list_panel); llo.setContentsMargins(8, 6, 8, 6)
        llo.addWidget(_lbl("SHOTS", GREEN, 12, bold=True))
        llo.addWidget(_hsep())
        self.shot_list = QListWidget()
        self.shot_list.setStyleSheet(f"""
            QListWidget {{ background:{MID2}; color:{TEXT}; border:1px solid {DIM};
                font-size:10px; font-family:monospace; }}
            QListWidget::item:selected {{ background:{ACC}; color:#000; }}
        """)
        self.shot_list.currentItemChanged.connect(self._on_shot_selected)
        llo.addWidget(self.shot_list, stretch=1)
        list_panel.setFixedWidth(220)
        rtlo.addWidget(list_panel)

        review_panel = QWidget(); review_panel.setStyleSheet(f"background:{MID}; border-radius:4px;")
        rlo = QVBoxLayout(review_panel); rlo.setContentsMargins(8, 6, 8, 6)
        rlo.addWidget(_lbl("REVIEW (only needed if auto-detection flagged a shot)", ACC, 11, bold=True))
        rlo.addWidget(_hsep())

        self.status_lbl = _lbl("Run a batch to begin.", TEXT, 10); self.status_lbl.setWordWrap(True)
        rlo.addWidget(self.status_lbl)

        self.canvas = SpectralCanvas()
        self.canvas.window_edited_cb = self._on_window_edited
        rlo.addWidget(self.canvas, stretch=1)

        edit_row = QHBoxLayout()
        edit_row.addWidget(_lbl("Editing window:", size=10))
        self.peak_group = QButtonGroup(self)
        self.radio_p1 = QRadioButton("peak1 (bluer)"); self.radio_p2 = QRadioButton("peak2 (redder)")
        self.radio_p1.setChecked(True)
        for r in (self.radio_p1, self.radio_p2):
            r.setStyleSheet(f"color:{TEXT}; font-size:10px; font-family:monospace;")
            self.peak_group.addButton(r)
        self.radio_p1.toggled.connect(lambda on: on and self.canvas.set_active_peak("peak1"))
        self.radio_p2.toggled.connect(lambda on: on and self.canvas.set_active_peak("peak2"))
        edit_row.addWidget(self.radio_p1); edit_row.addWidget(self.radio_p2); edit_row.addStretch()
        rlo.addLayout(edit_row)
        rlo.addWidget(_lbl("Click-drag on the spectrum panel to set the active window.", DIM, 9))

        nav = QHBoxLayout()
        self.btn_rerun = _btn("Re-run this shot", GREEN, 140)
        self.btn_rerun.clicked.connect(self._rerun_current_shot)
        nav.addWidget(self.btn_rerun); nav.addStretch()
        rlo.addLayout(nav)

        rtlo.addWidget(review_panel, stretch=1)
        tabs.addTab(review_tab, "Per-shot review")

        splitter.addWidget(tabs)
        splitter.setSizes([300, 1200])
        root.addWidget(splitter, stretch=1)

        self._manual_windows = {}   # shot_id -> list[PeakWindow] override

    # ── batch run ──────────────────────────────────────────────────────────
    def _on_run_batch(self, folder, csv_path, id_col, param_col, extra_cols, kwargs):
        self._kwargs = kwargs
        self._id_col = id_col
        self._shotbook_path = csv_path
        self._out_dir = os.path.join(folder, "_spectral_scan_out")
        gain_col, nd_col, delay_col = extra_cols["gain_col"], extra_cols["nd_col"], extra_cols["delay_col"]
        needed_cols = [param_col] + [c for c in (gain_col, nd_col, delay_col) if c]
        try:
            self._shots, layout_warnings = ssc.find_shot_images(folder)
            shotbook = ssc.load_shotbook(csv_path, id_col, needed_cols)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e)); return

        if not self._shots:
            QMessageBox.warning(self, "No .img files found",
                f"Checked {folder} for both layouts:\n"
                f"  \u2022 <shot_id>/<shot_id>.img (subfolder per shot)\n"
                f"  \u2022 <shot_id>.img (flat folder)\n"
                f"and found nothing. Check the folder path.")
            self.status_lbl.setText("No .img files found — see dialog.")
            return

        matched_ids = [sid for sid in self._shots if sid in shotbook]
        if not matched_ids:
            shotbook_ids = list(pd.read_excel(csv_path, sheet_name=0)[id_col].dropna().astype(str))[:6]
            QMessageBox.warning(self, "No shots matched the shotbook",
                f"Found {len(self._shots)} .img shot(s) on disk, but none matched "
                f"shotbook column '{id_col}'.\n\n"
                f"Example shot_id from disk: {list(self._shots)[:6]}\n"
                f"Example '{id_col}' values from shotbook: {shotbook_ids}\n\n"
                f"These need to share a common integer (e.g. folder 'Shot19' <-> "
                f"shotbook value 19) or match exactly as strings.")
            self.status_lbl.setText(f"{len(self._shots)} shot(s) on disk, 0 matched shotbook — see dialog.")
            return

        if layout_warnings:
            self.status_lbl.setText(" | ".join(layout_warnings))

        os.makedirs(self._out_dir, exist_ok=True)
        self.shot_list.clear()
        self._results = {}
        self._params = {}
        needs_review_ids, ok_ids = [], []

        def _clean(v, default):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return default
            return float(v)

        for shot_id, img_path in self._shots.items():
            if shot_id not in shotbook:
                continue
            rec = shotbook[shot_id]
            gain = _clean(rec.get(gain_col), 1.0) if gain_col else 1.0
            nd_od = _clean(rec.get(nd_col), 0.0) if nd_col else 0.0
            delay = _clean(rec.get(delay_col), 0.0) if delay_col else 0.0
            result = ssc.analyze_shot(img_path, gain=gain, nd_od=nd_od, delay=delay, **self._kwargs)
            self._results[shot_id] = result
            self._params[shot_id] = rec[param_col]
            self._save_shot(shot_id, result)
            (needs_review_ids if result["needs_review"] else ok_ids).append(shot_id)

        for shot_id in needs_review_ids + ok_ids:
            item = QListWidgetItem(("\u26a0 " if shot_id in needs_review_ids else "\u2713 ") + shot_id)
            item.setForeground(QColor(RED if shot_id in needs_review_ids else GREEN))
            item.setData(Qt.UserRole, shot_id)
            self.shot_list.addItem(item)

        self.status_lbl.setText(
            f"{len(self._results)} shots analyzed \u2014 {len(needs_review_ids)} need review, "
            f"{len(ok_ids)} OK. Output: {self._out_dir}")
        if self.shot_list.count():
            self.shot_list.setCurrentRow(0)

        self.comparison_panel.set_data(self._results, csv_path, id_col, param_col)

    def _save_shot(self, shot_id, result):
        shot_out = os.path.join(self._out_dir, shot_id)
        os.makedirs(shot_out, exist_ok=True)
        ssc.save_shot_csv(result, os.path.join(shot_out, f"{shot_id}_peaks.csv"))
        ssc.save_shot_plots(result, os.path.join(shot_out, shot_id))
        self._write_aggregate()

    def _write_aggregate(self):
        import pandas as pd
        long_rows, summary_rows = [], []
        param_col = self.setup_panel.param_col.text()
        for shot_id, result in self._results.items():
            param_val = self._params.get(shot_id)
            t = result["time_axis"]
            for i in range(len(t)):
                row = {"shot_id": shot_id, param_col: param_val, "time_ns": t[i]}
                for label in ("peak1", "peak2"):
                    tr = result["traces"].get(label)
                    row[f"{label}_lambda_nm"] = tr["lambda_nm"][i] if tr else np.nan
                    row[f"{label}_intensity"] = tr["intensity"][i] if tr else np.nan
                long_rows.append(row)
            summary = {"shot_id": shot_id, param_col: param_val,
                       "needs_review": result["needs_review"],
                       "n_peaks_found": len(result["windows"]),
                       "n_artifact_rows": result.get("n_artifact_rows", 0)}
            for label in ("peak1", "peak2"):
                tr = result["traces"].get(label)
                if tr is not None and tr["valid"].any():
                    summary[f"{label}_lambda_mean_nm"] = float(np.nanmean(tr["lambda_nm"]))
                    summary[f"{label}_lambda_std_nm"] = float(np.nanstd(tr["lambda_nm"]))
            summary_rows.append(summary)
        pd.DataFrame(long_rows).to_csv(os.path.join(self._out_dir, "scan_timeresolved.csv"), index=False)
        pd.DataFrame(summary_rows).to_csv(os.path.join(self._out_dir, "scan_summary.csv"), index=False)

    # ── shot selection / review ──────────────────────────────────────────────
    def _on_shot_selected(self, item):
        if item is None: return
        shot_id = item.data(Qt.UserRole)
        self._current_shot = shot_id
        self.canvas.load_result(self._results[shot_id])

    def _on_window_edited(self, label, lo_nm, hi_nm):
        # store pending manual bounds; applied on "Re-run this shot"
        pass

    def _rerun_current_shot(self):
        shot_id = self._current_shot
        if shot_id is None: return
        result = self._results[shot_id]
        lam = result["lambda_axis"]

        manual_windows = []
        pending = self.canvas._pending_window
        for w in result["windows"]:
            if w.label in pending:
                lo_nm, hi_nm = pending[w.label]
                left_idx = int(np.searchsorted(lam, lo_nm))
                right_idx = int(np.searchsorted(lam, hi_nm))
                center_idx = (left_idx + right_idx) // 2
                manual_windows.append(ssc.PeakWindow(
                    label=w.label, center_idx=center_idx, center_nm=float(lam[center_idx]),
                    left_idx=left_idx, right_idx=right_idx, prominence=w.prominence))
            else:
                manual_windows.append(w)
        manual_windows.sort(key=lambda w: w.center_nm)

        img_path = self._shots[shot_id]
        new_result = ssc.analyze_shot(img_path, manual_windows=manual_windows,
                                       gain=result.get("gain", 1.0), nd_od=result.get("nd_od", 0.0),
                                       delay=result.get("delay", 0.0), **self._kwargs)
        self._results[shot_id] = new_result
        self._save_shot(shot_id, new_result)
        self.canvas.load_result(new_result)
        self.comparison_panel.set_data(self._results, self._shotbook_path, self._id_col,
                                        self.comparison_panel.param_combo.currentText() or
                                        self.setup_panel.param_col.text())

        # update list item color/status
        for i in range(self.shot_list.count()):
            item = self.shot_list.item(i)
            if item.data(Qt.UserRole) == shot_id:
                item.setText(("\u26a0 " if new_result["needs_review"] else "\u2713 ") + shot_id)
                item.setForeground(QColor(RED if new_result["needs_review"] else GREEN))
        self.status_lbl.setText(f"{shot_id}: re-run with manual window(s). Saved to {self._out_dir}.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
