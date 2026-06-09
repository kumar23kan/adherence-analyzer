#!/usr/bin/env python3
"""
Adherence Analyzer
==================
Tracking-based cell adherence counter for phase-contrast timelapse microscopy.

Definition: a cell detected at the same XY position (< max_dist px) for
>= min_frames consecutive frames is classified as adhered.

Usage:
    python3 adherence_analyzer.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import glob
import os
import re

import numpy as np
from PIL import Image
from skimage import filters, feature
from scipy.spatial import cKDTree
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    'bg_sigma':    ('Background σ (px)',       80,   10, 200, 1),
    'bright_thr':  ('Bright blob threshold σ', 1.5, 0.3,  5.0, 0.1),
    'dark_thr':    ('Dark blob threshold σ',   0.9, 0.2,  3.0, 0.1),
    'min_sigma':   ('Min cell radius (px)',     4,   1,   20, 1),
    'max_sigma':   ('Max cell radius (px)',    18,   5,   60, 1),
    'max_dist':    ('Max track distance (px)',  8,   1,   30, 1),
    'min_frames':  ('Min frames = adhered',     3,   1,   10, 1),
}

ADHERED_COLOR  = '#22cc44'   # green
FLOATING_COLOR = '#ff5555'   # red
TRACK_COLOR    = '#ffdd00'   # yellow


# ── Core analysis functions ───────────────────────────────────────────────────

def load_files(folder):
    exts = ['*.tiff', '*.tif', '*.TIFF', '*.TIF']
    files = []
    for e in exts:
        files += glob.glob(os.path.join(folder, e))
    def sort_key(p):
        m = re.search(r'(\d+)\.\w+$', p)
        return int(m.group(1)) if m else 0
    return sorted(set(files), key=sort_key)


def background_correct(path, bg_sigma):
    arr  = np.array(Image.open(path)).astype(np.float32)
    bg   = filters.gaussian(arr, sigma=bg_sigma)
    corr = arr - bg
    std  = corr.std()
    return arr, corr / (std if std > 0 else 1.0)


def detect_spots(corr, bright_thr, dark_thr, min_sigma, max_sigma):
    """Return Nx2 array (row, col) of all detected cell centres."""
    kw = dict(min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=6)
    bright = feature.blob_log( corr, threshold=bright_thr, **kw)[:, :2]
    dark   = feature.blob_log(-corr, threshold=dark_thr,   **kw)[:, :2]

    if len(bright) == 0 and len(dark) == 0:
        return np.empty((0, 2), dtype=np.float32)
    elif len(bright) == 0:
        pts = dark
    elif len(dark) == 0:
        pts = bright
    else:
        pts = np.vstack([bright, dark])

    # De-duplicate: keep one point per 5-px neighbourhood
    pts = pts.astype(np.float32)
    tree = cKDTree(pts)
    used = np.zeros(len(pts), bool)
    keep = []
    for i, p in enumerate(pts):
        if used[i]:
            continue
        for j in tree.query_ball_point(p, r=5):
            used[j] = True
        keep.append(p)
    return np.array(keep, dtype=np.float32) if keep else np.empty((0, 2), dtype=np.float32)


def detect_floating_count(corr, bright_thr, min_sigma, max_sigma):
    b = feature.blob_log(corr, min_sigma=min_sigma, max_sigma=max_sigma,
                         num_sigma=6, threshold=bright_thr)
    return len(b)


def run_tracker(files, params, progress_cb=None, cancel_flag=None):
    """
    Run tracking over all frames.

    Returns
    -------
    results : list of dicts  (one per frame)
    track_history : list of lists  – track_history[fi] = list of (r,c,consec,track_id)
    """
    bg_sigma   = params['bg_sigma']
    bright_thr = params['bright_thr']
    dark_thr   = params['dark_thr']
    min_sigma  = int(params['min_sigma'])
    max_sigma  = int(params['max_sigma'])
    max_dist   = params['max_dist']
    min_frames = int(params['min_frames'])

    active_tracks = {}   # id -> {'pos': (r,c), 'consec': int}
    next_id       = 0
    results       = []
    track_history = []   # per-frame snapshot

    for fi, fpath in enumerate(files):
        if cancel_flag and cancel_flag():
            break

        _, corr  = background_correct(fpath, bg_sigma)
        spots    = detect_spots(corr, bright_thr, dark_thr, min_sigma, max_sigma)
        n_float  = detect_floating_count(corr, bright_thr, min_sigma, max_sigma)
        frame_no = fi + 1

        if len(spots) == 0:
            results.append({'frame': frame_no, 'adhered': 0,
                            'floating': n_float, 'total_detected': 0})
            track_history.append([])
            active_tracks = {}
            if progress_cb:
                progress_cb(fi + 1, len(files))
            continue

        if fi == 0:
            for s in spots:
                active_tracks[next_id] = {'pos': s, 'consec': 1}
                next_id += 1
            n_adhered = 0
        else:
            if len(active_tracks) == 0:
                for s in spots:
                    active_tracks[next_id] = {'pos': s, 'consec': 1}
                    next_id += 1
                n_adhered = 0
            else:
                tids     = list(active_tracks.keys())
                prev_pos = np.array([active_tracks[t]['pos'] for t in tids])
                tree     = cKDTree(prev_pos)
                dists, idxs = tree.query(spots, distance_upper_bound=max_dist + 1e-6)

                matched_prev  = {}
                for si in np.argsort(dists):
                    d, pi = dists[si], idxs[si]
                    if d > max_dist:
                        break
                    if pi not in matched_prev:
                        matched_prev[pi] = si

                matched_spots = set(matched_prev.values())
                new_active = {}
                for pi, si in matched_prev.items():
                    tid = tids[pi]
                    new_active[tid] = {'pos': spots[si],
                                       'consec': active_tracks[tid]['consec'] + 1}
                for si, s in enumerate(spots):
                    if si not in matched_spots:
                        new_active[next_id] = {'pos': s, 'consec': 1}
                        next_id += 1
                active_tracks = new_active

            n_adhered = sum(1 for t in active_tracks.values()
                            if t['consec'] >= min_frames)

        # Snapshot for visualisation
        snap = [(t['pos'][0], t['pos'][1], t['consec'], tid)
                for tid, t in active_tracks.items()]
        track_history.append(snap)

        results.append({'frame': frame_no, 'adhered': n_adhered,
                        'floating': n_float, 'total_detected': len(spots)})

        if progress_cb:
            progress_cb(fi + 1, len(files))

    return results, track_history


# ── GUI ───────────────────────────────────────────────────────────────────────

class AdherenceAnalyzer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('Adherence Analyzer — Phase Contrast Timelapse')
        self.configure(bg='#1e1e2e')
        self.resizable(True, True)

        # State
        self.image_files  = []
        self.current_fi   = 0
        self.results      = []
        self.track_history = []
        self._cancel_flag = False
        self._running     = False

        # Parameter variables
        self.pvars = {k: tk.DoubleVar(value=DEFAULTS[k][1]) for k in DEFAULTS}

        self._build_ui()
        self.geometry('1500x880')

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TFrame',       background='#1e1e2e')
        style.configure('TLabel',       background='#1e1e2e', foreground='#cdd6f4')
        style.configure('TLabelframe',  background='#1e1e2e', foreground='#89b4fa')
        style.configure('TLabelframe.Label', background='#1e1e2e', foreground='#89b4fa',
                        font=('Helvetica', 10, 'bold'))
        style.configure('Accent.TButton', background='#89b4fa', foreground='#1e1e2e',
                        font=('Helvetica', 10, 'bold'))
        style.configure('TButton',      background='#313244', foreground='#cdd6f4')
        style.configure('TProgressbar', troughcolor='#313244', background='#89b4fa')
        style.configure('TScale',       background='#1e1e2e', troughcolor='#313244')
        style.configure('TEntry',       fieldbackground='#313244', foreground='#cdd6f4')

        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self, bg='#181825', pady=6)
        top.pack(fill='x')
        tk.Label(top, text='Adherence Analyzer', bg='#181825', fg='#89b4fa',
                 font=('Helvetica', 15, 'bold')).pack(side='left', padx=14)
        self.lbl_folder = tk.Label(top, text='No folder loaded', bg='#181825',
                                   fg='#6c7086', font=('Helvetica', 9))
        self.lbl_folder.pack(side='left', padx=8)
        ttk.Button(top, text='Export Report', command=self._export,
                   style='TButton').pack(side='right', padx=8)
        self.btn_run = ttk.Button(top, text='▶  Run Analysis', command=self._start_analysis,
                                  style='Accent.TButton')
        self.btn_run.pack(side='right', padx=4)
        ttk.Button(top, text='📂  Load Folder', command=self._load_folder,
                   style='TButton').pack(side='right', padx=4)

        # ── Main area ─────────────────────────────────────────────────────────
        main = tk.Frame(self, bg='#1e1e2e')
        main.pack(fill='both', expand=True, padx=6, pady=(0, 4))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: Parameters
        self._build_params(main)

        # Middle: Preview
        self._build_preview(main)

        # Right: Results
        self._build_results(main)

        # ── Bottom status bar ─────────────────────────────────────────────────
        bot = tk.Frame(self, bg='#181825', pady=4)
        bot.pack(fill='x', side='bottom')
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(bot, variable=self.progress_var, maximum=100,
                        length=400).pack(side='left', padx=10)
        self.lbl_status = tk.Label(bot, text='Ready — load a folder to begin',
                                   bg='#181825', fg='#6c7086',
                                   font=('Helvetica', 9))
        self.lbl_status.pack(side='left', padx=6)

    # ── Parameters panel ─────────────────────────────────────────────────────

    def _build_params(self, parent):
        frame = tk.Frame(parent, bg='#1e1e2e', width=270)
        frame.grid(row=0, column=0, sticky='ns', padx=(0, 6), pady=4)
        frame.pack_propagate(False)

        # Detection group
        det = ttk.LabelFrame(frame, text='Detection', padding=8)
        det.pack(fill='x', pady=(4, 4))
        det_keys = ['bg_sigma', 'bright_thr', 'dark_thr', 'min_sigma', 'max_sigma']
        self._make_param_rows(det, det_keys)

        # Tracking group
        trk = ttk.LabelFrame(frame, text='Tracking', padding=8)
        trk.pack(fill='x', pady=(0, 4))
        trk_keys = ['max_dist', 'min_frames']
        self._make_param_rows(trk, trk_keys)

        # Legend
        leg = ttk.LabelFrame(frame, text='Legend', padding=8)
        leg.pack(fill='x', pady=(0, 4))
        for color, label in [(ADHERED_COLOR, 'Adhered (≥ min_frames)'),
                             (FLOATING_COLOR, 'Floating (bright round)'),
                             (TRACK_COLOR,    'Recent track path')]:
            row = tk.Frame(leg, bg='#1e1e2e')
            row.pack(anchor='w', pady=1)
            tk.Canvas(row, width=14, height=14, bg=color,
                      highlightthickness=0).pack(side='left', padx=(0, 6))
            tk.Label(row, text=label, bg='#1e1e2e', fg='#cdd6f4',
                     font=('Helvetica', 8)).pack(side='left')

        # Stats box (populated after run)
        self.stats_frame = ttk.LabelFrame(frame, text='Statistics', padding=8)
        self.stats_frame.pack(fill='x', pady=(0, 4))
        self.lbl_stats = tk.Label(self.stats_frame, text='— run analysis first —',
                                  bg='#1e1e2e', fg='#6c7086',
                                  font=('Helvetica', 8), justify='left')
        self.lbl_stats.pack(anchor='w')

    def _make_param_rows(self, parent, keys):
        for k in keys:
            label, default, lo, hi, step = DEFAULTS[k]
            row = tk.Frame(parent, bg='#1e1e2e')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, bg='#1e1e2e', fg='#cdd6f4',
                     font=('Helvetica', 8), width=22, anchor='w').pack(side='left')
            var = self.pvars[k]
            entry = ttk.Entry(row, textvariable=var, width=6)
            entry.pack(side='right')
            scale = ttk.Scale(row, from_=lo, to=hi, variable=var,
                              orient='horizontal', length=90,
                              command=lambda v, e=entry, kk=k: self._on_param_change(kk))
            scale.pack(side='right', padx=4)

    def _on_param_change(self, key):
        # Debounce: only re-preview if files loaded
        if self.image_files:
            self.after_cancel(getattr(self, '_debounce_id', None) or 0)
            self._debounce_id = self.after(600, self._update_preview)

    # ── Preview panel ─────────────────────────────────────────────────────────

    def _build_preview(self, parent):
        frame = tk.Frame(parent, bg='#1e1e2e')
        frame.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # Matplotlib figure
        self.fig_prev = Figure(figsize=(6, 5.5), facecolor='#1e1e2e')
        self.ax_prev  = self.fig_prev.add_subplot(111)
        self.ax_prev.set_facecolor('#181825')
        self.ax_prev.axis('off')
        self.ax_prev.text(0.5, 0.5, 'Load a folder to begin',
                          ha='center', va='center', color='#6c7086', fontsize=12,
                          transform=self.ax_prev.transAxes)
        self.canvas_prev = FigureCanvasTkAgg(self.fig_prev, master=frame)
        self.canvas_prev.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # Frame navigation bar
        nav = tk.Frame(frame, bg='#181825')
        nav.grid(row=1, column=0, sticky='ew', pady=(2, 0))

        self.btn_prev_frame = ttk.Button(nav, text='◄', width=3,
                                         command=lambda: self._nav_frame(-1))
        self.btn_prev_frame.pack(side='left', padx=4)

        self.frame_slider = ttk.Scale(nav, from_=0, to=0, orient='horizontal',
                                      command=self._on_slider)
        self.frame_slider.pack(side='left', fill='x', expand=True, padx=4)

        self.btn_next_frame = ttk.Button(nav, text='►', width=3,
                                          command=lambda: self._nav_frame(1))
        self.btn_next_frame.pack(side='left', padx=4)

        self.lbl_frame_info = tk.Label(nav, text='–/–', bg='#181825', fg='#89b4fa',
                                        font=('Helvetica', 9, 'bold'), width=18)
        self.lbl_frame_info.pack(side='left', padx=8)

        # Show mode radio
        self.show_mode = tk.StringVar(value='detected')
        mode_frame = tk.Frame(nav, bg='#181825')
        mode_frame.pack(side='right', padx=8)
        for text, val in [('Raw', 'raw'), ('Detected', 'detected')]:
            tk.Radiobutton(mode_frame, text=text, variable=self.show_mode,
                           value=val, bg='#181825', fg='#cdd6f4',
                           selectcolor='#313244', activebackground='#181825',
                           command=self._update_preview).pack(side='left', padx=4)

    # ── Results panel ─────────────────────────────────────────────────────────

    def _build_results(self, parent):
        frame = tk.Frame(parent, bg='#1e1e2e', width=380)
        frame.grid(row=0, column=2, sticky='ns', padx=(6, 0), pady=4)
        frame.pack_propagate(False)

        self.fig_res = Figure(figsize=(3.8, 5.5), facecolor='#1e1e2e')
        self.ax_res  = self.fig_res.add_subplot(111)
        self._clear_results_plot()
        self.canvas_res = FigureCanvasTkAgg(self.fig_res, master=frame)
        self.canvas_res.get_tk_widget().pack(fill='both', expand=True)

    def _clear_results_plot(self):
        self.ax_res.cla()
        self.ax_res.set_facecolor('#181825')
        self.ax_res.tick_params(colors='#6c7086', labelsize=7)
        for spine in self.ax_res.spines.values():
            spine.set_edgecolor('#313244')
        self.ax_res.text(0.5, 0.5, 'Run analysis\nto see results',
                         ha='center', va='center', color='#6c7086', fontsize=10,
                         transform=self.ax_res.transAxes)
        self.fig_res.tight_layout()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _load_folder(self):
        folder = filedialog.askdirectory(title='Select folder containing TIFF images')
        if not folder:
            return
        files = load_files(folder)
        if not files:
            messagebox.showerror('No images', 'No TIFF files found in selected folder.')
            return
        self.image_files = files
        self.current_fi  = 0
        self.results     = []
        self.track_history = []
        self.frame_slider.config(to=len(files) - 1)
        short = os.path.basename(folder)
        self.lbl_folder.config(text=f'{short}  ({len(files)} frames)', fg='#a6e3a1')
        self._set_status(f'Loaded {len(files)} frames from: {folder}')
        self._update_preview()

    def _nav_frame(self, delta):
        if not self.image_files:
            return
        self.current_fi = max(0, min(len(self.image_files) - 1, self.current_fi + delta))
        self.frame_slider.set(self.current_fi)
        self._update_preview()

    def _on_slider(self, val):
        fi = int(float(val))
        if fi != self.current_fi:
            self.current_fi = fi
            self._update_preview()

    def _get_params(self):
        return {k: float(self.pvars[k].get()) for k in self.pvars}

    def _update_preview(self):
        if not self.image_files:
            return
        fi = self.current_fi
        fpath = self.image_files[fi]
        total = len(self.image_files)
        N     = total - 1 if total > 1 else 1
        t_min = fi * 60.0 / N
        self.lbl_frame_info.config(text=f'Frame {fi+1}/{total}  ({t_min:.1f} min)')

        p     = self._get_params()
        arr, corr = background_correct(fpath, p['bg_sigma'])

        self.ax_prev.cla()
        self.ax_prev.set_facecolor('#181825')
        self.ax_prev.axis('off')

        vmin = np.percentile(arr, 1)
        vmax = np.percentile(arr, 99)
        self.ax_prev.imshow(arr, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')

        if self.show_mode.get() == 'detected':
            spots = detect_spots(corr, p['bright_thr'], p['dark_thr'],
                                 int(p['min_sigma']), int(p['max_sigma']))
            n_float = detect_floating_count(corr, p['bright_thr'],
                                            int(p['min_sigma']), int(p['max_sigma']))

            # If we have track history for this frame, use it
            if fi < len(self.track_history) and self.track_history:
                snap = self.track_history[fi]
                min_f = int(p['min_frames'])
                for r, c, consec, tid in snap:
                    color  = ADHERED_COLOR if consec >= min_f else FLOATING_COLOR
                    radius = int(p['max_sigma']) * 0.8
                    circ   = plt.Circle((c, r), radius, color=color,
                                        fill=False, lw=1.2, alpha=0.85)
                    self.ax_prev.add_patch(circ)
                    if consec >= min_f:
                        self.ax_prev.text(c + radius, r, str(consec),
                                          color=ADHERED_COLOR, fontsize=4, va='center')
                n_adh = sum(1 for _, _, c, _ in snap if c >= min_f)
                info  = f'Adhered: {n_adh}   Floating: {n_float}'
            else:
                # Pre-analysis: just show raw detections
                for r, c in spots:
                    circ = plt.Circle((c, r), int(p['max_sigma']) * 0.8,
                                      color='#89b4fa', fill=False, lw=0.8, alpha=0.6)
                    self.ax_prev.add_patch(circ)
                info = f'Detected: {len(spots)}'

            self.ax_prev.text(6, 18, info, color='white',
                              fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e2e',
                                        alpha=0.7, edgecolor='none'))

        self.ax_prev.set_title(f'Frame {fi+1}  |  t = {t_min:.1f} min',
                               color='#cdd6f4', fontsize=9, pad=4)
        self.fig_prev.tight_layout(pad=0.5)
        self.canvas_prev.draw()

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _start_analysis(self):
        if not self.image_files:
            messagebox.showwarning('No images', 'Load a folder first.')
            return
        if self._running:
            self._cancel_flag = True
            self.btn_run.config(text='▶  Run Analysis')
            self._running = False
            return
        self._cancel_flag = False
        self._running     = True
        self.btn_run.config(text='⏹  Cancel')
        self.progress_var.set(0)
        self._set_status('Running analysis…')
        params = self._get_params()
        threading.Thread(target=self._analysis_thread, args=(params,),
                         daemon=True).start()

    def _analysis_thread(self, params):
        def prog(done, total):
            pct = done / total * 100
            self.after(0, self.progress_var.set, pct)
            self.after(0, self._set_status,
                       f'Processing frame {done}/{total}…')

        results, track_history = run_tracker(
            self.image_files, params,
            progress_cb=prog,
            cancel_flag=lambda: self._cancel_flag
        )
        self.after(0, self._analysis_done, results, track_history)

    def _analysis_done(self, results, track_history):
        self._running      = False
        self._cancel_flag  = False
        self.btn_run.config(text='▶  Run Analysis')
        self.results       = results
        self.track_history = track_history

        df = pd.DataFrame(results)
        N  = len(self.image_files)
        df['time_min'] = (df['frame'] - 1) * (60.0 / max(N - 1, 1))
        df['adhered_roll5']  = df['adhered'].rolling(5,  center=True, min_periods=1).mean()
        df['floating_roll5'] = df['floating'].rolling(5, center=True, min_periods=1).mean()
        self._df = df

        self._update_results_plot(df)
        self._update_stats(df)
        self._update_preview()
        self._set_status(
            f'Done — {len(results)} frames processed. '
            f'Adhered peak: {df["adhered"].max()}  |  '
            f'Floating peak: {df["floating"].max()}'
        )
        self.progress_var.set(100)

    def _update_results_plot(self, df):
        self.ax_res.cla()
        ax = self.ax_res
        ax.set_facecolor('#181825')
        ax.tick_params(colors='#a6adc8', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#313244')

        ax.plot(df['time_min'], df['adhered'],  color=ADHERED_COLOR,
                alpha=0.25, lw=0.8)
        ax.plot(df['time_min'], df['adhered_roll5'],  color=ADHERED_COLOR,
                lw=2.2, label='Adhered')
        ax.plot(df['time_min'], df['floating'], color=FLOATING_COLOR,
                alpha=0.25, lw=0.8)
        ax.plot(df['time_min'], df['floating_roll5'], color=FLOATING_COLOR,
                lw=2.0, ls='--', label='Floating')

        ax.set_xlabel('Time (min)', color='#a6adc8', fontsize=8)
        ax.set_ylabel('Cell count',  color='#a6adc8', fontsize=8)
        ax.set_title('Adherence time course', color='#cdd6f4', fontsize=9, pad=4)
        ax.legend(fontsize=7, facecolor='#313244', labelcolor='#cdd6f4',
                  edgecolor='none')
        ax.grid(True, color='#313244', alpha=0.6)
        ax.set_ylim(bottom=0)

        self.fig_res.tight_layout(pad=0.8)
        self.canvas_res.draw()

    def _update_stats(self, df):
        p1 = df[df['time_min'] <= 20]
        p2 = df[df['time_min'] >  20]
        lines = []
        if len(p1):
            lines += [f'Phase 1  (0–20 min)',
                      f'  Adhered :  {p1["adhered"].mean():.0f} ± {p1["adhered"].std():.0f}',
                      f'  Floating:  {p1["floating"].mean():.0f} ± {p1["floating"].std():.0f}']
        if len(p2):
            lines += ['', f'Phase 2  (20–60 min)',
                      f'  Adhered :  {p2["adhered"].mean():.0f} ± {p2["adhered"].std():.0f}',
                      f'  Floating:  {p2["floating"].mean():.0f} ± {p2["floating"].std():.0f}']
        self.lbl_stats.config(text='\n'.join(lines), fg='#cdd6f4')

    # ── Export ────────────────────────────────────────────────────────────────

    def _export(self):
        if not self.results:
            messagebox.showwarning('No results', 'Run analysis before exporting.')
            return
        out_dir = filedialog.askdirectory(title='Select output folder for report')
        if not out_dir:
            return

        self._set_status('Exporting report…')
        self.progress_var.set(0)
        params = self._get_params()

        threading.Thread(
            target=self._export_thread,
            args=(out_dir, params),
            daemon=True
        ).start()

    def _export_thread(self, out_dir, params):
        df         = self._df
        files      = self.image_files
        track_hist = self.track_history
        N          = len(files)
        min_f      = int(params['min_frames'])

        # 1. CSV
        csv_path = os.path.join(out_dir, 'adherence_counts.csv')
        df.to_csv(csv_path, index=False)

        # 2. Summary time course plot
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')

        ax = axes[0]
        ax.plot(df['time_min'], df['adhered'],  color=ADHERED_COLOR, alpha=0.3, lw=1)
        ax.plot(df['time_min'], df['adhered_roll5'], color=ADHERED_COLOR,
                lw=2.5, label='Adhered (5-frame mean)')
        ax.plot(df['time_min'], df['floating'], color=FLOATING_COLOR, alpha=0.3, lw=1)
        ax.plot(df['time_min'], df['floating_roll5'], color=FLOATING_COLOR,
                lw=2.0, ls='--', label='Floating (bright round)')
        ax.set_ylabel('Cell count', fontsize=11)
        ax.set_title('Adherence Time Course', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        ax2 = axes[1]
        ax2.fill_between(df['time_min'], 0, df['adhered_roll5'],
                         alpha=0.55, color=ADHERED_COLOR, label='Adhered')
        ax2.fill_between(df['time_min'], df['adhered_roll5'],
                         df['adhered_roll5'] + df['floating_roll5'],
                         alpha=0.40, color=FLOATING_COLOR, label='Floating')
        ax2.set_xlabel('Time (min)', fontsize=11)
        ax2.set_ylabel('Cell count', fontsize=11)
        ax2.set_title('Stacked: Adhered vs Floating', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'timecourse_plot.png'), dpi=150)
        plt.close(fig)

        # 3. Contact sheet — key frames (0%, 20%, 40%, 60%, 80%, 100%)
        key_indices = [int(x * (N - 1) / 4) for x in range(5)]
        fig2, axes2 = plt.subplots(2, 5, figsize=(22, 9), facecolor='#111111')
        for col, fi in enumerate(key_indices):
            fpath  = files[fi]
            t_min  = fi * 60.0 / max(N - 1, 1)
            arr, _ = background_correct(fpath, params['bg_sigma'])
            vmin   = np.percentile(arr, 1)
            vmax   = np.percentile(arr, 99)

            for row in range(2):
                ax_k = axes2[row, col]
                ax_k.imshow(arr, cmap='gray', vmin=vmin, vmax=vmax)
                ax_k.axis('off')
                if row == 0:
                    ax_k.set_title(f'Frame {fi+1}\nt = {t_min:.1f} min',
                                   color='white', fontsize=9)
                else:
                    if fi < len(track_hist):
                        snap = track_hist[fi]
                        for r, c, consec, _ in snap:
                            color  = ADHERED_COLOR if consec >= min_f else FLOATING_COLOR
                            radius = params['max_sigma'] * 0.85
                            ax_k.add_patch(plt.Circle((c, r), radius,
                                                       color=color, fill=False,
                                                       lw=1.0, alpha=0.85))
                        n_adh  = sum(1 for _, _, cs, _ in snap if cs >= min_f)
                        n_flt  = df[df['frame'] == fi + 1]['floating'].values
                        n_flt  = int(n_flt[0]) if len(n_flt) else 0
                        ax_k.text(10, 30,
                                  f'Adhered: {n_adh}\nFloating: {n_flt}',
                                  color='white', fontsize=8,
                                  bbox=dict(facecolor='#000000', alpha=0.55,
                                            edgecolor='none', pad=3))

        axes2[0, 0].set_ylabel('Raw', color='white', fontsize=10)
        axes2[1, 0].set_ylabel('Detected', color='white', fontsize=10)
        fig2.patch.set_facecolor('#111111')
        fig2.suptitle('Key Frames — Adherence Progression', color='white',
                      fontsize=13, y=0.98)
        fig2.tight_layout(rect=[0, 0, 1, 0.96])
        fig2.savefig(os.path.join(out_dir, 'contact_sheet.png'),
                     dpi=130, facecolor='#111111')
        plt.close(fig2)

        # 4. Per-frame overlay images (every frame)
        overlays_dir = os.path.join(out_dir, 'frame_overlays')
        os.makedirs(overlays_dir, exist_ok=True)

        for fi, fpath in enumerate(files):
            t_min  = fi * 60.0 / max(N - 1, 1)
            arr, _ = background_correct(fpath, params['bg_sigma'])
            vmin   = np.percentile(arr, 1)
            vmax   = np.percentile(arr, 99)

            fig3, ax3 = plt.subplots(figsize=(8, 6.7), facecolor='black')
            ax3.imshow(arr, cmap='gray', vmin=vmin, vmax=vmax)
            ax3.axis('off')

            if fi < len(track_hist):
                snap  = track_hist[fi]
                n_adh = 0
                for r, c, consec, _ in snap:
                    is_adh = consec >= min_f
                    color  = ADHERED_COLOR if is_adh else FLOATING_COLOR
                    radius = params['max_sigma'] * 0.85
                    ax3.add_patch(plt.Circle((c, r), radius, color=color,
                                             fill=False, lw=1.2, alpha=0.9))
                    if is_adh:
                        n_adh += 1
                n_flt = df[df['frame'] == fi + 1]['floating'].values
                n_flt = int(n_flt[0]) if len(n_flt) else 0
            else:
                n_adh = n_flt = 0

            ax3.set_title(f'Frame {fi+1:03d}  |  t = {t_min:.1f} min  |  '
                          f'Adhered: {n_adh}   Floating: {n_flt}',
                          color='white', fontsize=9, pad=4)

            adh_patch = mpatches.Patch(color=ADHERED_COLOR,
                                       label=f'Adhered (≥{min_f} frames): {n_adh}')
            flt_patch = mpatches.Patch(color=FLOATING_COLOR,
                                       label=f'Floating: {n_flt}')
            ax3.legend(handles=[adh_patch, flt_patch], loc='lower right',
                       fontsize=7, facecolor='#111111', labelcolor='white',
                       edgecolor='none')

            fig3.tight_layout(pad=0.3)
            fig3.savefig(os.path.join(overlays_dir, f'frame_{fi+1:03d}.png'),
                         dpi=100, facecolor='black')
            plt.close(fig3)

            pct = (fi + 1) / N * 100
            self.after(0, self.progress_var.set, pct)
            self.after(0, self._set_status,
                       f'Exporting frame overlays {fi+1}/{N}…')

        self.after(0, self._set_status,
                   f'Export complete → {out_dir}')
        self.after(0, self.progress_var.set, 100)
        self.after(0, messagebox.showinfo, 'Export done',
                   f'Report saved to:\n{out_dir}\n\n'
                   f'  adherence_counts.csv\n'
                   f'  timecourse_plot.png\n'
                   f'  contact_sheet.png\n'
                   f'  frame_overlays/  ({N} images)')

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        self.lbl_status.config(text=msg)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = AdherenceAnalyzer()
    app.mainloop()
