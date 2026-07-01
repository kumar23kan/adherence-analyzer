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

import sys
import importlib

_REQUIRED = ['numpy', 'PIL', 'skimage', 'scipy', 'pandas', 'matplotlib']
_MISSING  = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]
if _MISSING:
    print(
        '\nMissing dependencies: ' + ', '.join(_MISSING) + '\n'
        'Run this first:\n'
        '    pip install -r requirements.txt\n'
        '\nThen re-run:\n'
        '    python adherence_analyzer.py\n'
    )
    sys.exit(1)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import glob
import os
import re
import json

import numpy as np
from PIL import Image
from skimage import filters, feature, transform, draw as skdraw, measure as skmeasure, \
    segmentation as skseg, morphology as skmorphology
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector, PolygonSelector

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS = {
    'downsample':  ('Downsample (1=full, 2=½, 4=¼)', 2,    1,   4,   1),
    'bg_sigma':    ('Background σ (px)',              80,   10, 200,   1),
    'bright_thr':  ('Bright blob threshold σ',       1.5, 0.3,  5.0, 0.1),
    'dark_thr':    ('Dark blob threshold σ',         0.9, 0.2,  3.0, 0.1),
    'min_sigma':   ('Min cell radius (px)',           4,    1,  20,   1),
    'max_sigma':   ('Max cell radius (px)',          18,    5,  60,   1),
    'um_per_px':   ('µm/pixel (orig. resolution)',  0.065, 0.01, 2.0, 0.01),
    'hmax_h':      ('H-maxima height (0 = off)',    0.0,  0.0,  5.0, 0.1),
    'max_dist':    ('Max track distance (px)',        8,    1,  30,   1),
    'min_frames':  ('Min frames = adhered',           3,    1,  10,   1),
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


def background_correct(path, bg_sigma, user_ds=1):
    arr = np.array(Image.open(path)).astype(np.float32)
    if user_ds > 1:
        arr = arr[::user_ds, ::user_ds].copy()
    h, w = arr.shape[:2]
    # Compute gaussian on 4× downsampled image (16× less memory), then upsample.
    # bg_sigma is expressed in original pixels; divide by user_ds to get the
    # equivalent sigma in the already-downscaled arr, then by ds for the 4× tile.
    ds = 4
    small = arr[::ds, ::ds]
    bg_small = filters.gaussian(small, sigma=max(1.0, bg_sigma / (user_ds * ds)))
    bg = transform.resize(bg_small, (h, w), order=1, anti_aliasing=False,
                          preserve_range=True)
    del small, bg_small
    corr = arr - bg
    del bg
    std = corr.std()
    return arr, corr / (std if std > 0 else 1.0), std


def detect_spots(corr, bright_thr, dark_thr, min_sigma, max_sigma, hmax_h=0.0):
    """Return (Nx2 array of (row, col), n_floating) for detected cell centres.

    hmax_h > 0 enables h-maxima refinement: additional peaks are found inside
    merged blobs that LoG collapses to a single centre (common at high density).
    """
    h, w = corr.shape[:2]
    ds = 1
    if max(h, w) > 2048:
        ds = 4
    elif max(h, w) > 1024:
        ds = 2
    if ds > 1:
        corr = corr[::ds, ::ds].copy()

    kw = dict(min_sigma=min_sigma / ds, max_sigma=max_sigma / ds, num_sigma=4,
              exclude_border=False)
    bright = feature.blob_log( corr, threshold=bright_thr, **kw)[:, :2] * ds
    n_float = len(bright)
    dark   = feature.blob_log(-corr, threshold=dark_thr,   **kw)[:, :2] * ds

    if len(bright) == 0 and len(dark) == 0:
        del corr
        return np.empty((0, 2), dtype=np.float32), n_float
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
    spots = np.array(keep, dtype=np.float32) if keep else np.empty((0, 2), dtype=np.float32)

    # H-maxima refinement: recovers peaks inside merged blobs missed by LoG
    if hmax_h > 0.0:
        corr_pos = np.clip(corr, 0, None)
        if corr_pos.max() > 0:
            hmax    = skmorphology.h_maxima(corr_pos, h=float(hmax_h))
            hlabels = skmeasure.label(hmax)
            hprops  = skmeasure.regionprops(hlabels)
            if hprops:
                hpts = np.array([[p.centroid[0] * ds, p.centroid[1] * ds]
                                  for p in hprops], dtype=np.float32)
                combined = np.vstack([spots, hpts]) if len(spots) > 0 else hpts
                tree2 = cKDTree(combined)
                used2 = np.zeros(len(combined), bool)
                keep2 = []
                for i, p in enumerate(combined):
                    if used2[i]:
                        continue
                    for j in tree2.query_ball_point(p, r=5):
                        used2[j] = True
                    keep2.append(p)
                if keep2:
                    spots = np.array(keep2, dtype=np.float32)

    del corr
    return spots, n_float


def run_tracker(files, params, progress_cb=None, cancel_flag=None,
                roi_mask=None, compute_masks=False, overrides=None):
    """
    Run tracking over all frames.

    Returns
    -------
    results : list of dicts  (one per frame)
    track_history : list of lists
        track_history[fi] = list of (r, c, consec, track_id, in_roi)
        in_roi is True when the spot falls inside any ROI (or when no ROI is defined).
    mask_history : list of ndarray or None
        Per-frame watershed label array when compute_masks=True, else list of None.
    """
    bg_sigma   = params['bg_sigma']
    bright_thr = params['bright_thr']
    dark_thr   = params['dark_thr']
    min_sigma  = int(params['min_sigma'])
    max_sigma  = int(params['max_sigma'])
    max_dist   = params['max_dist']
    min_frames = int(params['min_frames'])
    hmax_h     = float(params.get('hmax_h', 0.0))
    user_ds    = max(1, int(params.get('downsample', 1)))
    overrides  = overrides or {}

    # Scale pixel-space params to the downsampled coordinate system.
    eff_min_sigma = max(1,   min_sigma // user_ds)
    eff_max_sigma = max(2,   max_sigma // user_ds)
    eff_max_dist  = max_dist / user_ds

    active_tracks = {}   # id -> {'pos': (r,c), 'consec': int}
    next_id       = 0
    results       = []
    track_history = []   # per-frame snapshot
    mask_history  = []   # per-frame watershed label array (or None)
    snr_series    = []   # background std per frame (SNR proxy)

    for fi, fpath in enumerate(files):
        if cancel_flag and cancel_flag():
            break

        # Apply per-frame threshold overrides if set
        fr_bright = overrides[fi].get('bright_thr', bright_thr) if fi in overrides else bright_thr
        fr_dark   = overrides[fi].get('dark_thr',   dark_thr)   if fi in overrides else dark_thr
        fr_hmax   = overrides[fi].get('hmax_h',     hmax_h)     if fi in overrides else hmax_h

        arr, corr, frame_snr = background_correct(fpath, bg_sigma, user_ds)
        snr_series.append(float(frame_snr))
        del arr
        spots_all, n_float = detect_spots(corr, fr_bright, fr_dark,
                                          eff_min_sigma, eff_max_sigma,
                                          hmax_h=fr_hmax)
        frame_no = fi + 1

        # Separate spots into in-ROI and out-of-ROI
        has_roi = roi_mask is not None
        if has_roi and len(spots_all) > 0:
            rows_i = np.clip(spots_all[:, 0].astype(int), 0, roi_mask.shape[0] - 1)
            cols_i = np.clip(spots_all[:, 1].astype(int), 0, roi_mask.shape[1] - 1)
            flags  = roi_mask[rows_i, cols_i] > 0
            spots      = spots_all[flags]
            spots_out  = spots_all[~flags]
        else:
            spots     = spots_all
            spots_out = np.empty((0, 2), dtype=np.float32)
            flags     = np.ones(len(spots_all), dtype=bool)

        # Watershed mask (generated from in-ROI spots, same coordinate space as corr)
        if compute_masks and len(spots) > 0:
            lbl_mask = generate_mask(corr, spots, eff_min_sigma, eff_max_sigma)
        else:
            lbl_mask = None
        mask_history.append(lbl_mask)
        del corr

        if len(spots) == 0:
            # Still record out-of-ROI spots in snapshot so preview can dim them
            snap_out = [(s[0], s[1], 0, -1, False) for s in spots_out]
            results.append({'frame': frame_no, 'adhered': 0,
                            'floating': n_float, 'total_detected': 0})
            track_history.append(snap_out)
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
                dists, idxs = tree.query(spots, distance_upper_bound=eff_max_dist + 1e-6)

                matched_prev  = {}
                for si in np.argsort(dists):
                    d, pi = dists[si], idxs[si]
                    if d > eff_max_dist:
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

        # Snapshot: in-ROI tracked spots (5-tuple) + out-of-ROI spots (dimmed)
        snap_in  = [(t['pos'][0], t['pos'][1], t['consec'], tid, True)
                    for tid, t in active_tracks.items()]
        snap_out = [(s[0], s[1], 0, -1, False) for s in spots_out]
        track_history.append(snap_in + snap_out)

        results.append({'frame': frame_no, 'adhered': n_adhered,
                        'floating': n_float, 'total_detected': len(spots)})

        if progress_cb:
            progress_cb(fi + 1, len(files))

    reliability = compute_reliability(results, snr_series)
    return results, track_history, mask_history, reliability


def generate_mask(corr, spots, min_sigma, max_sigma):
    """Watershed segmentation mask seeded from LoG spot centres.

    Returns an int32 label array (0 = background, N > 0 = cell N).
    Works on the already-downsampled corrected image passed from run_tracker.
    """
    H, W = corr.shape[:2]
    if len(spots) == 0:
        return np.zeros((H, W), dtype=np.int32)

    # Foreground binary mask via Otsu on the positive signal
    corr_pos = np.clip(corr, 0, None).astype(np.float32)
    if corr_pos.max() > 0:
        try:
            thr = filters.threshold_otsu(corr_pos)
        except Exception:
            thr = corr_pos.mean()
        fg = corr_pos > thr
    else:
        return np.zeros((H, W), dtype=np.int32)

    if not fg.any():
        return np.zeros((H, W), dtype=np.int32)

    # Seed markers: small disk at each detected spot
    seed_r = max(1, int(min_sigma * 0.7))
    markers = np.zeros((H, W), dtype=np.int32)
    for i, (r, c) in enumerate(spots):
        rr, cc = skdraw.disk((int(r), int(c)), seed_r, shape=(H, W))
        markers[rr, cc] = i + 1   # label 1-based

    # Distance transform + watershed
    dist   = distance_transform_edt(fg)
    labels = skseg.watershed(-dist, markers=markers, mask=fg)
    return labels.astype(np.int32)


def mask_to_contours(label_mask, track_snap, min_frames):
    """Convert watershed label mask to a list of (xy_pts, color, alpha) for drawing.

    xy_pts is an Nx2 array in (col, row) = (x, y) matplotlib order.
    Color and alpha are chosen by matching each label's centroid to the
    nearest tracked spot in track_snap.
    """
    if label_mask is None:
        return []
    props = skmeasure.regionprops(label_mask)
    if not props:
        return []

    result = []
    snap_pos = np.array([[e[0], e[1]] for e in track_snap], dtype=np.float32) \
        if track_snap else np.empty((0, 2), dtype=np.float32)

    for prop in props:
        cr, cc = prop.centroid
        if len(snap_pos) > 0:
            dists  = np.hypot(snap_pos[:, 0] - cr, snap_pos[:, 1] - cc)
            idx    = int(np.argmin(dists))
            entry  = track_snap[idx]
            in_roi = entry[4]
            consec = entry[2]
            if in_roi and consec >= min_frames:
                color, alpha = ADHERED_COLOR, 0.80
            elif in_roi:
                color, alpha = FLOATING_COLOR, 0.70
            else:
                color, alpha = '#6c7086', 0.20
        else:
            color, alpha = '#6c7086', 0.20

        try:
            contours = skmeasure.find_contours(label_mask == prop.label, 0.5)
        except Exception:
            continue
        if not contours:
            continue
        contour = max(contours, key=len)        # longest = outer boundary
        xy = contour[:, ::-1].astype(np.float32)   # (row,col) → (x,y)
        result.append((xy, color, alpha))
    return result


def save_mask_tiff(label_mask, out_path):
    """Save binary mask TIFF (255 = cell, 0 = background)."""
    binary = (label_mask > 0).astype(np.uint8) * 255
    Image.fromarray(binary).save(out_path)


def compute_coverage(label_mask, roi_binary_mask, um_per_px_eff):
    """Coverage of bacteria pixels within one ROI.

    Parameters
    ----------
    label_mask      : int32 watershed array (0 = background)
    roi_binary_mask : uint8 array (1 = inside this ROI)
    um_per_px_eff   : µm per pixel in the downsampled coordinate space
                      = user-entered µm/pixel × user_ds

    Returns
    -------
    dict with keys 'area_fraction' (0–1) and 'area_um2' (float)
    """
    roi_pixels  = int(roi_binary_mask.sum())
    if roi_pixels == 0:
        return {'area_fraction': 0.0, 'area_um2': 0.0}
    cell_pixels = int(((label_mask > 0) & (roi_binary_mask > 0)).sum())
    return {
        'area_fraction': cell_pixels / roi_pixels,
        'area_um2':      cell_pixels * (um_per_px_eff ** 2),
    }


def compute_reliability(results, snr_series, jump_sigma=3.0):
    """Flag frames with potentially unreliable detections.

    Three independent checks per frame:
      1. count-jump      — total_detected deviates > jump_sigma×std from local
                           7-frame window (excludes the frame itself)
      2. low-SNR         — background std below 10th-percentile of the series
      3. abnormal-density — total_detected outside the 5th–95th percentile range

    Returns a list of dicts (one per frame): {'flagged': bool, 'reasons': [str]}
    """
    n = len(results)
    if n == 0:
        return []
    counts = np.array([r['total_detected'] for r in results], dtype=np.float32)
    snr    = np.array(snr_series, dtype=np.float32) if snr_series else np.array([])

    snr_thresh    = float(np.percentile(snr,    10)) if len(snr)    > 1 else -np.inf
    density_lo    = float(np.percentile(counts,  5)) if n > 1 else 0.0
    density_hi    = float(np.percentile(counts, 95)) if n > 1 else np.inf
    window        = 7

    flags = []
    for i in range(n):
        reasons = []

        # 1. Count-jump: compare to neighbours in a local window
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        neighbours = counts[np.arange(lo, hi)[np.arange(lo, hi) != i]]
        if len(neighbours) >= 3:
            mu  = float(neighbours.mean())
            sig = float(neighbours.std())
            if sig > 0 and abs(float(counts[i]) - mu) > jump_sigma * sig:
                reasons.append('count-jump')

        # 2. Low-SNR
        if i < len(snr) and float(snr[i]) < snr_thresh:
            reasons.append('low-SNR')

        # 3. Abnormal density
        if float(counts[i]) < density_lo or float(counts[i]) > density_hi:
            reasons.append('abnormal-density')

        flags.append({'flagged': bool(reasons), 'reasons': reasons})
    return flags


def _hstack(img_a, img_b):
    """Horizontally concatenate two PIL images of equal height."""
    out = Image.new('RGB', (img_a.width + img_b.width, img_a.height))
    out.paste(img_a, (0, 0))
    out.paste(img_b, (img_a.width, 0))
    return out


# ── ROI persistence & masking ────────────────────────────────────────────────

def save_rois(folder, rois):
    path = os.path.join(folder, 'roi.json')
    with open(path, 'w') as f:
        json.dump(rois, f, indent=2)


def load_rois(folder):
    path = os.path.join(folder, 'roi.json')
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return []


def build_roi_mask(shape, rois):
    """Binary uint8 mask (H, W): 1 inside union of all ROIs, 0 outside.
    ROI coords are in display (downsampled) pixel space — x=col, y=row.
    Returns all-ones mask when rois is empty (everything active)."""
    H, W = shape[:2]
    if not rois:
        return np.ones((H, W), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)
    for roi in rois:
        if roi['shape'] == 'rectangle':
            x0, y0, x1, y1 = roi['coords']
            r0 = max(0, int(min(y0, y1)))
            r1 = min(H - 1, int(max(y0, y1)))
            c0 = max(0, int(min(x0, x1)))
            c1 = min(W - 1, int(max(x0, x1)))
            mask[r0:r1 + 1, c0:c1 + 1] = 1
        else:
            coords = roi['coords']
            rows = np.array([pt[1] for pt in coords], dtype=np.float64)
            cols = np.array([pt[0] for pt in coords], dtype=np.float64)
            rr, cc = skdraw.polygon(rows, cols, shape=(H, W))
            mask[rr, cc] = 1
    return mask


# ── GUI ───────────────────────────────────────────────────────────────────────

class AdherenceAnalyzer(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title('Cell Tracker — Bacterial Adherence Analysis')
        self.configure(bg='#1e1e2e')
        self.resizable(True, True)

        # State
        self.image_files   = []
        self.current_fi    = 0
        self.results       = []
        self.track_history = []
        self._cancel_flag  = False
        self._running      = False
        self.loaded_folder = ''
        self.condition_label = tk.StringVar(value='')

        # ROI state
        self.rois              = []
        self.selected_roi_idx  = -1
        self._roi_shape_var    = tk.StringVar(value='rectangle')
        self._draw_mode_active = False
        self._rect_selector    = None
        self._poly_selector    = None

        # Mask state
        self.mask_history      = []
        self.compute_masks_var = tk.BooleanVar(value=False)
        self._radio_mask       = None

        # Per-frame threshold override state
        self.frame_overrides      = {}   # {frame_index: {'bright_thr': f, 'dark_thr': f}}
        self._ov_bright           = tk.StringVar()
        self._ov_dark             = tk.StringVar()
        self._override_frame_lbl  = tk.StringVar(value='Frame: –')

        # Detection quality state
        self.reliability        = []   # list of {'flagged': bool, 'reasons': [str]}
        self._flagged_positions = []   # frame indices flagged as unreliable

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
        tk.Label(top, text='Cell Tracker', bg='#181825', fg='#89b4fa',
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

        # Experiment group
        exp = ttk.LabelFrame(frame, text='Experiment', padding=8)
        exp.pack(fill='x', pady=(4, 4))
        tk.Label(exp, text='Condition label', bg='#1e1e2e', fg='#cdd6f4',
                 font=('Helvetica', 8)).pack(anchor='w')
        ttk.Entry(exp, textvariable=self.condition_label,
                  font=('Helvetica', 9)).pack(fill='x', pady=(2, 0))

        # Detection group
        det = ttk.LabelFrame(frame, text='Detection', padding=8)
        det.pack(fill='x', pady=(4, 4))
        det_keys = ['downsample', 'bg_sigma', 'bright_thr', 'dark_thr',
                    'min_sigma', 'max_sigma', 'um_per_px', 'hmax_h']
        self._make_param_rows(det, det_keys)
        tk.Checkbutton(det, text='Compute masks (watershed)',
                       variable=self.compute_masks_var,
                       bg='#1e1e2e', fg='#cdd6f4', selectcolor='#313244',
                       activebackground='#1e1e2e', font=('Helvetica', 8)
                       ).pack(anchor='w', pady=(4, 0))

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

        # Frame override panel
        ov_frm = ttk.LabelFrame(frame, text='Frame Override', padding=8)
        ov_frm.pack(fill='x', pady=(0, 4))

        tk.Label(ov_frm, textvariable=self._override_frame_lbl,
                 bg='#1e1e2e', fg='#89b4fa', font=('Helvetica', 8, 'bold')
                 ).pack(anchor='w')

        for lbl, var in [('bright_thr', self._ov_bright), ('dark_thr', self._ov_dark)]:
            row = tk.Frame(ov_frm, bg='#1e1e2e')
            row.pack(fill='x', pady=1)
            tk.Label(row, text=lbl, bg='#1e1e2e', fg='#cdd6f4',
                     font=('Helvetica', 8), width=10, anchor='w').pack(side='left')
            ttk.Entry(row, textvariable=var, width=7).pack(side='left', padx=4)

        ov_btn_row = tk.Frame(ov_frm, bg='#1e1e2e')
        ov_btn_row.pack(fill='x', pady=(4, 2))
        ttk.Button(ov_btn_row, text='Apply', command=self._apply_override,
                   style='TButton').pack(side='left', expand=True, fill='x', padx=(0, 2))
        ttk.Button(ov_btn_row, text='Clear', command=self._clear_override,
                   style='TButton').pack(side='left', expand=True, fill='x', padx=(0, 2))
        ttk.Button(ov_btn_row, text='Clear All', command=self._clear_all_overrides,
                   style='TButton').pack(side='left', expand=True, fill='x')

        self.override_listbox = tk.Listbox(ov_frm, bg='#313244', fg='#cdd6f4',
                                           height=3, font=('Helvetica', 7),
                                           selectbackground='#89b4fa',
                                           selectforeground='#1e1e2e',
                                           activestyle='none')
        self.override_listbox.pack(fill='x', pady=(2, 0))

        # ROI panel
        roi_frm = ttk.LabelFrame(frame, text='Regions of Interest', padding=8)
        roi_frm.pack(fill='x', pady=(0, 4))

        shape_row = tk.Frame(roi_frm, bg='#1e1e2e')
        shape_row.pack(fill='x', pady=(0, 4))
        tk.Label(shape_row, text='Shape:', bg='#1e1e2e', fg='#cdd6f4',
                 font=('Helvetica', 8)).pack(side='left')
        ttk.OptionMenu(shape_row, self._roi_shape_var,
                       'rectangle', 'rectangle', 'polygon').pack(side='left', padx=6)

        self.btn_draw_roi = ttk.Button(roi_frm, text='Draw ROI',
                                       command=self._toggle_draw_roi, style='TButton')
        self.btn_draw_roi.pack(fill='x', pady=(0, 4))

        self.roi_listbox = tk.Listbox(roi_frm, bg='#313244', fg='#cdd6f4',
                                      selectbackground='#89b4fa', selectforeground='#1e1e2e',
                                      height=4, font=('Helvetica', 8), activestyle='none')
        self.roi_listbox.pack(fill='x', pady=(0, 2))
        self.roi_listbox.bind('<<ListboxSelect>>', self._on_roi_select)

        btn_row2 = tk.Frame(roi_frm, bg='#1e1e2e')
        btn_row2.pack(fill='x')
        ttk.Button(btn_row2, text='Delete', command=self._delete_roi,
                   style='TButton').pack(side='left', expand=True, fill='x', padx=(0, 2))
        ttk.Button(btn_row2, text='Clear All', command=self._clear_rois,
                   style='TButton').pack(side='left', expand=True, fill='x')

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
            if getattr(self, '_debounce_id', None) is not None:
                self.after_cancel(self._debounce_id)
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
        self._radio_mask = tk.Radiobutton(
            mode_frame, text='Mask', variable=self.show_mode,
            value='masks', bg='#181825', fg='#cdd6f4',
            selectcolor='#313244', activebackground='#181825',
            state='disabled', command=self._update_preview)
        self._radio_mask.pack(side='left', padx=4)

        # Thin tick canvas: orange marks at flagged frames (row=2)
        self.tick_canvas = tk.Canvas(frame, height=8, bg='#181825',
                                     highlightthickness=0)
        self.tick_canvas.grid(row=2, column=0, sticky='ew')
        self.tick_canvas.bind('<Configure>', lambda e: self._redraw_ticks())

    def _mark_unreliable_frames(self):
        self._flagged_positions = [
            fi for fi, r in enumerate(self.reliability) if r['flagged']
        ]
        self._redraw_ticks()

    def _redraw_ticks(self):
        if not hasattr(self, 'tick_canvas'):
            return
        self.tick_canvas.delete('all')
        n = len(self.image_files)
        if n == 0 or not self._flagged_positions:
            return
        w = self.tick_canvas.winfo_width()
        h = self.tick_canvas.winfo_height()
        if w < 2 or h < 1:
            return
        for fi in self._flagged_positions:
            x = int(fi / max(n - 1, 1) * (w - 2)) + 1
            self.tick_canvas.create_line(x, 0, x, h, fill='#fab387', width=2)

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
        self.image_files   = files
        self.current_fi    = 0
        self.results       = []
        self.track_history = []
        self.loaded_folder = folder
        self.frame_slider.config(to=len(files) - 1)
        short = os.path.basename(folder)
        self.condition_label.set(short)
        self.lbl_folder.config(text=f'{short}  ({len(files)} frames)', fg='#a6e3a1')

        # Load ROIs saved alongside this folder's images
        self._disconnect_selectors()
        self._draw_mode_active = False
        self.btn_draw_roi.config(text='Draw ROI')
        self.rois = load_rois(folder)
        self.selected_roi_idx = -1
        self._refresh_roi_listbox()

        # Reset per-frame overrides for new folder
        self.frame_overrides = {}
        self._ov_bright.set('')
        self._ov_dark.set('')
        self._refresh_override_listbox()

        self._set_status(f'Loaded {len(files)} frames from: {folder}')
        self._update_preview()

    def _nav_frame(self, delta):
        if not self.image_files:
            return
        self._disconnect_selectors()
        self._draw_mode_active = False
        self.btn_draw_roi.config(text='Draw ROI')
        self.current_fi = max(0, min(len(self.image_files) - 1, self.current_fi + delta))
        self.frame_slider.set(self.current_fi)
        self._update_preview()

    def _on_slider(self, val):
        fi = int(float(val))
        if fi != self.current_fi:
            self._disconnect_selectors()
            self._draw_mode_active = False
            self.btn_draw_roi.config(text='Draw ROI')
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

        # Sync override panel to current frame
        self._override_frame_lbl.set(
            f'Frame {fi+1}' + (' ★ overridden' if fi in self.frame_overrides else ''))
        if fi in self.frame_overrides:
            ov = self.frame_overrides[fi]
            self._ov_bright.set(str(ov.get('bright_thr', '')))
            self._ov_dark.set(str(ov.get('dark_thr', '')))
        else:
            self._ov_bright.set('')
            self._ov_dark.set('')
        self.lbl_frame_info.config(text=f'Frame {fi+1}/{total}  ({t_min:.1f} min)')

        p       = self._get_params()
        user_ds = max(1, int(p.get('downsample', 1)))
        arr, corr, _ = background_correct(fpath, p['bg_sigma'], user_ds)

        self.ax_prev.cla()
        self.ax_prev.set_facecolor('#181825')
        self.ax_prev.axis('off')

        vmin = np.percentile(arr, 1)
        vmax = np.percentile(arr, 99)
        self.ax_prev.imshow(arr, cmap='gray', vmin=vmin, vmax=vmax, aspect='auto')

        if self.show_mode.get() == 'masks':
            snap = self.track_history[fi] if fi < len(self.track_history) else []
            lbl  = self.mask_history[fi]  if fi < len(self.mask_history)  else None
            min_f = int(p['min_frames'])
            if lbl is not None:
                contours = mask_to_contours(lbl, snap, min_f)
                for xy, color, alpha in contours:
                    if len(xy) < 3:
                        continue
                    codes  = ([MplPath.MOVETO]
                              + [MplPath.LINETO] * (len(xy) - 2)
                              + [MplPath.CLOSEPOLY])
                    path   = MplPath(np.vstack([xy, xy[:1]]), codes)
                    self.ax_prev.add_patch(
                        PathPatch(path, facecolor=color, edgecolor='none',
                                  alpha=alpha * 0.35))
                    self.ax_prev.add_patch(
                        PathPatch(path, facecolor='none', edgecolor=color,
                                  linewidth=0.9, alpha=alpha))
                n_adh = sum(1 for e in snap if e[4] and e[2] >= min_f)
                info  = f'Adhered: {n_adh}  (mask view)'
            else:
                info = 'No mask — enable "Compute masks" and re-run'
            self.ax_prev.text(6, 18, info, color='white', fontsize=9,
                              fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e2e',
                                        alpha=0.7, edgecolor='none'))

        elif self.show_mode.get() == 'detected':
            eff_min = max(1, int(p['min_sigma']) // user_ds)
            eff_max = max(2, int(p['max_sigma']) // user_ds)
            spots, n_float = detect_spots(corr, p['bright_thr'], p['dark_thr'],
                                          eff_min, eff_max)
            radius = int(p['max_sigma']) * 0.8

            # If we have track history for this frame, use it
            if fi < len(self.track_history) and self.track_history:
                snap  = self.track_history[fi]
                min_f = int(p['min_frames'])
                for entry in snap:
                    r, c, consec, tid, in_roi = entry
                    if in_roi:
                        color = ADHERED_COLOR if consec >= min_f else FLOATING_COLOR
                        alpha = 0.85
                        lw    = 1.2
                    else:
                        color = '#6c7086'
                        alpha = 0.25
                        lw    = 0.8
                    circ = plt.Circle((c, r), radius, color=color,
                                      fill=False, lw=lw, alpha=alpha)
                    self.ax_prev.add_patch(circ)
                    if in_roi and consec >= min_f:
                        self.ax_prev.text(c + radius, r, str(consec),
                                          color=ADHERED_COLOR, fontsize=4, va='center')
                n_adh = sum(1 for e in snap if e[4] and e[2] >= int(p['min_frames']))
                info  = f'Adhered: {n_adh}   Floating: {n_float}'
            else:
                # Pre-analysis: show raw detections, dim spots outside ROIs
                roi_mask = build_roi_mask(arr.shape, self.rois)
                for r, c in spots:
                    ri, ci  = int(np.clip(r, 0, roi_mask.shape[0]-1)), \
                              int(np.clip(c, 0, roi_mask.shape[1]-1))
                    in_roi  = bool(roi_mask[ri, ci])
                    color   = '#89b4fa' if in_roi else '#6c7086'
                    alpha   = 0.6      if in_roi else 0.2
                    circ = plt.Circle((c, r), radius, color=color,
                                      fill=False, lw=0.8, alpha=alpha)
                    self.ax_prev.add_patch(circ)
                n_in = int(roi_mask[
                    np.clip(spots[:, 0].astype(int), 0, roi_mask.shape[0]-1),
                    np.clip(spots[:, 1].astype(int), 0, roi_mask.shape[1]-1)
                ].sum()) if len(spots) > 0 else 0
                info = f'Detected: {len(spots)}  (in ROI: {n_in})'

            self.ax_prev.text(6, 18, info, color='white',
                              fontsize=9, fontweight='bold',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e2e',
                                        alpha=0.7, edgecolor='none'))

        # Quality flag badge (all view modes)
        if fi < len(self.reliability) and self.reliability[fi]['flagged']:
            reasons = ', '.join(self.reliability[fi]['reasons'])
            self.ax_prev.text(
                0.01, 0.02, f'⚠  {reasons}',
                color='#fab387', fontsize=8, fontweight='bold', va='bottom',
                transform=self.ax_prev.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#1e1e2e',
                          alpha=0.8, edgecolor='#fab387'))

        # Draw ROI boundaries on top
        for i, roi in enumerate(self.rois):
            is_sel = (i == self.selected_roi_idx)
            clr    = '#f5c2e7' if is_sel else '#f38ba8'
            lw     = 2.5       if is_sel else 1.5
            if roi['shape'] == 'rectangle':
                x0, y0, x1, y1 = roi['coords']
                patch = mpatches.Rectangle(
                    (min(x0, x1), min(y0, y1)), abs(x1 - x0), abs(y1 - y0),
                    linewidth=lw, edgecolor=clr, facecolor='none',
                    linestyle='--', alpha=0.85)
                self.ax_prev.add_patch(patch)
                self.ax_prev.text(min(x0, x1) + 3, min(y0, y1) + 3, roi['name'],
                                  color=clr, fontsize=6, va='top', alpha=0.95)
            else:
                pts   = np.array([[p[0], p[1]] for p in roi['coords']])
                patch = mpatches.Polygon(pts, closed=True, linewidth=lw,
                                         edgecolor=clr, facecolor='none',
                                         linestyle='--', alpha=0.85)
                self.ax_prev.add_patch(patch)
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                self.ax_prev.text(cx, cy, roi['name'],
                                  color=clr, fontsize=6, ha='center', va='center', alpha=0.95)

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
        # Build ROI mask in the main thread before handing off (thread-safe snapshot)
        if self.image_files and self.rois:
            sample_arr, _, _ = background_correct(self.image_files[0], params['bg_sigma'],
                                                 max(1, int(params.get('downsample', 1))))
            roi_mask = build_roi_mask(sample_arr.shape, self.rois)
        else:
            roi_mask = None
        compute_masks    = bool(self.compute_masks_var.get())
        overrides_snap   = dict(self.frame_overrides)   # thread-safe snapshot
        threading.Thread(target=self._analysis_thread,
                         args=(params, roi_mask, compute_masks, overrides_snap),
                         daemon=True).start()

    def _analysis_thread(self, params, roi_mask, compute_masks, overrides):
        def prog(done, total):
            pct = done / total * 100
            self.after(0, self.progress_var.set, pct)
            self.after(0, self._set_status,
                       f'Processing frame {done}/{total}…')

        results, track_history, mask_history, reliability = run_tracker(
            self.image_files, params,
            progress_cb=prog,
            cancel_flag=lambda: self._cancel_flag,
            roi_mask=roi_mask,
            compute_masks=compute_masks,
            overrides=overrides
        )
        self.after(0, self._analysis_done, results, track_history, mask_history, reliability)

    def _analysis_done(self, results, track_history, mask_history, reliability):
        self._running      = False
        self._cancel_flag  = False
        self.btn_run.config(text='▶  Run Analysis')
        self.results       = results
        self.track_history = track_history
        self.mask_history  = mask_history
        self.reliability   = reliability

        # Enable Mask view only when masks were actually computed
        has_masks = any(m is not None for m in mask_history)
        if self._radio_mask is not None:
            self._radio_mask.config(state='normal' if has_masks else 'disabled')
        if not has_masks and self.show_mode.get() == 'masks':
            self.show_mode.set('detected')

        df = pd.DataFrame(results)
        N  = len(self.image_files)
        df['condition'] = self.condition_label.get()
        df['time_min'] = (df['frame'] - 1) * (60.0 / max(N - 1, 1))
        df['adhered_roll5']  = df['adhered'].rolling(5,  center=True, min_periods=1).mean()
        df['floating_roll5'] = df['floating'].rolling(5, center=True, min_periods=1).mean()

        # Per-ROI counts (post-hoc spatial filter of track_history)
        if self.rois and track_history:
            min_f = int(self._get_params()['min_frames'])
            for roi in self.rois:
                adh_col = f'adhered_{roi["name"]}'
                flt_col = f'floating_{roi["name"]}'
                adh_vals, flt_vals = [], []
                for fi, snap in enumerate(track_history):
                    if roi['shape'] == 'rectangle':
                        x0, y0, x1, y1 = roi['coords']
                        def _in(r, c):
                            return (min(y0,y1) <= r <= max(y0,y1) and
                                    min(x0,x1) <= c <= max(x0,x1))
                    else:
                        pts_r = [pt[1] for pt in roi['coords']]
                        pts_c = [pt[0] for pt in roi['coords']]
                        def _in(r, c, _r=pts_r, _c=pts_c):
                            return bool(skmeasure.points_in_poly(
                                np.array([[r, c]]),
                                np.column_stack([_r, _c]))[0])
                    n_adh = sum(1 for e in snap if e[4] and e[2] >= min_f and _in(e[0], e[1]))
                    n_flt = results[fi]['floating'] if fi < len(results) else 0
                    adh_vals.append(n_adh)
                    flt_vals.append(n_flt)
                df[adh_col] = adh_vals
                df[flt_col] = flt_vals

        # Coverage metrics per ROI per frame (requires watershed masks)
        has_masks = any(m is not None for m in mask_history)
        if self.rois and has_masks:
            p        = self._get_params()
            user_ds  = max(1, int(p.get('downsample', 1)))
            um_eff   = float(p.get('um_per_px', 0.065)) * user_ds
            for roi in self.rois:
                frac_vals, um2_vals = [], []
                roi_bin = None   # built lazily from first non-None mask shape
                for fi, lbl in enumerate(mask_history):
                    if lbl is None:
                        frac_vals.append(float('nan'))
                        um2_vals.append(float('nan'))
                        continue
                    if roi_bin is None or roi_bin.shape != lbl.shape:
                        roi_bin = build_roi_mask(lbl.shape, [roi])
                    cov = compute_coverage(lbl, roi_bin, um_eff)
                    frac_vals.append(cov['area_fraction'])
                    um2_vals.append(cov['area_um2'])
                df[f'area_frac_{roi["name"]}'] = frac_vals
                df[f'area_um2_{roi["name"]}']  = um2_vals

        if reliability:
            df['quality_flag'] = [r['flagged'] for r in reliability]
            df['flag_reason']  = [', '.join(r['reasons']) for r in reliability]

        self._df = df

        self._update_results_plot(df)
        self._update_stats(df)
        self._mark_unreliable_frames()
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

        condition = self.condition_label.get()
        ax.set_xlabel('Time (min)', color='#a6adc8', fontsize=8)
        ax.set_ylabel('Cell count',  color='#a6adc8', fontsize=8)
        title = f'Adherence time course — {condition}' if condition else 'Adherence time course'
        ax.set_title(title, color='#cdd6f4', fontsize=9, pad=4)
        ax.legend(fontsize=7, facecolor='#313244', labelcolor='#cdd6f4',
                  edgecolor='none')
        ax.grid(True, color='#313244', alpha=0.6)
        ax.set_ylim(bottom=0)

        self.fig_res.tight_layout(pad=0.8)
        self.canvas_res.draw()

    def _update_stats(self, df):
        p1 = df[df['time_min'] <= 20]
        p2 = df[df['time_min'] >  20]
        condition = self.condition_label.get()
        lines = [f'Condition: {condition}', ''] if condition else []
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
        import gc
        from PIL import ImageDraw, ImageFont

        df         = self._df
        files      = self.image_files
        track_hist = self.track_history
        condition  = self.condition_label.get()
        N          = len(files)
        min_f      = int(params['min_frames'])
        user_ds    = max(1, int(params.get('downsample', 1)))
        # radius and lw are in downscaled-image pixels (matching track_history coords)
        radius     = int(params['max_sigma'] / user_ds * 0.85)
        lw         = max(2, radius // 6)
        out_dpi    = max(72, int(600 / user_ds))

        # Decode colour strings to RGB tuples for PIL
        def hex2rgb(h):
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        adh_rgb = hex2rgb(ADHERED_COLOR)
        flt_rgb = hex2rgb(FLOATING_COLOR)

        # Try to load a small system font for overlay text; fall back to default
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 28)
            font_sm = ImageFont.truetype(
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
        except Exception:
            font = font_sm = ImageFont.load_default()

        # 1. CSV
        df.to_csv(os.path.join(out_dir, 'adherence_counts.csv'), index=False)

        # 2. Summary time course plot (matplotlib — only 2 figures total)
        self.after(0, self._set_status, 'Exporting time course plot…')
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), facecolor='white')
        for ax in axes:
            ax.set_facecolor('white')
        ax = axes[0]
        ax.plot(df['time_min'], df['adhered'],       color=ADHERED_COLOR, alpha=0.3, lw=1)
        ax.plot(df['time_min'], df['adhered_roll5'], color=ADHERED_COLOR,
                lw=2.5, label='Adhered (5-frame mean)')
        ax.plot(df['time_min'], df['floating'],       color=FLOATING_COLOR, alpha=0.3, lw=1)
        ax.plot(df['time_min'], df['floating_roll5'], color=FLOATING_COLOR,
                lw=2.0, ls='--', label='Floating (bright round)')
        ax.set_ylabel('Cell count', fontsize=11)
        ax.set_title('Adherence Time Course', fontsize=13)
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
        ax2 = axes[1]
        ax2.fill_between(df['time_min'], 0, df['adhered_roll5'],
                         alpha=0.55, color=ADHERED_COLOR, label='Adhered')
        ax2.fill_between(df['time_min'], df['adhered_roll5'],
                         df['adhered_roll5'] + df['floating_roll5'],
                         alpha=0.40, color=FLOATING_COLOR, label='Floating')
        ax2.set_xlabel('Time (min)', fontsize=11); ax2.set_ylabel('Cell count', fontsize=11)
        ax2.set_title('Stacked: Adhered vs Floating', fontsize=12)
        ax2.legend(fontsize=10); ax2.grid(True, alpha=0.3); ax2.set_ylim(bottom=0)
        condition = self.condition_label.get()
        if condition:
            fig.suptitle(condition, fontsize=14, fontweight='bold', y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'timecourse_plot.tiff'), dpi=600, bbox_inches='tight')
        plt.close(fig); plt.close('all'); gc.collect()

        # 3. Contact sheet — 5 key frames, drawn with PIL (memory-safe)
        self.after(0, self._set_status, 'Exporting contact sheet…')
        key_indices = [int(x * (N - 1) / 4) for x in range(5)]
        contact_imgs = []
        for fi in key_indices:
            fpath = files[fi]
            t_min = fi * 60.0 / max(N - 1, 1)
            raw   = np.array(Image.open(fpath))
            if user_ds > 1:
                raw = raw[::user_ds, ::user_ds].copy()
            lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
            norm  = np.clip((raw.astype(np.float32) - lo) / max(hi - lo, 1) * 255,
                            0, 255).astype(np.uint8)

            raw_pil = Image.fromarray(norm).convert('RGB')
            det_pil = raw_pil.copy()
            draw    = ImageDraw.Draw(det_pil)

            n_adh = 0
            if fi < len(track_hist):
                for entry in track_hist[fi]:
                    r, c, consec, _, in_roi = entry
                    is_adh = consec >= min_f and in_roi
                    col_px = adh_rgb if is_adh else (flt_rgb if in_roi else (100, 100, 120))
                    alpha_lw = lw if in_roi else max(1, lw // 2)
                    x0, y0 = int(c) - radius, int(r) - radius
                    x1, y1 = int(c) + radius, int(r) + radius
                    draw.ellipse([x0, y0, x1, y1], outline=col_px, width=alpha_lw)
                    if is_adh:
                        n_adh += 1
                n_flt = df[df['frame'] == fi + 1]['floating'].values
                n_flt = int(n_flt[0]) if len(n_flt) else 0
                draw.text((10, 10),
                          f'Adhered: {n_adh}  Floating: {n_flt}',
                          fill=(255, 255, 255), font=font_sm)

            # Stack raw on top of detected, add frame label
            gap    = Image.new('RGB', (raw_pil.width, 6), (17, 17, 27))
            label  = Image.new('RGB', (raw_pil.width, 50), (17, 17, 27))
            ld     = ImageDraw.Draw(label)
            ld.text((4, 8), f'Frame {fi+1}  t={t_min:.1f} min',
                    fill=(180, 180, 220), font=font_sm)
            col_img = Image.new('RGB',
                                (raw_pil.width, label.height + raw_pil.height + gap.height + det_pil.height))
            col_img.paste(label,   (0, 0))
            col_img.paste(raw_pil, (0, label.height))
            col_img.paste(gap,     (0, label.height + raw_pil.height))
            col_img.paste(det_pil, (0, label.height + raw_pil.height + gap.height))
            contact_imgs.append(col_img)
            del raw_pil, det_pil, draw, label, ld, raw, norm
            gc.collect()

        sep   = Image.new('RGB', (10, contact_imgs[0].height), (17, 17, 27))
        sheet = contact_imgs[0]
        for ci in contact_imgs[1:]:
            sheet = _hstack(sheet, sep)
            sheet = _hstack(sheet, ci)
        sheet.save(os.path.join(out_dir, 'contact_sheet.tiff'), dpi=(out_dpi, out_dpi))
        del sheet, contact_imgs, sep; gc.collect()

        # 4. Per-frame overlays — PIL only, one image at a time (no matplotlib)
        overlays_dir = os.path.join(out_dir, 'frame_overlays')
        os.makedirs(overlays_dir, exist_ok=True)

        for fi, fpath in enumerate(files):
            t_min = fi * 60.0 / max(N - 1, 1)
            raw   = np.array(Image.open(fpath))
            if user_ds > 1:
                raw = raw[::user_ds, ::user_ds].copy()
            lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
            norm  = np.clip((raw.astype(np.float32) - lo) / max(hi - lo, 1) * 255,
                            0, 255).astype(np.uint8)
            img   = Image.fromarray(norm).convert('RGB')
            draw  = ImageDraw.Draw(img)

            n_adh = 0
            if fi < len(track_hist):
                for entry in track_hist[fi]:
                    r, c, consec, _, in_roi = entry
                    is_adh = consec >= min_f and in_roi
                    col_px = adh_rgb if is_adh else (flt_rgb if in_roi else (100, 100, 120))
                    alpha_lw = lw if in_roi else max(1, lw // 2)
                    x0, y0 = int(c) - radius, int(r) - radius
                    x1, y1 = int(c) + radius, int(r) + radius
                    draw.ellipse([x0, y0, x1, y1], outline=col_px, width=alpha_lw)
                    if is_adh:
                        n_adh += 1
                n_flt = df[df['frame'] == fi + 1]['floating'].values
                n_flt = int(n_flt[0]) if len(n_flt) else 0
            else:
                n_flt = 0

            # Header bar with frame info
            bar  = Image.new('RGB', (img.width, 55), (17, 17, 27))
            bd   = ImageDraw.Draw(bar)
            cond_prefix = f'{condition}  |  ' if condition else ''
            bd.text((8, 6),
                    f'{cond_prefix}Frame {fi+1:03d}  |  t = {t_min:.1f} min  |  '
                    f'Adhered: {n_adh}   Floating: {n_flt}',
                    fill=(200, 200, 230), font=font)
            # Legend bar
            leg  = Image.new('RGB', (img.width, 40), (17, 17, 27))
            ld2  = ImageDraw.Draw(leg)
            ld2.rectangle([8, 12, 28, 32],  fill=adh_rgb)
            ld2.text((34, 12), f'Adhered (≥{min_f} frames)',
                     fill=(200, 200, 230), font=font_sm)
            ld2.rectangle([260, 12, 280, 32], fill=flt_rgb)
            ld2.text((286, 12), 'Floating', fill=(200, 200, 230), font=font_sm)

            out_img = Image.new('RGB', (img.width, bar.height + img.height + leg.height))
            out_img.paste(bar, (0, 0))
            out_img.paste(img, (0, bar.height))
            out_img.paste(leg, (0, bar.height + img.height))
            out_img.save(os.path.join(overlays_dir, f'frame_{fi+1:03d}.tiff'),
                         dpi=(out_dpi, out_dpi))

            # Save binary mask TIFF if available
            if fi < len(self.mask_history) and self.mask_history[fi] is not None:
                save_mask_tiff(self.mask_history[fi],
                               os.path.join(overlays_dir, f'mask_{fi+1:03d}.tiff'))

            del img, draw, bar, bd, leg, ld2, out_img, raw, norm
            gc.collect()

            pct = (fi + 1) / N * 100
            self.after(0, self.progress_var.set, pct)
            self.after(0, self._set_status, f'Exporting frame overlays {fi+1}/{N}…')

        has_masks = any(m is not None for m in self.mask_history)
        self.after(0, self._set_status, f'Export complete → {out_dir}')
        self.after(0, self.progress_var.set, 100)
        self.after(0, messagebox.showinfo, 'Export done',
                   f'Report saved to:\n{out_dir}\n\n'
                   f'  adherence_counts.csv\n'
                   f'  timecourse_plot.tiff\n'
                   f'  contact_sheet.tiff\n'
                   f'  frame_overlays/  ({N} images)'
                   + (f'\n  mask_NNN.tiff  ({N} masks)' if has_masks else ''))

    # ── Frame override management ─────────────────────────────────────────────

    def _apply_override(self):
        fi = self.current_fi
        ov = {}
        try:
            v = self._ov_bright.get().strip()
            if v:
                ov['bright_thr'] = float(v)
        except ValueError:
            messagebox.showerror('Invalid', 'bright_thr must be a number.')
            return
        try:
            v = self._ov_dark.get().strip()
            if v:
                ov['dark_thr'] = float(v)
        except ValueError:
            messagebox.showerror('Invalid', 'dark_thr must be a number.')
            return
        if ov:
            self.frame_overrides[fi] = ov
        elif fi in self.frame_overrides:
            del self.frame_overrides[fi]
        self._refresh_override_listbox()
        self._update_preview()

    def _clear_override(self):
        fi = self.current_fi
        self.frame_overrides.pop(fi, None)
        self._ov_bright.set('')
        self._ov_dark.set('')
        self._refresh_override_listbox()
        self._update_preview()

    def _clear_all_overrides(self):
        self.frame_overrides.clear()
        self._ov_bright.set('')
        self._ov_dark.set('')
        self._refresh_override_listbox()
        self._update_preview()

    def _refresh_override_listbox(self):
        self.override_listbox.delete(0, 'end')
        for fi in sorted(self.frame_overrides):
            ov  = self.frame_overrides[fi]
            txt = f'Fr {fi+1}: ' + '  '.join(f'{k}={v:.2f}' for k, v in ov.items())
            self.override_listbox.insert('end', txt)

    # ── ROI drawing & management ──────────────────────────────────────────────

    def _toggle_draw_roi(self):
        if not self.image_files:
            messagebox.showwarning('No images', 'Load a folder first.')
            return
        if self._draw_mode_active:
            self._disconnect_selectors()
            self._draw_mode_active = False
            self.btn_draw_roi.config(text='Draw ROI')
            return
        self._draw_mode_active = True
        self.btn_draw_roi.config(text='Stop Drawing')
        shape = self._roi_shape_var.get()
        try:
            if shape == 'rectangle':
                self._rect_selector = RectangleSelector(
                    self.ax_prev, self._on_rect_done,
                    useblit=False, button=[1],
                    props=dict(edgecolor='#f38ba8', facecolor='none',
                               linestyle='--', linewidth=1.5))
            else:
                self._poly_selector = PolygonSelector(
                    self.ax_prev, self._on_poly_done,
                    props=dict(color='#f38ba8', linestyle='--', linewidth=1.5))
        except TypeError:
            # Older matplotlib without props kwarg
            if shape == 'rectangle':
                self._rect_selector = RectangleSelector(
                    self.ax_prev, self._on_rect_done,
                    useblit=False, button=[1])
            else:
                self._poly_selector = PolygonSelector(
                    self.ax_prev, self._on_poly_done)
        self.canvas_prev.draw()

    def _on_rect_done(self, eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if None in (x0, y0, x1, y1):
            return
        if abs(x1 - x0) < 5 or abs(y1 - y0) < 5:
            return
        self._disconnect_selectors()
        self._draw_mode_active = False
        self.btn_draw_roi.config(text='Draw ROI')
        name = simpledialog.askstring('ROI Name', 'Enter a name for this ROI:',
                                      parent=self)
        if not name:
            name = f'ROI {len(self.rois) + 1}'
        self.rois.append({'name': name, 'shape': 'rectangle',
                          'coords': [x0, y0, x1, y1]})
        if self.loaded_folder:
            save_rois(self.loaded_folder, self.rois)
        self._refresh_roi_listbox()
        self._update_preview()

    def _on_poly_done(self, verts):
        if len(verts) < 3:
            return
        self._disconnect_selectors()
        self._draw_mode_active = False
        self.btn_draw_roi.config(text='Draw ROI')
        name = simpledialog.askstring('ROI Name', 'Enter a name for this ROI:',
                                      parent=self)
        if not name:
            name = f'ROI {len(self.rois) + 1}'
        self.rois.append({'name': name, 'shape': 'polygon',
                          'coords': list(verts)})
        if self.loaded_folder:
            save_rois(self.loaded_folder, self.rois)
        self._refresh_roi_listbox()
        self._update_preview()

    def _disconnect_selectors(self):
        if self._rect_selector is not None:
            try:
                self._rect_selector.set_active(False)
            except Exception:
                pass
            self._rect_selector = None
        if self._poly_selector is not None:
            try:
                self._poly_selector.set_active(False)
            except Exception:
                pass
            self._poly_selector = None

    def _refresh_roi_listbox(self):
        self.roi_listbox.delete(0, 'end')
        for roi in self.rois:
            tag = 'R' if roi['shape'] == 'rectangle' else 'P'
            self.roi_listbox.insert('end', f'[{tag}] {roi["name"]}')

    def _on_roi_select(self, event):
        sel = self.roi_listbox.curselection()
        self.selected_roi_idx = sel[0] if sel else -1
        self._update_preview()

    def _delete_roi(self):
        sel = self.roi_listbox.curselection()
        if not sel:
            return
        del self.rois[sel[0]]
        self.selected_roi_idx = -1
        if self.loaded_folder:
            save_rois(self.loaded_folder, self.rois)
        self._refresh_roi_listbox()
        self._update_preview()

    def _clear_rois(self):
        self.rois = []
        self.selected_roi_idx = -1
        if self.loaded_folder:
            save_rois(self.loaded_folder, self.rois)
        self._refresh_roi_listbox()
        self._update_preview()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _set_status(self, msg):
        self.lbl_status.config(text=msg)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app = AdherenceAnalyzer()
    app.mainloop()
