# Adherence Analysis — Session Log

---

## Session: 2026-06-29 — Feature Development

### Scope agreed
1. General treatment comparison tool (condition labels, not G-force specific)
2. ROI support — named zones, rect/polygon, multi-ROI, roi.json persistence, per-ROI CSV counts
3. Shape-aware detection + masking (watershed, toggleable circle/mask view, mask TIFFs)
4. Coverage metrics per ROI per frame (area fraction + µm²)
5. Better high-density detection (h-maxima + watershed, per-frame threshold override)
6. Detection quality metric (auto-flag unreliable frames, timeline indicator)

### Implementation log

**Phase 1 — DONE** (condition labels)
- Window/heading renamed to "Cell Tracker"
- "Experiment" panel with condition label text field added to left column
- `condition` column added to CSV output
- Plot titles, stats panel, and per-frame overlay headers include condition label

**Phase 2 — DONE** (ROI support)
- New imports: `json`, `simpledialog`, `skimage.draw`, `skimage.measure`, `RectangleSelector`, `PolygonSelector`
- New functions: `save_rois()`, `load_rois()`, `build_roi_mask()`
- ROI panel added (shape dropdown, Draw ROI button, listbox, Delete/Clear All)
- Interactive drawing on preview canvas: rectangle (click+drag) and polygon (click points, double-click to close)
- ROIs named by user on creation; saved to `roi.json` next to TIFFs; auto-loaded on folder open
- Spots outside ROIs tracked separately, shown as dimmed grey circles in preview and exports
- `run_tracker()` gains `roi_mask` parameter; snaps now 5-tuples `(r, c, consec, tid, in_roi)`
- Per-ROI adhered counts added as `adhered_<name>` / `floating_<name>` columns in CSV
- Both export loops (contact sheet + frame overlays) updated for 5-tuple format

**Phase 3 — DONE** (shape-aware masking)
- New imports: `skimage.segmentation`, `scipy.ndimage.distance_transform_edt`, `matplotlib.path.Path`, `matplotlib.patches.PathPatch`
- New functions: `generate_mask()` (Otsu threshold → disk seeds → distance transform → watershed), `mask_to_contours()` (label → matplotlib PathPatch list), `save_mask_tiff()` (binary TIFF export)
- "Compute masks (watershed)" checkbox added to Detection group
- Third "Mask" radio button added to preview nav bar (disabled until masks computed)
- `run_tracker()` gains `compute_masks` flag; generates mask after spot detection (before `del corr`); returns 3-tuple `(results, track_history, mask_history)`
- `_analysis_done()` stores `mask_history`, enables Mask radio when masks present
- `_update_preview()`: mask view draws filled+outlined PathPatch contours coloured by adhered/floating/out-of-ROI status
- Export saves `mask_NNN.tiff` per frame alongside overlays; export dialog mentions mask count

**Phase 4 — DONE** (coverage metrics)
- `um_per_px` added to DEFAULTS (0.065 µm/px default, slider 0.01–2.0); appears in Detection group
- New function `compute_coverage(label_mask, roi_binary_mask, um_per_px_eff)` → `{area_fraction, area_um2}`
- In `_analysis_done`: after per-ROI counts, iterates `mask_history`; for each ROI builds `roi_binary_mask` via `build_roi_mask([roi])`, calls `compute_coverage`, writes `area_frac_<name>` and `area_um2_<name>` columns to df
- Effective µm/px = user-entered value × `user_ds` (calibration applied in downsampled space)
- NaN written for frames where mask was not computed (when checkbox was off)

**Phase 5 — DONE** (high-density detection fix)
- `skimage.morphology` imported as `skmorphology`
- `hmax_h` added to DEFAULTS (0.0=off, slider 0–5); appears in Detection group
- `detect_spots()` gains `hmax_h` parameter: after LoG deduplication, clips corr to non-negative, runs `skmorphology.h_maxima`, labels connected components, extracts centroids, merges with existing spots via 5-px cKDTree deduplication
- `run_tracker()` gains `overrides` parameter (dict `{frame_idx: {key: val}}`); extracts `hmax_h` from params; applies per-frame bright_thr / dark_thr / hmax_h overrides inside the loop before calling `detect_spots`
- New state: `self.frame_overrides`, `self._ov_bright`, `self._ov_dark`, `self._override_frame_lbl`
- "Frame Override" LabelFrame between Tracking and ROI panels: shows current frame label (★ if overridden), bright_thr / dark_thr entry fields, Apply / Clear / Clear All buttons, listbox of all overridden frames
- `_update_preview` syncs override label and pre-fills entry fields for current frame
- Override dict snapshotted at analysis start (thread-safe); cleared on new folder load

**Phase 6 — DONE** (detection quality metric)
- New standalone function `compute_reliability(results, snr_series, jump_sigma=3.0)` added before `_hstack`
  - Three checks per frame: count-jump (local 7-frame window, z-score), low-SNR (below 10th percentile of series), abnormal-density (outside 5th–95th percentile)
  - Returns list of `{'flagged': bool, 'reasons': [str]}`
- `background_correct()` now returns 3-tuple `(arr, corr, std)` — `std` is the SNR proxy
- All 3 call sites updated: `run_tracker` loop (`frame_snr`), `_update_preview` (`_`), `_start_analysis` (`_`)
- `run_tracker` accumulates `snr_series`, calls `compute_reliability` at end, returns 4-tuple `(results, track_history, mask_history, reliability)`
- `__init__` gains `self.reliability = []` and `self._flagged_positions = []`
- Thin 8-px `tk.Canvas` (`self.tick_canvas`) added at `row=2` below frame slider nav bar; bound to `<Configure>` → `_redraw_ticks`
- New methods `_mark_unreliable_frames()` and `_redraw_ticks()`: draw orange (#fab387) tick marks at flagged frame positions along the timeline
- `_update_preview` shows an amber `⚠ reason, reason` badge (bottom-left, transAxes coords) on any flagged frame, in all view modes
- `_analysis_thread` unpacks 4-tuple; passes `reliability` to `_analysis_done`
- `_analysis_done` stores `self.reliability`, adds `quality_flag` and `flag_reason` columns to CSV df, calls `_mark_unreliable_frames()`

### Folder Analysis

**Purpose:** GUI tool for quantifying cell adhesion kinetics from phase-contrast timelapse microscopy.

**Files (excluding images):**

| File | Lines | Role |
|---|---|---|
| `adherence_analyzer.py` | 891 | Main application — GUI + all analysis logic |
| `requirements.txt` | 6 | Dependencies: numpy, Pillow, scikit-image, scipy, pandas, matplotlib |
| `README.md` | 85 | Usage docs and parameter table |
| `Csv data/` | — | Output directory; only T10 1-hour time point processed so far |

**Dataset:** `C0 (16.04.26)/` — 22 time-point folders (zero hour → 24 hour), ~100–104 TIFF frames each. Only 1 time point analyzed so far.

**Analysis pipeline:**
1. Background correction — Gaussian subtraction (σ=80 px) on 4× downsampled tile
2. Spot detection — dual LoG (bright LoG = floating cells, dark LoG = all bodies), deduplicated within 5 px
3. Frame-to-frame tracking — nearest-neighbour via `cKDTree`
4. Adherence classification — stationary for ≥ 3 consecutive frames = adhered

**Code structure:**
- Headless core: `load_files()`, `background_correct()`, `detect_spots()`, `run_tracker()`
- GUI: `AdherenceAnalyzer(tk.Tk)` — threaded analysis + export, debounced live preview

**Key observation:** 22 time points in dataset, only 1 processed. A batch script wrapping `run_tracker()` would automate the rest.
