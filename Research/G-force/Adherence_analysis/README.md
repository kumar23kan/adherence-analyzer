# Cell Tracker — Bacterial Adherence Analyzer

A GUI tool for quantifying cell adhesion kinetics from phase-contrast timelapse microscopy.
Works for any before/after treatment experiment — not limited to gravitational effects.

## Overview

Cells are classified as **adhered** if they remain stationary (within a configurable pixel
distance) for a minimum number of consecutive frames. This tracking-based criterion works
regardless of cell shape — round settling cells and flat spread cells are both captured.

## Method

1. **Background correction** — large Gaussian subtraction removes illumination gradients
2. **Spot detection** — Laplacian of Gaussian (LoG) run twice on the σ-normalised image:
   - Bright LoG → floating cells (bright round halos)
   - Dark LoG (inverted) → all cell bodies (floating and adhered)
   - Duplicate spots within 5 px are merged
3. **H-maxima refinement** *(optional)* — recovers merged blobs at high cell density using
   the h-maxima transform followed by watershed centroid extraction
4. **Frame-to-frame tracking** — nearest-neighbour linking with a configurable max displacement
5. **Adherence classification** — spots tracked at the same location for ≥ N consecutive frames
   = adhered
6. **Shape-aware masking** *(optional)* — Otsu threshold + distance-transform watershed produces
   a per-cell label mask; coverage metrics (area fraction and µm²) are computed per ROI per frame

## Installation

**Python 3.9 or later is required.** `tkinter` is included with standard Python installs.

### Windows

```bat
pip install -r requirements.txt
python adherence_analyzer.py
```

### macOS / Linux

```bash
pip install -r requirements.txt
python3 adherence_analyzer.py
```

> If `pip install` fails on Windows, use a virtual environment:
> ```bat
> python -m venv venv
> venv\Scripts\activate
> pip install -r requirements.txt
> python adherence_analyzer.py
> ```

## Quick Start

1. Click **Load Folder** and select the directory containing TIFF frames
   (filenames must end with a trailing integer, e.g. `image_001.tiff`)
2. Enter a **Condition label** to tag this experiment in the CSV output
3. Adjust parameters in the left panel — the preview updates automatically
4. *(Optional)* Draw one or more **ROIs** to restrict analysis to named zones
5. Click **▶ Run Analysis** to process all frames
6. Click **Export Report** to save results to a chosen folder

## Parameters

| Parameter | Default | Description |
|---|---|---|
| Downsample | 2 | Spatial downsampling factor applied before processing (1 = full res) |
| Background σ | 80 px | Gaussian kernel for illumination correction |
| Bright blob threshold | 1.5 σ | Detection sensitivity for floating cells |
| Dark blob threshold | 0.9 σ | Detection sensitivity for all cell bodies |
| Min cell radius | 4 px | Smallest blob to consider |
| Max cell radius | 18 px | Largest blob to consider |
| µm/pixel | 0.065 | Physical pixel size at the original (full) resolution |
| H-maxima height | 0.0 | Height parameter for h-maxima refinement (0 = disabled); raises sensitivity in dense cultures |
| Max track distance | 8 px | Max displacement (px) between frames to link as the same cell |
| Min frames = adhered | 3 | Consecutive stationary frames required to call a cell adhered |

### Per-frame overrides

Individual frames can have independent `bright_thr` and `dark_thr` values set via the
**Frame Override** panel. Overridden frames are marked ★ in the frame label and listed in the
override box. Overrides are applied at analysis time, not during live preview.

## Regions of Interest (ROI)

- Choose **Rectangle** or **Polygon** from the shape dropdown, then click **Draw ROI**
- Draw on the preview canvas; you will be prompted to name the ROI
- ROIs are saved automatically to `roi.json` next to the TIFFs and reloaded on next open
- Spots **outside** all ROIs are shown as dimmed grey circles and excluded from adhered counts
- The CSV gains `adhered_<name>` / `floating_<name>` columns for each named ROI

## Watershed Masks & Coverage

Enable **Compute masks (watershed)** before running analysis to generate per-cell shape masks:

- Switch the preview to **Mask** mode to see filled cell outlines coloured by status
- Mask TIFFs (`mask_NNN.tiff`) are exported alongside frame overlays
- Coverage columns added to CSV per ROI: `area_frac_<name>` (0–1) and `area_um2_<name>`

## Detection Quality Metric

After analysis, frames that may have unreliable detections are automatically flagged using
three independent checks:

| Flag | Trigger |
|---|---|
| `count-jump` | Total detected cells deviates > 3 σ from the local 7-frame rolling window |
| `low-SNR` | Background std is below the 10th percentile of the full run |
| `abnormal-density` | Total count falls outside the 5th–95th percentile range |

Flagged frames are shown as **orange tick marks** below the frame slider. The preview
shows an amber **⚠ reason** badge on each flagged frame. The CSV gains `quality_flag`
(boolean) and `flag_reason` (comma-separated string) columns.

## Output

| File | Description |
|---|---|
| `adherence_counts.csv` | Per-frame counts with condition label, rolling means, ROI columns, coverage columns, and quality flags |
| `timecourse_plot.tiff` | Two-panel time course: raw + rolling mean, stacked area chart |
| `contact_sheet.tiff` | Key frames (0%, 25%, 50%, 75%, 100%) with cell overlays |
| `frame_overlays/frame_NNN.tiff` | Every frame with green (adhered) and red (floating) circles |
| `frame_overlays/mask_NNN.tiff` | Binary watershed mask per frame (when masks enabled) |

## Visual Guide

| Colour | Meaning |
|---|---|
| Green circle | Adhered cell — stationary for ≥ min_frames consecutive frames |
| Red circle | Floating cell — bright round spot, not yet adhered |
| Grey circle (dim) | Detected spot outside all defined ROIs |
| Orange tick (timeline) | Frame flagged by the detection quality metric |
| Amber ⚠ badge (preview) | Same flag shown on the frame image |

## Notes

- Images must be single-channel (grayscale) TIFF files
- Filenames must contain a trailing integer index for correct frame ordering
- The tool expects one field of view imaged continuously; stage movements between
  acquisitions will break tracks at that point
- The adhered count in late frames may be a conservative undercount once cells have
  spread fully flat (spread flat cells have lower phase-contrast signal than round
  floating cells)
- All pixel coordinates are in the downsampled space (`downsample` factor applied);
  µm/pixel calibration is automatically scaled accordingly
