# Adherence Analyzer

A GUI tool for quantifying cell adhesion kinetics from phase-contrast timelapse microscopy.

## Overview

Cells are classified as **adhered** if they remain stationary (within a configurable pixel distance) for a minimum number of consecutive frames. This tracking-based criterion is more robust than morphology-based approaches because it works regardless of cell shape — round settling cells and flat spread cells are both captured.

## Method

1. **Background correction** — large Gaussian subtraction removes illumination gradients
2. **Spot detection** — Laplacian of Gaussian (LoG) run twice on the σ-normalised image:
   - Bright LoG → floating cells (bright round halos)
   - Dark LoG (inverted) → all cell bodies (both floating and adhered)
   - Duplicate spots within 5 px are merged
3. **Frame-to-frame tracking** — nearest-neighbour linking with a configurable max displacement
4. **Adherence classification** — spots tracked at the same location for ≥ N consecutive frames = adhered

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

> If `pip install` fails on Windows, try running the command prompt as Administrator,
> or use a virtual environment:
> ```bat
> python -m venv venv
> venv\Scripts\activate
> pip install -r requirements.txt
> python adherence_analyzer.py
> ```

1. Click **Load Folder** and select the directory containing TIFF frames (named with a trailing number, e.g. `image_001.tiff`)
2. Adjust parameters in the left panel (preview updates automatically)
3. Click **▶ Run Analysis** to process all frames
4. Click **Export Report** to save results to a chosen folder

## Parameters

| Parameter | Default | Description |
|---|---|---|
| Background σ | 80 px | Gaussian kernel for illumination correction |
| Bright blob threshold | 1.5 σ | Detection sensitivity for floating cells |
| Dark blob threshold | 0.9 σ | Detection sensitivity for all cell bodies |
| Min cell radius | 4 px | Smallest blob to consider |
| Max cell radius | 18 px | Largest blob to consider |
| Max track distance | 8 px | Max displacement (px) between frames to link as same cell |
| Min frames = adhered | 3 | Consecutive stationary frames required to call a cell adhered |

## Output

| File | Description |
|---|---|
| `adherence_counts.csv` | Per-frame counts: frame, time_min, adhered, floating, total_detected, rolling means |
| `timecourse_plot.png` | Two-panel time course: raw + rolling mean, stacked area chart |
| `contact_sheet.png` | Key frames (0%, 25%, 50%, 75%, 100%) with cell overlays |
| `frame_overlays/frame_NNN.png` | Every frame with green (adhered) and red (floating) circles |

## Visual Guide

| Colour | Meaning |
|---|---|
| 🟢 Green circle | Adhered cell — stationary for ≥ min_frames consecutive frames |
| 🔴 Red circle | Floating cell — bright round spot, not yet classified as adhered |

## Notes

- Images must be single-channel (grayscale) TIFF files
- Filenames must contain a trailing integer index for correct frame ordering
- The tool expects one field of view imaged continuously; stage movements between acquisitions will break tracks at that point
- The adhered count in late frames may be a conservative undercount once cells have spread fully flat (spread cells have lower phase-contrast signal than round floating cells)
