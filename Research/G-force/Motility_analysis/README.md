# Bacterial Motility Analysis

Analyses trackpy CSV files from bacterial tracking experiments and produces 19 motility metrics per condition, plus cross-condition statistical comparisons.

## Launch the GUI

```bash
cd /home/kumar-perinbam/Research/G-force/Motility_analysis
python3 motility_gui.py
```

## Command-line usage

```bash
python3 analyze_motility.py data.csv
python3 analyze_motility.py file1.csv file2.csv file3.csv --fps 50 --px-per-um 50
python3 analyze_motility.py data.csv --skip-bac-bac --output-dir results/
```

## Input

Trackpy CSV with columns: `y, x, mass, size, ecc, signal, raw_mass, ep, frame, particle`

Default input files:
```
../graphs/C10 16 hours tracking.csv
../graphs/C10 18 hours tracking.csv
../graphs/C10 23 hours tracking.csv
```

## Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--fps` | 50.0 | Camera frame rate |
| `--px-per-um` | 50.0 | Pixels per micron |
| `--min-track-length` | 10 | Minimum track length (frames) |
| `--ep-max` | 5.0 | Max localisation error to keep |
| `--bac-radius` | 0.5 | Bacterium radius (µm) |
| `--output-dir` | `motility_analysis/` | Output folder |

See `ANALYSIS_PARAMETERS.md` for the full parameter reference.

## Output

Results are saved to `motility_analysis/<file_stem>/` — one subfolder per input file, plus cross-file comparisons in the root output folder.
