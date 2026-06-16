# Microfluidic CFD Simulation

A two-part toolchain for running computational fluid dynamics (CFD) simulations on microfluidic channel geometries exported from Fusion 360.

---

## Overview

| Component | File | Purpose |
|---|---|---|
| Desktop UI | `uf_sim_ui.py` | Set parameters, choose simulation, export job config |
| Colab notebook | `microfluidic_cfd.ipynb` | Mesh, solve, and export results for ParaView |

The workflow is:
1. Export your microfluidic channel geometry from Fusion 360 as **STL**
2. Open the **desktop UI** to configure your simulation and export a job `.json`
3. Open the **Colab notebook**, upload the `.json` and `.stl`, and run the simulation
4. Download the results and open them in **ParaView**

---

## Requirements

### Local machine

- Python 3.9 or later
- `tkinter` (included with most Python installs)

```bash
# No additional pip packages needed for the UI
python3 uf_sim_ui.py
```

### Google Colab

Dependencies are installed automatically by Cell 2 of the notebook:

- [FEniCSx](https://fenicsproject.org/) — finite element solver (via FEM-on-Colab)
- [Gmsh](https://gmsh.info/) — 3D mesh generation from STL
- `meshio`, `h5py`, `scipy` — mesh I/O and particle integration

---

## Step-by-step Instructions

### 1. Export STL from Fusion 360

1. Open your `.f3d` file in Fusion 360
2. Right-click the body/component in the browser → **Save As Mesh**
3. Format: **STL**, Units: **mm**, Refinement: High
4. Save the `.stl` file

---

### 2. Configure the simulation (Desktop UI)

```bash
python3 uf_sim_ui.py
```

| Section | What to do |
|---|---|
| **Geometry File** | Browse and select your `.stl` file |
| **Simulation Type** | Choose from the dropdown |
| **Parameters** | Edit the pre-filled values |
| **Mesh Resolution** | Coarse / Medium / Fine / Very Fine |
| **Output Format** | VTU or XDMF (both open in ParaView) |
| **Export Job File** | Click to save a `.json` config file |

The exported `.json` bundles all settings and is passed to Colab.

#### Simulation types

| Type | Physics solved |
|---|---|
| **Pressure Drop** | Stokes flow — velocity + pressure field, pressure drop value |
| **Flow Mixing** | Advection-diffusion — species concentration, mixing efficiency |
| **Particle Tracking** | Lagrangian — particle trajectories through the flow field |
| **Heat Transfer** | Energy equation — temperature distribution |
| **Dean Flow** | Stokes approximation for curved/spiral channels |

---

### 3. Run the simulation (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `microfluidic_cfd.ipynb` via **File → Upload notebook**
3. Run the cells in order:

| Cell | Action | Notes |
|---|---|---|
| **Cell 2** | Install FEniCSx | Run once per session — takes ~5-10 min |
| **Cell 3** | Imports | Confirm versions print without error |
| **Cell 4** | Mount Drive + upload job | Upload your `.json` when prompted |
| **Cell 5** | Load config + upload STL | Upload your `.stl` when prompted |
| **Cell 6** | Generate mesh | Gmsh builds tetrahedral mesh from STL |
| **Cell 7-10** | Solver definitions | Functions are defined, not run yet |
| **Cell 11** | Run simulation | Automatically calls the right solver |
| **Cell 12** | Download results | Downloads a `.zip` of all output files |

> **Tip:** Enable GPU runtime in Colab for faster mesh generation:  
> Runtime → Change runtime type → T4 GPU

---

### 4. Visualise in ParaView

1. Download and install [ParaView](https://www.paraview.org/download/)
2. Extract the downloaded `.zip`
3. Open ParaView → **File → Open**
4. Select the output file for your simulation:

| Simulation | File to open |
|---|---|
| Pressure Drop | `pressure.xdmf`, `velocity.xdmf` |
| Flow Mixing | `concentration.xdmf` |
| Particle Tracking | `particle_tracks.vtu` |
| Heat Transfer | `temperature.xdmf` |

5. Click **Apply** in the Properties panel
6. Use the dropdown at the top toolbar to colour by field:
   - `pressure`, `velocity`, `concentration`, `temperature`
7. For particle tracks: add a **Tube** filter to give the lines width

---

## Boundary Conditions

The mesh builder auto-detects boundaries from the STL bounding box:

| Boundary | Condition |
|---|---|
| **Inlet** | Face at the minimum extent of the longest axis — uniform inlet velocity |
| **Outlet** | Face at the maximum extent — zero pressure (outflow) |
| **Walls** | All remaining surfaces — no-slip |

If auto-detection fails for your geometry (e.g. channels not axis-aligned), edit the `build_mesh()` function in Cell 6 of the notebook and manually assign physical group tags.

---

## File Structure

```
Research/uF_Sim/
├── uf_sim_ui.py            # Desktop parameter UI
├── microfluidic_cfd.ipynb  # Google Colab simulation notebook
├── generate_notebook.py    # Script that regenerates the notebook
└── README.md               # This file
```

---

## Troubleshooting

**FEniCSx installation fails in Colab**  
FEM-on-Colab may be temporarily unavailable. The notebook falls back to `condacolab` automatically — this will restart the runtime. Re-run from Cell 3 after the restart.

**Mesh generation fails with "no volume"**  
The STL may have open surfaces or self-intersections. In Fusion 360:
- Use **Inspect → Check** to find issues
- Re-export with **Watertight** body only

**Solver diverges or gives zero velocity**  
Check the inlet direction. The notebook assumes flow along the longest axis (+Z by default). If your channel is oriented differently, adjust the inlet velocity vector in `solve_stokes()` Cell 7:
```python
u_in = Constant(domain, PETSc.ScalarType((inlet_vel, 0.0, 0.0)))  # X-axis flow
```

**ParaView shows empty scene**  
Open the `.xdmf` file (not the `.h5`). Both files must be in the same folder.
