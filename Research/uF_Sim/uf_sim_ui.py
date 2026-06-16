"""
Microfluidic Simulation UI
Generates a job config + uploads to Google Drive for Colab computation.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import os
import datetime


# ── Simulation type definitions ────────────────────────────────────────────────

SIMULATION_TYPES = {
    "Pressure Drop": {
        "description": "Compute pressure distribution and drop across channels (Stokes flow).",
        "params": [
            ("Inlet Velocity (m/s)", "0.001"),
            ("Fluid Viscosity (Pa·s)", "0.001"),
            ("Fluid Density (kg/m³)", "1000"),
            ("Outlet Pressure (Pa)", "0"),
        ],
        "solver": "stokes",
    },
    "Flow Mixing": {
        "description": "Species transport and mixing between two inlet streams.",
        "params": [
            ("Inlet Velocity (m/s)", "0.001"),
            ("Fluid Viscosity (Pa·s)", "0.001"),
            ("Fluid Density (kg/m³)", "1000"),
            ("Diffusion Coefficient (m²/s)", "1e-9"),
            ("Inlet 1 Concentration (mol/m³)", "1.0"),
            ("Inlet 2 Concentration (mol/m³)", "0.0"),
        ],
        "solver": "mixing",
    },
    "Particle Tracking": {
        "description": "Lagrangian particle tracking through the channel flow field.",
        "params": [
            ("Inlet Velocity (m/s)", "0.001"),
            ("Fluid Viscosity (Pa·s)", "0.001"),
            ("Fluid Density (kg/m³)", "1000"),
            ("Particle Diameter (µm)", "5"),
            ("Particle Density (kg/m³)", "1050"),
            ("Number of Particles", "100"),
        ],
        "solver": "particle_tracking",
    },
    "Heat Transfer": {
        "description": "Conjugate heat transfer with fluid flow in channels.",
        "params": [
            ("Inlet Velocity (m/s)", "0.001"),
            ("Fluid Viscosity (Pa·s)", "0.001"),
            ("Fluid Density (kg/m³)", "1000"),
            ("Thermal Conductivity (W/m·K)", "0.6"),
            ("Specific Heat Cp (J/kg·K)", "4182"),
            ("Inlet Temperature (°C)", "20"),
            ("Wall Temperature (°C)", "37"),
        ],
        "solver": "heat_transfer",
    },
    "Dean Flow (Curved Channels)": {
        "description": "Secondary Dean vortex flow in curved/spiral microchannels.",
        "params": [
            ("Inlet Velocity (m/s)", "0.01"),
            ("Fluid Viscosity (Pa·s)", "0.001"),
            ("Fluid Density (kg/m³)", "1000"),
            ("Channel Radius of Curvature (mm)", "5"),
        ],
        "solver": "dean_flow",
    },
}

MESH_RESOLUTIONS = ["Coarse", "Medium", "Fine", "Very Fine"]
SOLVERS = ["FEniCSx (Recommended)", "OpenFOAM"]
OUTPUT_FORMATS = ["VTU (ParaView)", "VTK (ParaView)", "OpenFOAM Case"]


# ── Main UI ────────────────────────────────────────────────────────────────────

class MicrofluidicSimUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Microfluidic Simulation Setup")
        self.root.geometry("780x720")
        self.root.resizable(True, True)
        self.root.configure(bg="#f0f4f8")

        self.stl_path = tk.StringVar()
        self.sim_type = tk.StringVar(value=list(SIMULATION_TYPES.keys())[0])
        self.mesh_res = tk.StringVar(value="Medium")
        self.solver_choice = tk.StringVar(value=SOLVERS[0])
        self.output_format = tk.StringVar(value=OUTPUT_FORMATS[0])
        self.drive_upload = tk.BooleanVar(value=False)
        self.param_entries = {}

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 14, "bold"), background="#f0f4f8")
        style.configure("Section.TLabel", font=("Helvetica", 10, "bold"), background="#f0f4f8")
        style.configure("TFrame", background="#f0f4f8")
        style.configure("Card.TFrame", background="#ffffff", relief="flat")
        style.configure("TButton", font=("Helvetica", 10))
        style.configure("Primary.TButton", font=("Helvetica", 11, "bold"))

        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Title ──
        ttk.Label(main, text="Microfluidic CFD Simulation Setup", style="Title.TLabel").pack(anchor="w", pady=(0, 12))

        # ── STL File ──
        self._section(main, "1. Geometry File (.stl)")
        stl_frame = ttk.Frame(main)
        stl_frame.pack(fill=tk.X, pady=(2, 10))
        ttk.Entry(stl_frame, textvariable=self.stl_path, width=55).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(stl_frame, text="Browse…", command=self._browse_stl).pack(side=tk.LEFT)

        # ── Simulation Type ──
        self._section(main, "2. Simulation Type")
        sim_frame = ttk.Frame(main)
        sim_frame.pack(fill=tk.X, pady=(2, 2))
        sim_combo = ttk.Combobox(
            sim_frame,
            textvariable=self.sim_type,
            values=list(SIMULATION_TYPES.keys()),
            state="readonly",
            width=35,
        )
        sim_combo.pack(side=tk.LEFT)
        sim_combo.bind("<<ComboboxSelected>>", self._on_sim_change)

        self.desc_label = ttk.Label(main, text="", foreground="#555", background="#f0f4f8", wraplength=700)
        self.desc_label.pack(anchor="w", pady=(4, 8))

        # ── Parameters ──
        self._section(main, "3. Simulation Parameters")
        self.param_frame = ttk.Frame(main, style="Card.TFrame", padding=10)
        self.param_frame.pack(fill=tk.X, pady=(2, 10))
        self._build_params()

        # ── Mesh & Solver ──
        self._section(main, "4. Mesh & Solver Options")
        opts_frame = ttk.Frame(main)
        opts_frame.pack(fill=tk.X, pady=(2, 10))

        ttk.Label(opts_frame, text="Mesh Resolution:", background="#f0f4f8").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Combobox(opts_frame, textvariable=self.mesh_res, values=MESH_RESOLUTIONS, state="readonly", width=14).grid(row=0, column=1, padx=(0, 24))

        ttk.Label(opts_frame, text="Solver:", background="#f0f4f8").grid(row=0, column=2, sticky="w", padx=(0, 8))
        ttk.Combobox(opts_frame, textvariable=self.solver_choice, values=SOLVERS, state="readonly", width=22).grid(row=0, column=3, padx=(0, 24))

        ttk.Label(opts_frame, text="Output Format:", background="#f0f4f8").grid(row=0, column=4, sticky="w", padx=(0, 8))
        ttk.Combobox(opts_frame, textvariable=self.output_format, values=OUTPUT_FORMATS, state="readonly", width=18).grid(row=0, column=5)

        # ── Google Drive upload toggle ──
        self._section(main, "5. Export & Upload")
        drive_frame = ttk.Frame(main)
        drive_frame.pack(fill=tk.X, pady=(2, 10))
        ttk.Checkbutton(drive_frame, text="Upload job to Google Drive after export (requires Drive credentials)", variable=self.drive_upload).pack(anchor="w")

        # ── Action buttons ──
        sep = ttk.Separator(main, orient="horizontal")
        sep.pack(fill=tk.X, pady=10)

        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Preview Config", command=self._preview_config).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Export Job File", command=self._export_job, style="Primary.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_frame, text="Clear", command=self._clear).pack(side=tk.RIGHT)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(main, textvariable=self.status_var, foreground="#444", background="#dce8f5", relief="sunken", anchor="w", padding=(6, 2))
        status_bar.pack(fill=tk.X, pady=(10, 0))

        self._update_description()

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", pady=(6, 0))

    def _browse_stl(self):
        path = filedialog.askopenfilename(
            title="Select STL file",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        if path:
            self.stl_path.set(path)
            self.status_var.set(f"Loaded: {os.path.basename(path)}")

    def _on_sim_change(self, _event=None):
        self._build_params()
        self._update_description()

    def _update_description(self):
        sim = SIMULATION_TYPES.get(self.sim_type.get(), {})
        self.desc_label.config(text=sim.get("description", ""))

    def _build_params(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_entries.clear()

        params = SIMULATION_TYPES[self.sim_type.get()]["params"]
        cols = 2
        for i, (label, default) in enumerate(params):
            row, col = divmod(i, cols)
            frm = ttk.Frame(self.param_frame)
            frm.grid(row=row, column=col, sticky="w", padx=12, pady=4)
            ttk.Label(frm, text=label, background="#ffffff", width=28, anchor="w").pack(side=tk.LEFT)
            entry = ttk.Entry(frm, width=14)
            entry.insert(0, default)
            entry.pack(side=tk.LEFT)
            self.param_entries[label] = entry

    def _collect_config(self):
        stl = self.stl_path.get().strip()
        if not stl:
            messagebox.showwarning("Missing File", "Please select an STL geometry file.")
            return None
        if not os.path.isfile(stl):
            messagebox.showerror("File Not Found", f"STL file not found:\n{stl}")
            return None

        params = {}
        for label, entry in self.param_entries.items():
            val = entry.get().strip()
            try:
                params[label] = float(val)
            except ValueError:
                params[label] = val

        sim_key = self.sim_type.get()
        config = {
            "job_id": datetime.datetime.now().strftime("uFSim_%Y%m%d_%H%M%S"),
            "stl_file": stl,
            "stl_filename": os.path.basename(stl),
            "simulation_type": sim_key,
            "solver_backend": SIMULATION_TYPES[sim_key]["solver"],
            "mesh_resolution": self.mesh_res.get(),
            "solver": self.solver_choice.get(),
            "output_format": self.output_format.get(),
            "parameters": params,
            "upload_to_drive": self.drive_upload.get(),
            "created_at": datetime.datetime.now().isoformat(),
        }
        return config

    def _preview_config(self):
        config = self._collect_config()
        if config is None:
            return
        win = tk.Toplevel(self.root)
        win.title("Config Preview")
        win.geometry("560x420")
        text = tk.Text(win, wrap=tk.WORD, font=("Courier", 10), padx=8, pady=8)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, json.dumps(config, indent=2))
        text.config(state=tk.DISABLED)

    def _export_job(self):
        config = self._collect_config()
        if config is None:
            return

        save_dir = os.path.dirname(self.stl_path.get()) or os.path.expanduser("~")
        job_path = filedialog.asksaveasfilename(
            initialdir=save_dir,
            initialfile=config["job_id"] + ".json",
            defaultextension=".json",
            filetypes=[("JSON job file", "*.json")],
            title="Save Job File",
        )
        if not job_path:
            return

        with open(job_path, "w") as f:
            json.dump(config, f, indent=2)

        self.status_var.set(f"Job exported: {os.path.basename(job_path)}")
        msg = f"Job file saved:\n{job_path}\n\nNext step: open the Colab notebook and load this file."
        if self.drive_upload.get():
            msg += "\n\nDrive upload: coming in next step (requires credentials)."
        messagebox.showinfo("Export Complete", msg)

    def _clear(self):
        self.stl_path.set("")
        self.sim_type.set(list(SIMULATION_TYPES.keys())[0])
        self.mesh_res.set("Medium")
        self.solver_choice.set(SOLVERS[0])
        self.output_format.set(OUTPUT_FORMATS[0])
        self.drive_upload.set(False)
        self._build_params()
        self._update_description()
        self.status_var.set("Cleared.")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    app = MicrofluidicSimUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
