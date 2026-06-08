#!/usr/bin/env python3
"""
GUI front-end for analyze_motility.py
--------------------------------------
• Add / remove trackpy CSV files
• Configure all analysis parameters
• Choose output folder
• Live log with colour-coded output
• Indeterminate progress bar while running
• Stop button to terminate the subprocess
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import subprocess
import sys
import queue
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ANALYZER   = SCRIPT_DIR / 'analyze_motility.py'


# ─────────────────────────────────────────────────────────────────────────────
# Main application window
# ─────────────────────────────────────────────────────────────────────────────

class MotilityGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Bacterial Motility Analyzer')
        self.geometry('1150x780')
        self.minsize(900, 620)

        self._apply_style()

        self._files: list[str] = []
        self._running  = False
        self._process  = None
        self._log_q: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_log()          # start 80-ms polling loop for log output

    # ── Styling ───────────────────────────────────────────────────────────────
    def _apply_style(self):
        s = ttk.Style(self)
        try:
            s.theme_use('clam')
        except tk.TclError:
            pass

        s.configure('Run.TButton',
                    font=('TkDefaultFont', 10, 'bold'),
                    foreground='white', background='#2e7d32', padding=6)
        s.map('Run.TButton',
              background=[('active', '#388e3c'), ('disabled', '#9e9e9e')])

        s.configure('Stop.TButton',
                    font=('TkDefaultFont', 10),
                    foreground='white', background='#c62828', padding=6)
        s.map('Stop.TButton',
              background=[('active', '#d32f2f'), ('disabled', '#9e9e9e')])

    # ── UI skeleton ───────────────────────────────────────────────────────────
    def _build_ui(self):
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left  = ttk.Frame(pw, padding=4)
        right = ttk.Frame(pw, padding=4)
        pw.add(left,  weight=1)
        pw.add(right, weight=3)

        self._build_file_panel(left)
        self._build_right_panel(right)

    # ── Left: file list ───────────────────────────────────────────────────────
    def _build_file_panel(self, parent):
        ttk.Label(parent, text='Input CSV Files',
                  font=('TkDefaultFont', 10, 'bold')).pack(anchor='w', pady=(0, 4))

        # Listbox
        lf = ttk.Frame(parent)
        lf.pack(fill=tk.BOTH, expand=True)

        sb_y = ttk.Scrollbar(lf, orient=tk.VERTICAL)
        sb_x = ttk.Scrollbar(lf, orient=tk.HORIZONTAL)
        self.lb = tk.Listbox(
            lf, selectmode=tk.EXTENDED,
            yscrollcommand=sb_y.set, xscrollcommand=sb_x.set,
            font=('TkFixedFont', 9), bg='white',
            relief='flat', highlightthickness=1, highlightcolor='#1976d2',
        )
        sb_y.config(command=self.lb.yview)
        sb_x.config(command=self.lb.xview)
        sb_y.pack(side=tk.RIGHT,  fill=tk.Y)
        sb_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.lb.pack(fill=tk.BOTH, expand=True)
        self.lb.bind('<<ListboxSelect>>', self._on_select)

        # Full-path label (updates on selection)
        self.path_var = tk.StringVar(value='')
        ttk.Label(parent, textvariable=self.path_var,
                  foreground='grey', font=('TkDefaultFont', 8),
                  wraplength=220, justify='left').pack(anchor='w', pady=(3, 0))

        # File count
        self.count_var = tk.StringVar(value='0 files')
        ttk.Label(parent, textvariable=self.count_var,
                  foreground='grey').pack(anchor='e', pady=(2, 6))

        # Buttons
        bf = ttk.Frame(parent)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text='＋  Add Files',
                   command=self._add_files).pack(fill=tk.X, pady=(0, 3))
        ttk.Button(bf, text='－  Remove Selected',
                   command=self._remove_files).pack(fill=tk.X, pady=(0, 3))
        ttk.Button(bf, text='✕  Clear All',
                   command=self._clear_files).pack(fill=tk.X)

    # ── Right: params + output + log ──────────────────────────────────────────
    def _build_right_panel(self, parent):
        # Parameters notebook
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.X, pady=(0, 6))

        basic = ttk.Frame(nb, padding=(14, 10))
        adv   = ttk.Frame(nb, padding=(14, 10))
        nb.add(basic, text='  Basic Parameters  ')
        nb.add(adv,   text='  Advanced Parameters  ')
        self._build_basic_tab(basic)
        self._build_adv_tab(adv)

        # Output folder
        olf = ttk.LabelFrame(parent, text=' Output Folder ', padding=(8, 5))
        olf.pack(fill=tk.X, pady=(0, 6))

        self.out_var = tk.StringVar(value=str(SCRIPT_DIR / 'motility_analysis'))
        ttk.Entry(olf, textvariable=self.out_var,
                  font=('TkFixedFont', 9)).pack(side=tk.LEFT, fill=tk.X,
                                                 expand=True, padx=(0, 6))
        ttk.Button(olf, text='Browse…', command=self._browse_out).pack(side=tk.LEFT)

        # Log
        llf = ttk.LabelFrame(parent, text=' Analysis Log ', padding=(5, 5))
        llf.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.log = scrolledtext.ScrolledText(
            llf, font=('TkFixedFont', 9),
            bg='#1e1e1e', fg='#d4d4d4', insertbackground='white',
            state=tk.DISABLED, wrap=tk.NONE, height=16,
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_config('err', foreground='#f48771')
        self.log.tag_config('ok',  foreground='#89d185')
        self.log.tag_config('hdr', foreground='#569cd6')
        self.log.tag_config('dim', foreground='#888888')

        # Progress bar
        self.progress = ttk.Progressbar(parent, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=(0, 4))

        # Action buttons
        bf = ttk.Frame(parent)
        bf.pack(fill=tk.X)

        self.run_btn = ttk.Button(bf, text='▶   Run Analysis',
                                   command=self._run, style='Run.TButton')
        self.run_btn.pack(side=tk.LEFT)

        self.stop_btn = ttk.Button(bf, text='■   Stop',
                                    command=self._stop, state=tk.DISABLED,
                                    style='Stop.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Button(bf, text='Open Output Folder',
                   command=self._open_out).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(bf, text='Clear Log',
                   command=self._clear_log).pack(side=tk.RIGHT)

        # Status bar
        self.status_var = tk.StringVar(value='Ready.')
        ttk.Label(parent, textvariable=self.status_var,
                  relief='sunken', anchor='w',
                  padding=(6, 2)).pack(fill=tk.X, pady=(4, 0))

    # ── Parameter tabs ────────────────────────────────────────────────────────
    def _param_row(self, parent, row, label, vtype, default, hint=''):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky='w', padx=(0, 12), pady=4)
        var = vtype(value=default)
        ttk.Entry(parent, textvariable=var, width=12).grid(
            row=row, column=1, sticky='w', pady=4)
        if hint:
            ttk.Label(parent, text=hint, foreground='grey',
                      font=('TkDefaultFont', 8)).grid(
                row=row, column=2, sticky='w', padx=(10, 0))
        return var

    def _build_basic_tab(self, p):
        p.columnconfigure(2, weight=1)
        self.fps_var     = self._param_row(p, 0, 'FPS',
                                            tk.DoubleVar, 50.0,
                                            'Camera frame rate (frames per second)')
        self.px_var      = self._param_row(p, 1, 'Pixels / µm',
                                            tk.DoubleVar, 50.0,
                                            'Spatial calibration  (50 px/µm → 1 px = 20 nm)')
        self.min_trk_var = self._param_row(p, 2, 'Min track length',
                                            tk.IntVar,    10,
                                            'Frames — tracks shorter than this are excluded')
        self.ep_var      = self._param_row(p, 3, 'ep max',
                                            tk.DoubleVar, 5.0,
                                            'Max localisation error |ep| to keep (trackpy units)')
        self.bac_r_var   = self._param_row(p, 4, 'Bacterium radius (µm)',
                                            tk.DoubleVar, 0.5,
                                            'Half-width used for collision distance thresholds')

    def _build_adv_tab(self, p):
        p.columnconfigure(2, weight=1)
        self.tumble_var  = self._param_row(p, 0, 'Tumble angle (°)',
                                            tk.DoubleVar, 90.0,
                                            'Direction change above this → classified as tumble')
        self.max_lag_var = self._param_row(p, 1, 'Max lag (frames)',
                                            tk.IntVar,    20,
                                            'Maximum lag computed for MSD and autocorrelation')
        self.stat_sp_var = self._param_row(p, 2, 'Stationary speed (µm/s)',
                                            tk.DoubleVar, 0.5,
                                            'Steps slower than this → classified as stationary')

        ttk.Separator(p, orient='horizontal').grid(
            row=3, column=0, columnspan=3, sticky='ew', pady=8)
        ttk.Label(p, text='Arena boundaries  (leave empty → auto-detect from data)',
                  foreground='grey',
                  font=('TkDefaultFont', 8)).grid(row=4, column=0, columnspan=3, sticky='w')

        self.bnd_xlo_var = self._param_row(p, 5, 'Boundary X lo (µm)', tk.StringVar, '', '')
        self.bnd_xhi_var = self._param_row(p, 6, 'Boundary X hi (µm)', tk.StringVar, '', '')
        self.bnd_ylo_var = self._param_row(p, 7, 'Boundary Y lo (µm)', tk.StringVar, '', '')
        self.bnd_yhi_var = self._param_row(p, 8, 'Boundary Y hi (µm)', tk.StringVar, '', '')

        ttk.Separator(p, orient='horizontal').grid(
            row=9, column=0, columnspan=3, sticky='ew', pady=8)
        self.skip_bac_var = tk.BooleanVar(value=False)
        self.skip_gr_var  = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            p, text='Skip bacteria–bacteria collisions  (recommended for >10 000 bacteria/frame)',
            variable=self.skip_bac_var,
        ).grid(row=10, column=0, columnspan=3, sticky='w')
        ttk.Checkbutton(
            p, text='Skip pair correlation g(r)  (slow for large datasets)',
            variable=self.skip_gr_var,
        ).grid(row=11, column=0, columnspan=3, sticky='w')

    # ── File management ───────────────────────────────────────────────────────
    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title='Select trackpy CSV files',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
        )
        existing = set(self._files)
        for p in paths:
            if p not in existing:
                self._files.append(p)
                self.lb.insert(tk.END, Path(p).name)
                existing.add(p)
        self._update_count()
        if paths:
            self.lb.see(tk.END)

    def _remove_files(self):
        for i in reversed(self.lb.curselection()):
            self.lb.delete(i)
            del self._files[i]
        self._update_count()
        self.path_var.set('')

    def _clear_files(self):
        self.lb.delete(0, tk.END)
        self._files.clear()
        self._update_count()
        self.path_var.set('')

    def _update_count(self):
        n = len(self._files)
        self.count_var.set(f'{n} file{"s" if n != 1 else ""}')

    def _on_select(self, _event=None):
        sel = self.lb.curselection()
        if sel:
            self.path_var.set(self._files[sel[-1]])
        else:
            self.path_var.set('')

    def _browse_out(self):
        d = filedialog.askdirectory(
            title='Select output folder',
            initialdir=self.out_var.get() or '.',
        )
        if d:
            self.out_var.set(d)

    def _open_out(self):
        import os, subprocess as sp
        p = self.out_var.get() or '.'
        Path(p).mkdir(parents=True, exist_ok=True)
        try:
            sp.Popen(['xdg-open', p])
        except FileNotFoundError:
            messagebox.showinfo('Output folder', p)

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _log_write(self, text, tag=None):
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, text, tag)
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _clear_log(self):
        self.log.config(state=tk.NORMAL)
        self.log.delete('1.0', tk.END)
        self.log.config(state=tk.DISABLED)

    def _poll_log(self):
        while not self._log_q.empty():
            text, tag = self._log_q.get_nowait()
            self._log_write(text, tag)
        self.after(80, self._poll_log)

    # ── Build subprocess command ──────────────────────────────────────────────
    def _build_cmd(self) -> list[str]:
        if not self._files:
            raise ValueError('Add at least one CSV file before running.')
        if not ANALYZER.exists():
            raise FileNotFoundError(f'Analyzer script not found:\n{ANALYZER}')

        cmd = [
            sys.executable, str(ANALYZER),
            *self._files,
            '--fps',              str(self.fps_var.get()),
            '--px-per-um',        str(self.px_var.get()),
            '--min-track-length', str(self.min_trk_var.get()),
            '--ep-max',           str(self.ep_var.get()),
            '--bac-radius',       str(self.bac_r_var.get()),
            '--tumble-angle',     str(self.tumble_var.get()),
            '--max-lag',          str(self.max_lag_var.get()),
            '--stationary-speed', str(self.stat_sp_var.get()),
            '--output-dir',       self.out_var.get() or 'motility_analysis',
        ]
        if self.skip_bac_var.get():
            cmd.append('--skip-bac-bac')
        if self.skip_gr_var.get():
            cmd.append('--skip-gr')
        for val, flag in [
            (self.bnd_xlo_var.get(), '--boundary-x-lo'),
            (self.bnd_xhi_var.get(), '--boundary-x-hi'),
            (self.bnd_ylo_var.get(), '--boundary-y-lo'),
            (self.bnd_yhi_var.get(), '--boundary-y-hi'),
        ]:
            if val.strip():
                cmd += [flag, val.strip()]
        return cmd

    # ── Run / stop ────────────────────────────────────────────────────────────
    def _run(self):
        if self._running:
            return
        try:
            cmd = self._build_cmd()
        except (ValueError, FileNotFoundError) as e:
            messagebox.showwarning('Cannot start', str(e))
            return

        self._running = True
        self.run_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set('Running analysis…')
        self._clear_log()

        n = len(self._files)
        self._log_write(
            f'Files : {n} CSV file{"s" if n != 1 else ""}\n'
            f'Output: {self.out_var.get()}\n'
            f'{'─' * 60}\n\n', 'hdr',
        )

        self.progress.start(12)
        threading.Thread(target=self._worker, args=(cmd,), daemon=True).start()

    def _worker(self, cmd: list[str]):
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in self._process.stdout:
                tag = self._classify_line(line)
                self._log_q.put((line, tag))
            self._process.wait()
            rc = self._process.returncode
            if rc == 0:
                self._log_q.put(('\n✓  Analysis completed successfully.\n', 'ok'))
                msg = f'Done — results saved to:  {self.out_var.get()}'
            else:
                self._log_q.put((f'\n✗  Process exited with code {rc}\n', 'err'))
                msg = f'Failed  (exit code {rc})'
        except Exception as exc:
            self._log_q.put((f'\n✗  {exc}\n', 'err'))
            msg = 'Error — see log'
        finally:
            self._running  = False
            self._process  = None
            self.after(0, lambda m=msg: self._on_done(m))

    @staticmethod
    def _classify_line(line: str) -> str | None:
        lo = line.lower()
        if any(k in lo for k in ('error', 'traceback', 'exception', '✗')):
            return 'err'
        if line.startswith('==') or 'SUMMARY' in line or line.strip().startswith('✓'):
            return 'ok'
        if line.strip().startswith('[') or line.strip().startswith('Files'):
            return 'hdr'
        if line.strip().startswith('  ') and any(
            k in lo for k in ('mean', 'median', 'drift', 'frac', 'α', 'tau')
        ):
            return 'dim'
        return None

    def _on_done(self, msg: str):
        self.progress.stop()
        self.run_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set(msg)

    def _stop(self):
        if self._process and self._running:
            self._process.terminate()
            self._log_q.put(('\n⚠  Stopped by user.\n', 'err'))
            self.status_var.set('Stopped.')


# ─────────────────────────────────────────────────────────────────────────────

def main():
    app = MotilityGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
