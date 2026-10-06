# -*- coding: utf-8 -*-
"""
eqlsite/gui.py
Interfaccia grafica (Tkinter + matplotlib) per l'analisi lineare equivalente.

Schede:
  1. Profilo      - stratigrafia, falda, curve di degrado, suddivisione strati
  2. Moto sismico - caricamento / generazione / scalatura dell'accelerogramma
  3. Analisi      - punto di input (within/outcrop), punti di output, parametri
  4. Risultati    - grafici, tabella per strato, esportazione
  5. Curve        - anteprima G/Gmax e D(gamma) degli strati

Avvio:  python run_gui.py
"""
from __future__ import annotations

import copy
import os
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from . import __version__, plots
from .core import (G_ACC, EquivalentLinearAnalysis, EQLResult, Location, Motion,
                   response_spectrum)
from .project import (CURVE_LABELS, CURVE_TYPES, default_curve, default_halfspace,
                      default_layer, default_specs, demo_specs, export_profile_csv,
                      import_profile_csv, load_project, profile_from_specs,
                      save_project, subdivide_specs, validate_specs)

NO_WATER = 1e9


# =============================================================================
#  Widget di utilita'
# =============================================================================
class PlotFrame(ttk.Frame):
    """Figura matplotlib incorporata con barra degli strumenti."""

    def __init__(self, master, figsize=(6, 4), toolbar=True):
        super().__init__(master)
        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        if toolbar:
            tb = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
            tb.update()
            tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def draw(self):
        self.canvas.draw_idle()


def make_tree(parent, columns, headings, widths, height=8, stretch_col=None, selectmode="browse"):
    """Treeview con barre di scorrimento. Ritorna (frame, tree)."""
    fr = ttk.Frame(parent)
    tree = ttk.Treeview(fr, columns=columns, show="headings", height=height,
                        selectmode=selectmode)
    for c, h, w in zip(columns, headings, widths):
        tree.heading(c, text=h)
        tree.column(c, width=w, minwidth=30, anchor="center",
                    stretch=(c == stretch_col))
    ys = ttk.Scrollbar(fr, orient="vertical", command=tree.yview)
    xs = ttk.Scrollbar(fr, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
    tree.grid(row=0, column=0, sticky="nsew")
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")
    fr.rowconfigure(0, weight=1)
    fr.columnconfigure(0, weight=1)
    return fr, tree


def parse_float(txt: str, name: str, minimum: Optional[float] = None,
                maximum: Optional[float] = None, strict_min: bool = False) -> float:
    try:
        v = float(str(txt).strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"'{name}': valore numerico non valido ({txt!r}).")
    if minimum is not None and (v <= minimum if strict_min else v < minimum):
        raise ValueError(f"'{name}' deve essere {'>' if strict_min else '>='} {minimum:g}.")
    if maximum is not None and v > maximum:
        raise ValueError(f"'{name}' deve essere <= {maximum:g}.")
    return v


# =============================================================================
#  Finestre di dialogo
# =============================================================================
class ParamDialog(tk.Toplevel):
    """Dialogo generico con N campi numerici. result = lista di stringhe o None."""

    def __init__(self, master, title, fields, note=""):
        super().__init__(master)
        self.title(title)
        self.transient(master)
        self.resizable(False, False)
        self.result = None
        self.vars = []
        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)
        if note:
            ttk.Label(body, text=note, wraplength=360, justify="left").grid(
                row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        for i, (lab, default) in enumerate(fields):
            ttk.Label(body, text=lab).grid(row=i + 1, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=str(default))
            ttk.Entry(body, textvariable=v, width=14).grid(row=i + 1, column=1, padx=8)
            self.vars.append(v)
        bt = ttk.Frame(body)
        bt.grid(row=len(fields) + 1, column=0, columnspan=2, pady=(12, 0), sticky="e")
        ttk.Button(bt, text="OK", command=self._ok).pack(side="left", padx=4)
        ttk.Button(bt, text="Annulla", command=self.destroy).pack(side="left")
        self.bind("<Return>", lambda e: self._ok())
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()
        self.wait_window()

    def _ok(self):
        self.result = [v.get() for v in self.vars]
        self.destroy()


class LayerDialog(tk.Toplevel):
    """Modifica completa di uno strato (dati geotecnici + curve)."""

    def __init__(self, master, spec: dict, index: int, is_last: bool):
        super().__init__(master)
        self.title(f"Strato {index + 1}" + (" (semispazio)" if is_last else ""))
        self.transient(master)
        self.resizable(False, False)
        self.result: Optional[dict] = None
        self.spec = copy.deepcopy(spec)
        self.is_last = is_last
        c = self.spec.get("curve", default_curve("linear"))

        body = ttk.Frame(self, padding=12)
        body.pack(fill="both", expand=True)

        gen = ttk.LabelFrame(body, text="Dati dello strato", padding=8)
        gen.pack(fill="x")
        self.v_name = tk.StringVar(value=self.spec.get("name", ""))
        self.v_h = tk.StringVar(value="inf" if is_last else f"{self.spec['thickness']:g}")
        self.v_vs = tk.StringVar(value=f"{self.spec['vs']:g}")
        self.v_uw = tk.StringVar(value=f"{self.spec['unit_weight']:g}")
        rows = [("Nome", self.v_name, None), ("Spessore h [m]", self.v_h, "h"),
                ("Vs [m/s]", self.v_vs, None), ("Peso di volume [kN/m3]", self.v_uw, None)]
        for i, (lab, var, tag) in enumerate(rows):
            ttk.Label(gen, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            e = ttk.Entry(gen, textvariable=var, width=18)
            e.grid(row=i, column=1, padx=8)
            if tag == "h" and is_last:
                e.state(["disabled"])

        cv = ttk.LabelFrame(body, text="Comportamento non lineare", padding=8)
        cv.pack(fill="x", pady=(8, 0))
        ttk.Label(cv, text="Tipo di curva").grid(row=0, column=0, sticky="w")
        self.v_type = tk.StringVar(value=CURVE_LABELS[c.get("type", "linear")])
        self.cb = ttk.Combobox(cv, textvariable=self.v_type, state="readonly", width=22,
                               values=[CURVE_LABELS[t] for t in CURVE_TYPES])
        if is_last:
            self.v_type.set(CURVE_LABELS["linear"])
            self.cb.state(["disabled"])
        self.cb.grid(row=0, column=1, padx=8, pady=2)
        self.cb.bind("<<ComboboxSelected>>", lambda e: self._show_panel())

        self.panel_host = ttk.Frame(cv)
        self.panel_host.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.panels = {}
        self._build_panels(c)
        self._show_panel()

        bt = ttk.Frame(body)
        bt.pack(fill="x", pady=(10, 0))
        ttk.Button(bt, text="OK", command=self._ok).pack(side="right", padx=4)
        ttk.Button(bt, text="Annulla", command=self.destroy).pack(side="right")
        self.bind("<Escape>", lambda e: self.destroy())
        self.grab_set()
        self.wait_window()

    # ---- pannelli parametri per tipo di curva
    def _grid_fields(self, frame, items):
        vars_ = {}
        for i, (key, lab, val) in enumerate(items):
            ttk.Label(frame, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=f"{val:g}")
            ttk.Entry(frame, textvariable=v, width=12).grid(row=i, column=1, padx=8)
            vars_[key] = v
        return vars_

    def _build_panels(self, c: dict):
        dd = default_curve("darendeli")
        dh = default_curve("hyperbolic")
        dt = default_curve("tabular")
        dar = {**dd, **c} if c.get("type") == "darendeli" else dd
        hyp = {**dh, **c} if c.get("type") == "hyperbolic" else dh
        tab = c if c.get("type") == "tabular" else dt

        f = ttk.Frame(self.panel_host)
        self.panels["darendeli"] = (f, self._grid_fields(f, [
            ("pi", "Indice di plasticita' PI [%]", dar["pi"]),
            ("ocr", "OCR", dar["ocr"]),
            ("k0", "K0 (spinta a riposo)", dar["k0"]),
            ("freq", "Frequenza di carico [Hz]", dar["freq"]),
            ("n_cycles", "N. cicli", dar["n_cycles"])]))
        ttk.Label(f, text="La tensione media efficace e' calcolata dal profilo\n"
                          "(peso di volume + falda).", foreground="gray30").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        f = ttk.Frame(self.panel_host)
        self.panels["hyperbolic"] = (f, self._grid_fields(f, [
            ("gamma_ref_pct", "Deformazione di riferimento [%]", hyp["gamma_ref_pct"]),
            ("d_min_pct", "Smorzamento minimo [%]", hyp["d_min_pct"]),
            ("a", "Esponente a", hyp["a"])]))

        f = ttk.Frame(self.panel_host)
        ttk.Label(f, text="Una riga per punto:  gamma[%]   G/Gmax   D[%]").pack(anchor="w")
        self.txt = tk.Text(f, width=44, height=10, font=("Courier", 9))
        self.txt.pack(fill="both", expand=True)
        rows = zip(tab["gamma_pct"], tab["g_gmax"], tab["damping_pct"])
        self.txt.insert("1.0", "\n".join(f"{g:<10g} {r:<8g} {d:<8g}" for g, r, d in rows))
        ttk.Button(f, text="Carica da CSV...", command=self._load_csv).pack(anchor="w", pady=4)
        self.panels["tabular"] = (f, {})

        f = ttk.Frame(self.panel_host)
        self.panels["linear"] = (f, self._grid_fields(f, [
            ("damping_pct", "Smorzamento D [%]", float(self.spec.get("damping_pct", 2.0)))]))

    def _kind(self) -> str:
        lab = self.v_type.get()
        return next(k for k, v in CURVE_LABELS.items() if v == lab)

    def _show_panel(self):
        for f, _ in self.panels.values():
            f.pack_forget()
        self.panels[self._kind()][0].pack(fill="x")

    def _load_csv(self):
        p = filedialog.askopenfilename(parent=self, title="Curva tabellare (gamma %, G/Gmax, D %)",
                                       filetypes=[("CSV/testo", "*.csv *.txt *.dat"), ("Tutti", "*.*")])
        if not p:
            return
        rows = []
        with open(p, "r", errors="ignore") as fh:
            for ln in fh:
                parts = ln.replace(",", " ").replace(";", " ").split()
                try:
                    v = [float(x) for x in parts[:3]]
                except ValueError:
                    continue
                if len(v) == 3:
                    rows.append(v)
        if not rows:
            messagebox.showerror("Errore", "Nessuna riga a 3 colonne numeriche trovata.", parent=self)
            return
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", "\n".join(f"{g:<10g} {r:<8g} {d:<8g}" for g, r, d in rows))

    # ---- conferma
    def _ok(self):
        try:
            s = copy.deepcopy(self.spec)
            s["name"] = self.v_name.get().strip()
            if not self.is_last:
                s["thickness"] = parse_float(self.v_h.get(), "Spessore", 0, strict_min=True)
            s["vs"] = parse_float(self.v_vs.get(), "Vs", 0, strict_min=True)
            s["unit_weight"] = parse_float(self.v_uw.get(), "Peso di volume", 0, strict_min=True)
            kind = "linear" if self.is_last else self._kind()
            vars_ = self.panels[kind][1]
            if kind == "darendeli":
                c = dict(type="darendeli",
                         pi=parse_float(vars_["pi"].get(), "PI", 0),
                         ocr=parse_float(vars_["ocr"].get(), "OCR", 1),
                         k0=parse_float(vars_["k0"].get(), "K0", 0, strict_min=True),
                         freq=parse_float(vars_["freq"].get(), "Frequenza", 0, strict_min=True),
                         n_cycles=parse_float(vars_["n_cycles"].get(), "N. cicli", 1))
            elif kind == "hyperbolic":
                c = dict(type="hyperbolic",
                         gamma_ref_pct=parse_float(vars_["gamma_ref_pct"].get(), "gamma ref", 0, strict_min=True),
                         d_min_pct=parse_float(vars_["d_min_pct"].get(), "D min", 0, 30),
                         a=parse_float(vars_["a"].get(), "Esponente a", 0, strict_min=True))
            elif kind == "tabular":
                rows = []
                for ln in self.txt.get("1.0", "end").splitlines():
                    parts = ln.replace(",", " ").replace(";", " ").split()
                    if not parts:
                        continue
                    if len(parts) < 3:
                        raise ValueError(f"Riga non valida: {ln!r}")
                    rows.append([float(x) for x in parts[:3]])
                if len(rows) < 2:
                    raise ValueError("Servono almeno 2 punti.")
                rows.sort(key=lambda r: r[0])
                arr = np.array(rows)
                if np.any(np.diff(arr[:, 0]) <= 0) or arr[0, 0] <= 0:
                    raise ValueError("Le deformazioni gamma devono essere > 0 e distinte.")
                if np.any(arr[:, 1] <= 0) or np.any(arr[:, 1] > 1.2):
                    raise ValueError("G/Gmax deve essere compreso fra 0 e 1.")
                if np.any(arr[:, 2] < 0) or np.any(arr[:, 2] >= 50):
                    raise ValueError("D [%] deve essere compreso fra 0 e 50.")
                c = dict(type="tabular", gamma_pct=arr[:, 0].tolist(),
                         g_gmax=arr[:, 1].tolist(), damping_pct=arr[:, 2].tolist())
            else:
                c = dict(type="linear")
                s["damping_pct"] = parse_float(vars_["damping_pct"].get(), "Smorzamento", 0, 50)
            s["curve"] = c
            self.result = s
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Dati non validi", str(e), parent=self)


# =============================================================================
#  Applicazione principale
# =============================================================================
class App(tk.Tk):
    VIEWS = ["Accelerogrammi", "Spettri di risposta", "Funzioni di trasferimento",
             "Spettri di Fourier", "Profili (gamma, G/Gmax, D, tau, PGA)", "Profilo Vs",
             "Convergenza", "Cruscotto riassuntivo"]

    def __init__(self):
        super().__init__()
        self.title("EQL Site Response - analisi lineare equivalente 1D")
        self.geometry("1280x800")
        self.minsize(1000, 650)

        # ---- stato
        self.specs: List[dict] = default_specs()
        self.water_var = tk.StringVar(value="2.0")
        self.use_water = tk.BooleanVar(value=True)
        self.motion: Optional[Motion] = None
        self.result: Optional[EQLResult] = None
        self.project_path: Optional[str] = None
        self._undo_specs: Optional[List[dict]] = None
        self._queue: "queue.Queue" = queue.Queue()
        self._running = False
        self._edit_entry: Optional[ttk.Entry] = None

        # parametri analisi
        self.in_layer = tk.StringVar(value="3")
        self.in_kind = tk.StringVar(value="outcrop")
        self.out_layer = tk.StringVar(value="1")
        self.out_kind = tk.StringVar(value="within")
        self.v_mag = tk.StringVar(value="7.0")
        self.use_rg = tk.BooleanVar(value=False)
        self.v_rg = tk.StringVar(value="0.65")
        self.v_tol = tk.StringVar(value="1.0")
        self.v_maxit = tk.StringVar(value="20")
        self.v_pad = tk.StringVar(value="10")
        self.v_sa_damp = tk.StringVar(value="5")

        style = ttk.Style(self)
        style.configure("Treeview", rowheight=24)
        style.configure("Big.TButton", font=("TkDefaultFont", 11, "bold"), padding=6)

        self._build_menu()
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True)
        self._build_tab_profile()
        self._build_tab_motion()
        self._build_tab_analysis()
        self._build_tab_results()
        self._build_tab_curves()
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        self.status = tk.StringVar(value="Pronto.")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(6, 2)).pack(fill="x", side="bottom")

        self._set_default_analysis_points()
        self._on_profile_changed()
        self.after(100, self._poll_queue)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ------------------------------------------------------------------ menu
    def _build_menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="Nuovo progetto", command=self.new_project)
        f.add_command(label="Apri progetto...", command=self.open_project, accelerator="Ctrl+O")
        f.add_command(label="Salva progetto", command=self.save_project, accelerator="Ctrl+S")
        f.add_command(label="Salva progetto con nome...", command=lambda: self.save_project(True))
        f.add_separator()
        f.add_command(label="Carica esempio (profilo + moto sintetico)", command=self.load_demo)
        f.add_separator()
        f.add_command(label="Esci", command=self._quit)
        m.add_cascade(label="File", menu=f)
        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Within / Outcrop: guida", command=self._help_within)
        h.add_command(label="Informazioni", command=self._about)
        m.add_cascade(label="Aiuto", menu=h)
        self.config(menu=m)
        self.bind("<Control-s>", lambda e: self.save_project())
        self.bind("<Control-o>", lambda e: self.open_project())

    def _help_within(self):
        messagebox.showinfo(
            "Within / Outcrop",
            "Gli strati sono numerati da 1 (superficie) a N (semispazio/bedrock).\n\n"
            "WITHIN: moto alla sommita' dello strato, all'interno della colonna di "
            "terreno (onda ascendente + discendente).\n\n"
            "OUTCROP: moto che si avrebbe alla sommita' dello strato se questa fosse "
            "una superficie libera (2 x onda ascendente). Tipico input: registrazione "
            "su roccia affiorante = OUTCROP sullo strato N.\n\n"
            "Nello strato 1 within e outcrop coincidono.\n\n"
            "L'input e ciascun output possono essere assegnati come within o outcrop "
            "a qualsiasi strato, come nel foglio EERA.")

    def _about(self):
        messagebox.showinfo(
            "Informazioni",
            f"EQL Site Response {__version__}\n\n"
            "Analisi lineare equivalente 1D (onde SH) nel dominio della frequenza,\n"
            "sullo schema di EERA (Bardet et al., 2000) e STRATA (Kottke & Rathje, 2008).\n"
            "Curve di Darendeli (2001).")

    def _quit(self):
        if self._running and not messagebox.askyesno("Analisi in corso", "Uscire comunque?"):
            return
        self.destroy()

    def set_status(self, txt: str):
        self.status.set(txt)
        self.update_idletasks()

    # ============================================================== PROFILO
    COLS = ("n", "name", "ztop", "h", "vs", "uw", "curve", "par", "gmax")

    def _build_tab_profile(self):
        tab = ttk.Frame(self.nb, padding=6)
        self.nb.add(tab, text="1. Profilo")

        top = ttk.Frame(tab)
        top.pack(fill="x", pady=(0, 6))
        ttk.Checkbutton(top, text="Falda presente a profondita' [m]:", variable=self.use_water,
                        command=self._on_profile_changed).pack(side="left")
        e = ttk.Entry(top, textvariable=self.water_var, width=8)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda ev: self._on_profile_changed())
        e.bind("<FocusOut>", lambda ev: self._on_profile_changed())
        ttk.Label(top, text="   Doppio clic su una cella per modificarla "
                            "(Nome, h, Vs, peso di volume; curva -> finestra completa).",
                  foreground="gray30").pack(side="left", padx=10)

        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw)
        pw.add(left, weight=3)

        fr, self.tree = make_tree(
            left, self.COLS,
            ["#", "Nome", "z sommita' [m]", "h [m]", "Vs [m/s]", "g [kN/m3]", "Curva",
             "Parametri", "Gmax [MPa]"],
            [34, 130, 85, 60, 70, 75, 115, 185, 80], height=16, stretch_col="par")
        fr.pack(fill="both", expand=True)
        self.tree.bind("<Double-1>", self._on_tree_double_click)

        bt = ttk.Frame(left)
        bt.pack(fill="x", pady=6)
        for txt, cmd in [("Aggiungi strato", self.add_layer), ("Duplica", self.dup_layer),
                         ("Elimina", self.del_layer), ("Su", lambda: self.move_layer(-1)),
                         ("Giu'", lambda: self.move_layer(1)),
                         ("Modifica...", self.edit_selected)]:
            ttk.Button(bt, text=txt, command=cmd).pack(side="left", padx=2)
        bt2 = ttk.Frame(left)
        bt2.pack(fill="x")
        for txt, cmd in [("Suddividi strati...", self.subdivide),
                         ("Annulla suddivisione", self.undo_subdivide),
                         ("Importa CSV...", self.import_csv),
                         ("Esporta CSV...", self.export_csv)]:
            ttk.Button(bt2, text=txt, command=cmd).pack(side="left", padx=2)
        ttk.Label(left, text="L'ultima riga e' il semispazio (bedrock): spessore infinito, comportamento lineare.",
                  foreground="gray30").pack(anchor="w", pady=(6, 0))

        self.vs_plot = PlotFrame(pw, figsize=(3.6, 5), toolbar=False)
        pw.add(self.vs_plot, weight=1)

    def _water(self) -> float:
        if not self.use_water.get():
            return NO_WATER
        try:
            return parse_float(self.water_var.get(), "Profondita' falda", 0)
        except ValueError:
            return NO_WATER

    @staticmethod
    def _describe_curve(s: dict) -> (str, str):
        c = s.get("curve", {"type": "linear"})
        t = c.get("type", "linear")
        if t == "darendeli":
            return CURVE_LABELS[t], f"PI={c['pi']:g}  OCR={c['ocr']:g}  K0={c['k0']:g}"
        if t == "hyperbolic":
            return CURVE_LABELS[t], f"gamma_ref={c['gamma_ref_pct']:g} %  Dmin={c['d_min_pct']:g} %"
        if t == "tabular":
            return CURVE_LABELS[t], f"{len(c['gamma_pct'])} punti"
        return CURVE_LABELS["linear"], f"D = {s.get('damping_pct', 2.0):g} %"

    def _refresh_tree(self, select: Optional[int] = None):
        self.tree.delete(*self.tree.get_children())
        z = 0.0
        n = len(self.specs)
        for i, s in enumerate(self.specs):
            last = i == n - 1
            ctype, par = self._describe_curve(s)
            gmax = s["unit_weight"] / G_ACC * s["vs"] ** 2 / 1000.0
            self.tree.insert("", "end", iid=str(i), values=(
                i + 1, s["name"], f"{z:.2f}", "inf" if last else f"{s['thickness']:.4g}",
                f"{s['vs']:g}", f"{s['unit_weight']:g}", ctype, par, f"{gmax:.1f}"))
            if not last:
                z += float(s["thickness"])
        if select is not None and 0 <= select < n:
            self.tree.selection_set(str(select))
            self.tree.focus(str(select))
            self.tree.see(str(select))

    def _refresh_vs_plot(self):
        fig = self.vs_plot.fig
        fig.clear()
        ax = fig.add_subplot(111)
        z = 0.0
        n = len(self.specs)
        total = sum(float(s["thickness"]) for s in self.specs[:-1]) or 1.0
        ext = max(5.0, 0.15 * total)
        for i, s in enumerate(self.specs):
            h = ext if i == n - 1 else float(s["thickness"])
            ax.plot([s["vs"], s["vs"]], [z, z + h], "C0-", lw=2)
            if i < n - 1:
                ax.plot([s["vs"], self.specs[i + 1]["vs"]], [z + h, z + h], "C0-", lw=2)
            z += h
        wt = self._water()
        if wt < total + ext:
            ax.axhline(wt, color="C9", ls="--", lw=1)
            ax.text(0.98, wt, " falda", color="C9", va="bottom", ha="right",
                    transform=ax.get_yaxis_transform(), fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel="Vs [m/s]", ylabel="profondita' [m]", title="Profilo di Vs")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        self.vs_plot.draw()

    def _on_profile_changed(self, select: Optional[int] = None):
        self._refresh_tree(select)
        self._refresh_vs_plot()
        n = len(self.specs)
        for sb in (self.sb_in, self.sb_out):
            sb.configure(to=n)
        self.lbl_n.configure(text=f"Strati: 1 ... {n}   (strato {n} = semispazio / bedrock)")
        self._refresh_curve_list()

    # -------- editing
    def _sel_index(self) -> Optional[int]:
        sel = self.tree.selection()
        return int(sel[0]) if sel else None

    def add_layer(self):
        i = self._sel_index()
        n = len(self.specs)
        pos = i + 1 if (i is not None and i < n - 1) else n - 1     # mai dopo il semispazio
        s = copy.deepcopy(self.specs[pos - 1]) if pos > 0 else default_layer()
        s["name"] = f"strato {n}"
        self.specs.insert(pos, s)
        self._on_profile_changed(pos)

    def dup_layer(self):
        i = self._sel_index()
        if i is None or i == len(self.specs) - 1:
            return
        s = copy.deepcopy(self.specs[i])
        self.specs.insert(i + 1, s)
        self._on_profile_changed(i + 1)

    def del_layer(self):
        i = self._sel_index()
        if i is None:
            return
        if i == len(self.specs) - 1:
            messagebox.showinfo("Semispazio", "Il semispazio (ultima riga) non si puo' eliminare.")
            return
        if len(self.specs) <= 2:
            messagebox.showinfo("Profilo", "Serve almeno uno strato sopra il semispazio.")
            return
        del self.specs[i]
        self._on_profile_changed(min(i, len(self.specs) - 1))

    def move_layer(self, d: int):
        i = self._sel_index()
        if i is None:
            return
        j = i + d
        if i == len(self.specs) - 1 or j < 0 or j >= len(self.specs) - 1:
            return
        self.specs[i], self.specs[j] = self.specs[j], self.specs[i]
        self._on_profile_changed(j)

    def edit_selected(self):
        i = self._sel_index()
        if i is None:
            messagebox.showinfo("Modifica", "Seleziona uno strato.")
            return
        self._edit_layer_dialog(i)

    def _edit_layer_dialog(self, i: int):
        dlg = LayerDialog(self, self.specs[i], i, i == len(self.specs) - 1)
        if dlg.result is not None:
            self.specs[i] = dlg.result
            self._on_profile_changed(i)

    def _on_tree_double_click(self, event):
        if self.tree.identify("region", event.x, event.y) != "cell":
            return
        row = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        if not row:
            return
        idx = int(row)
        cname = self.COLS[int(col[1:]) - 1]
        last = idx == len(self.specs) - 1
        if cname in ("curve", "par"):
            self._edit_layer_dialog(idx)
        elif cname in ("name", "h", "vs", "uw"):
            if cname == "h" and last:
                return
            self._inline_edit(row, col, idx, cname)

    def _inline_edit(self, row, col, idx, cname):
        if self._edit_entry is not None:
            self._edit_entry.destroy()
        x, y, w, h = self.tree.bbox(row, col)
        key = {"name": "name", "h": "thickness", "vs": "vs", "uw": "unit_weight"}[cname]
        var = tk.StringVar(value=str(self.specs[idx][key]))
        e = ttk.Entry(self.tree, textvariable=var)
        e.place(x=x, y=y, width=w, height=h)
        e.focus_set()
        e.select_range(0, "end")
        self._edit_entry = e

        def commit(_=None):
            if self._edit_entry is not e:
                return
            self._edit_entry = None
            try:
                if key == "name":
                    self.specs[idx][key] = var.get().strip()
                else:
                    self.specs[idx][key] = parse_float(var.get(), cname, 0, strict_min=True)
            except ValueError as ex:
                messagebox.showerror("Valore non valido", str(ex))
            e.destroy()
            self._on_profile_changed(idx)

        def cancel(_=None):
            if self._edit_entry is e:
                self._edit_entry = None
                e.destroy()

        e.bind("<Return>", commit)
        e.bind("<FocusOut>", commit)
        e.bind("<Escape>", cancel)

    # -------- suddivisione / CSV
    def subdivide(self):
        dlg = ParamDialog(self, "Suddividi strati",
                          [("f max [Hz]", 20), ("Punti per lunghezza d'onda", 5),
                           ("h massimo [m] (0 = nessuno)", 0)],
                          note="Ogni strato viene diviso in sottostrati con "
                               "h <= Vs / (n * f max), come nel criterio di EERA/SHAKE.")
        if dlg.result is None:
            return
        try:
            fmax = parse_float(dlg.result[0], "f max", 0, strict_min=True)
            npw = int(parse_float(dlg.result[1], "Punti per lunghezza d'onda", 1))
            hmax = parse_float(dlg.result[2], "h massimo", 0)
        except ValueError as e:
            messagebox.showerror("Dati non validi", str(e))
            return
        self._undo_specs = copy.deepcopy(self.specs)
        self.specs = subdivide_specs(self.specs, fmax, npw, hmax or None)
        self._on_profile_changed(0)
        self._set_default_analysis_points()
        self.set_status(f"Profilo suddiviso: {len(self.specs)} strati (incluso il semispazio).")

    def undo_subdivide(self):
        if self._undo_specs is None:
            messagebox.showinfo("Annulla", "Nessuna suddivisione da annullare.")
            return
        self.specs, self._undo_specs = self._undo_specs, None
        self._on_profile_changed(0)
        self._set_default_analysis_points()

    def import_csv(self):
        p = filedialog.askopenfilename(title="Importa profilo da CSV",
                                       filetypes=[("CSV", "*.csv"), ("Tutti", "*.*")])
        if not p:
            return
        try:
            specs = import_profile_csv(p)
            err = validate_specs(specs)
            if err:
                raise ValueError("\n".join(err))
        except Exception as e:
            messagebox.showerror("Errore di importazione", str(e))
            return
        self._undo_specs = copy.deepcopy(self.specs)
        self.specs = specs
        self._on_profile_changed(0)
        self._set_default_analysis_points()

    def export_csv(self):
        p = filedialog.asksaveasfilename(title="Esporta profilo in CSV", defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if not p:
            return
        ok = export_profile_csv(p, self.specs)
        if not ok:
            messagebox.showwarning("Curve tabellari",
                                   "Le curve tabellari non sono esportabili in CSV (salvate come "
                                   "lineari). Usa il salvataggio del progetto (.json) per conservarle.")
        self.set_status(f"Profilo esportato: {p}")

    # ============================================================ MOTO SISMICO
    def _build_tab_motion(self):
        tab = ttk.Frame(self.nb, padding=6)
        self.nb.add(tab, text="2. Moto sismico")
        pw = ttk.PanedWindow(tab, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, padding=(0, 0, 8, 0))
        pw.add(left, weight=1)

        lf = ttk.LabelFrame(left, text="Carica accelerogramma", padding=8)
        lf.pack(fill="x")
        self.m_units = tk.StringVar(value="g")
        self.m_dt = tk.StringVar(value="")
        ttk.Label(lf, text="Unita' dei valori").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(lf, textvariable=self.m_units, values=["g", "m/s2", "cm/s2"],
                     state="readonly", width=8).grid(row=0, column=1, padx=6)
        ttk.Label(lf, text="dt [s] (solo file a 1 colonna)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(lf, textvariable=self.m_dt, width=10).grid(row=1, column=1, padx=6)
        ttk.Button(lf, text="Apri file...", command=self.load_motion).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Label(lf, text="Formati: PEER NGA (.AT2), testo/CSV a 1 colonna (con dt) o\n"
                           "2 colonne (tempo, accelerazione).", foreground="gray30").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        sf = ttk.LabelFrame(left, text="Genera moto sintetico", padding=8)
        sf.pack(fill="x", pady=8)
        self.s_pga = tk.StringVar(value="0.25")
        self.s_dur = tk.StringVar(value="30")
        self.s_dt = tk.StringVar(value="0.01")
        self.s_flo = tk.StringVar(value="0.3")
        self.s_fhi = tk.StringVar(value="15")
        self.s_seed = tk.StringVar(value="1")
        for i, (lab, v) in enumerate([("PGA [g]", self.s_pga), ("Durata [s]", self.s_dur),
                                      ("dt [s]", self.s_dt), ("f min [Hz]", self.s_flo),
                                      ("f max [Hz]", self.s_fhi), ("Seed", self.s_seed)]):
            ttk.Label(sf, text=lab).grid(row=i, column=0, sticky="w", pady=1)
            ttk.Entry(sf, textvariable=v, width=10).grid(row=i, column=1, padx=6)
        ttk.Button(sf, text="Genera", command=self.gen_motion).grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        cf = ttk.LabelFrame(left, text="Elaborazione", padding=8)
        cf.pack(fill="x")
        self.m_pga_target = tk.StringVar(value="0.20")
        self.m_factor = tk.StringVar(value="1.0")
        ttk.Label(cf, text="PGA target [g]").grid(row=0, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.m_pga_target, width=8).grid(row=0, column=1, padx=6)
        ttk.Button(cf, text="Scala a PGA", command=self.scale_pga).grid(row=0, column=2)
        ttk.Label(cf, text="Fattore di scala").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(cf, textvariable=self.m_factor, width=8).grid(row=1, column=1, padx=6)
        ttk.Button(cf, text="Applica", command=self.scale_factor).grid(row=1, column=2)
        ttk.Button(cf, text="Rimuovi media (baseline)", command=self.remove_mean).grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        ttk.Button(cf, text="Salva moto (CSV, in g)...", command=self.save_motion).grid(
            row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))

        self.m_info = tk.StringVar(value="Nessun accelerogramma caricato.")
        ttk.Label(left, textvariable=self.m_info, wraplength=300, justify="left").pack(
            anchor="w", pady=10)

        self.motion_plot = PlotFrame(pw, figsize=(7, 6))
        pw.add(self.motion_plot, weight=4)
        self._refresh_motion()

    def _refresh_motion(self):
        m = self.motion
        if m is None:
            self.m_info.set("Nessun accelerogramma caricato.")
        else:
            n = len(m.acc)
            arias = np.pi / (2 * G_ACC) * np.trapezoid(m.acc ** 2, dx=m.dt) if hasattr(np, "trapezoid") \
                else np.pi / (2 * G_ACC) * np.trapz(m.acc ** 2, dx=m.dt)
            self.m_info.set(f"{m.name}\nPunti: {n}   dt = {m.dt:g} s   durata = {n * m.dt:.2f} s\n"
                            f"PGA = {m.pga / G_ACC:.4f} g   Intensita' di Arias = {arias:.3f} m/s")
        plots.plot_motion(self.motion_plot.fig, m)
        self.motion_plot.draw()

    def load_motion(self):
        p = filedialog.askopenfilename(title="Apri accelerogramma",
                                       filetypes=[("Accelerogrammi", "*.AT2 *.at2 *.csv *.txt *.dat"),
                                                  ("Tutti", "*.*")])
        if not p:
            return
        try:
            dt = parse_float(self.m_dt.get(), "dt", 0, strict_min=True) if self.m_dt.get().strip() else None
            self.motion = Motion.from_file(p, dt=dt, units=self.m_units.get(),
                                           name=os.path.basename(p))
        except Exception as e:
            messagebox.showerror("Errore di lettura", str(e))
            return
        self._refresh_motion()
        self.set_status(f"Accelerogramma caricato: {p}")

    def gen_motion(self):
        try:
            self.motion = Motion.synthetic(
                pga_g=parse_float(self.s_pga.get(), "PGA", 0, strict_min=True),
                duration=parse_float(self.s_dur.get(), "Durata", 1),
                dt=parse_float(self.s_dt.get(), "dt", 0.0005, 0.1),
                f_lo=parse_float(self.s_flo.get(), "f min", 0.01, strict_min=True),
                f_hi=parse_float(self.s_fhi.get(), "f max", 0.1, strict_min=True),
                seed=int(parse_float(self.s_seed.get(), "Seed", 0)))
        except Exception as e:
            messagebox.showerror("Dati non validi", str(e))
            return
        self._refresh_motion()

    def _need_motion(self) -> bool:
        if self.motion is None:
            messagebox.showinfo("Moto sismico", "Carica o genera prima un accelerogramma.")
            return False
        return True

    def scale_pga(self):
        if not self._need_motion():
            return
        try:
            self.motion = self.motion.scaled_to_pga(parse_float(self.m_pga_target.get(), "PGA", 0, strict_min=True))
        except ValueError as e:
            messagebox.showerror("Dati non validi", str(e))
            return
        self._refresh_motion()

    def scale_factor(self):
        if not self._need_motion():
            return
        try:
            self.motion = self.motion.scaled(parse_float(self.m_factor.get(), "Fattore", 0, strict_min=True))
        except ValueError as e:
            messagebox.showerror("Dati non validi", str(e))
            return
        self._refresh_motion()

    def remove_mean(self):
        if not self._need_motion():
            return
        self.motion = Motion(self.motion.acc - np.mean(self.motion.acc), self.motion.dt, self.motion.name)
        self._refresh_motion()

    def save_motion(self):
        if not self._need_motion():
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not p:
            return
        np.savetxt(p, np.column_stack([self.motion.time, self.motion.acc / G_ACC]),
                   delimiter=",", header="t[s],a[g]", comments="")
        self.set_status(f"Moto salvato: {p}")

    # ================================================================ ANALISI
    def _build_tab_analysis(self):
        tab = ttk.Frame(self.nb, padding=8)
        self.nb.add(tab, text="3. Analisi")
        left = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=(0, 10))

        self.lbl_n = ttk.Label(left, text="")
        self.lbl_n.pack(anchor="w", pady=(0, 6))

        inf = ttk.LabelFrame(left, text="Moto di INPUT (assegnato a...)", padding=8)
        inf.pack(fill="x")
        ttk.Label(inf, text="Strato").grid(row=0, column=0, sticky="w")
        self.sb_in = ttk.Spinbox(inf, from_=1, to=3, textvariable=self.in_layer, width=6)
        self.sb_in.grid(row=0, column=1, padx=6)
        ttk.Radiobutton(inf, text="Within", variable=self.in_kind, value="within").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(inf, text="Outcrop", variable=self.in_kind, value="outcrop").grid(row=1, column=1, sticky="w")
        ttk.Button(inf, text="Bedrock outcrop", command=self._input_bedrock).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        outf = ttk.LabelFrame(left, text="Punti di OUTPUT (accelerogrammi richiesti)", padding=8)
        outf.pack(fill="both", expand=True, pady=8)
        row = ttk.Frame(outf)
        row.pack(fill="x")
        ttk.Label(row, text="Strato").pack(side="left")
        self.sb_out = ttk.Spinbox(row, from_=1, to=3, textvariable=self.out_layer, width=6)
        self.sb_out.pack(side="left", padx=4)
        ttk.Radiobutton(row, text="Within", variable=self.out_kind, value="within").pack(side="left")
        ttk.Radiobutton(row, text="Outcrop", variable=self.out_kind, value="outcrop").pack(side="left")
        ttk.Button(row, text="Aggiungi", command=self.add_output).pack(side="left", padx=6)
        fr, self.out_tree = make_tree(outf, ("layer", "kind", "depth"),
                                      ["Strato", "Tipo", "z sommita' [m]"], [70, 90, 110],
                                      height=7, selectmode="extended")
        fr.pack(fill="both", expand=True, pady=6)
        b = ttk.Frame(outf)
        b.pack(fill="x")
        ttk.Button(b, text="Rimuovi selezionati", command=self.remove_outputs).pack(side="left", padx=2)
        ttk.Button(b, text="Tutti (within)", command=self.outputs_all_within).pack(side="left", padx=2)
        ttk.Button(b, text="Predefiniti", command=self._set_default_outputs).pack(side="left", padx=2)

        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True)

        pf = ttk.LabelFrame(right, text="Parametri dell'analisi lineare equivalente", padding=8)
        pf.pack(fill="x")
        items = [("Magnitudo M  (R_gamma = (M-1)/10)", self.v_mag),
                 ("Tolleranza di convergenza [%]", self.v_tol),
                 ("Iterazioni massime", self.v_maxit),
                 ("Zeri aggiunti in coda [s]", self.v_pad)]
        for i, (lab, v) in enumerate(items):
            ttk.Label(pf, text=lab).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(pf, textvariable=v, width=10).grid(row=i, column=1, padx=8)
        ttk.Checkbutton(pf, text="Imposta R_gamma manualmente:", variable=self.use_rg).grid(
            row=4, column=0, sticky="w", pady=2)
        ttk.Entry(pf, textvariable=self.v_rg, width=10).grid(row=4, column=1, padx=8)

        ttk.Label(right, text="WITHIN = moto alla sommita' dello strato nella colonna;  "
                              "OUTCROP = moto di affioramento libero (2 x onda ascendente).",
                  foreground="gray30", wraplength=620, justify="left").pack(anchor="w", pady=8)

        self.btn_run = ttk.Button(right, text="ESEGUI ANALISI", style="Big.TButton",
                                  command=self.run_analysis)
        self.btn_run.pack(anchor="w")
        self.pbar = ttk.Progressbar(right, mode="indeterminate")
        self.pbar.pack(fill="x", pady=8)

        lf = ttk.LabelFrame(right, text="Registro", padding=4)
        lf.pack(fill="both", expand=True)
        self.log = tk.Text(lf, height=12, state="disabled", font=("Courier", 9), wrap="word")
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _input_bedrock(self):
        self.in_layer.set(str(len(self.specs)))
        self.in_kind.set("outcrop")

    def _set_default_outputs(self):
        n = len(self.specs)
        self.out_tree.delete(*self.out_tree.get_children())
        self._insert_output(1, "within")
        self._insert_output(n, "outcrop")

    def _set_default_analysis_points(self):
        n = len(self.specs)
        self.in_layer.set(str(n))
        self.in_kind.set("outcrop")
        self._set_default_outputs()

    def _depth_top(self, layer: int) -> float:
        return float(sum(float(s["thickness"]) for s in self.specs[:layer - 1]))

    def _insert_output(self, layer: int, kind: str):
        key = f"{layer}-{kind}"
        if self.out_tree.exists(key):
            return
        self.out_tree.insert("", "end", iid=key,
                             values=(layer, kind, f"{self._depth_top(layer):.2f}"))

    def add_output(self):
        try:
            layer = int(parse_float(self.out_layer.get(), "Strato", 1, len(self.specs)))
        except ValueError as e:
            messagebox.showerror("Dati non validi", str(e))
            return
        self._insert_output(layer, self.out_kind.get())

    def remove_outputs(self):
        for i in self.out_tree.selection():
            self.out_tree.delete(i)

    def outputs_all_within(self):
        self.out_tree.delete(*self.out_tree.get_children())
        for i in range(1, len(self.specs) + 1):
            self._insert_output(i, "within")

    def _collect_outputs(self) -> List[Location]:
        return [Location(int(self.out_tree.set(i, "layer")), self.out_tree.set(i, "kind"))
                for i in self.out_tree.get_children()]

    # -------------------------------------------------------------- esecuzione
    def run_analysis(self, blocking: bool = False):
        """Avvia l'analisi (thread separato; blocking=True per uso programmatico)."""
        if self._running:
            return
        if not self._need_motion():
            return
        err = validate_specs(self.specs)
        if err:
            messagebox.showerror("Profilo non valido", "\n".join(err))
            return
        n = len(self.specs)
        try:
            in_layer = int(parse_float(self.in_layer.get(), "Strato di input", 1, n))
            in_loc = Location(in_layer, self.in_kind.get())
            outs = [lo for lo in self._collect_outputs()]
            for lo in outs:
                if lo.layer > n:
                    raise ValueError(f"Output sullo strato {lo.layer}: il profilo ha solo {n} strati.")
            if not outs:
                outs = [Location(1, "within")]
                self._insert_output(1, "within")
                self._log("Nessun output selezionato: aggiunto strato 1 (within).")
            mag = parse_float(self.v_mag.get(), "Magnitudo", 3, 10)
            rg = parse_float(self.v_rg.get(), "R_gamma", 0, 1, True) if self.use_rg.get() else None
            tol = parse_float(self.v_tol.get(), "Tolleranza", 0, 50, True) / 100.0
            maxit = int(parse_float(self.v_maxit.get(), "Iterazioni massime", 1, 200))
            pad = parse_float(self.v_pad.get(), "Zeri in coda", 0)
            profile = profile_from_specs(copy.deepcopy(self.specs), self._water())
        except ValueError as e:
            messagebox.showerror("Dati non validi", str(e))
            return

        motion = self.motion
        self._log("-" * 60)
        self._log(f"Avvio: input {in_loc}, {len(outs)} output, M={mag:g}, tol={tol * 100:g}%")

        def progress(it, e):
            self._queue.put(("log", f"  iterazione {it:2d}:  errore max = {e * 100:8.3f} %"))

        def worker():
            try:
                res = EquivalentLinearAnalysis(profile).run(
                    motion, in_loc, outs, magnitude=mag, r_gamma=rg, tol=tol,
                    max_iter=maxit, pad_time=pad, verbose=False, progress=progress)
                self._queue.put(("done", res))
            except Exception:
                self._queue.put(("error", traceback.format_exc()))

        self._running = True
        self.btn_run.state(["disabled"])
        self.pbar.configure(mode="indeterminate")
        self.pbar.start(12)
        self.set_status("Analisi in corso...")
        if blocking:
            worker()
            self._poll_queue(once=True)
        else:
            threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self, once: bool = False):
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._log(payload)
                elif kind == "done":
                    self._finish(payload)
                elif kind == "error":
                    self._running = False
                    self.pbar.stop()
                    self.pbar.configure(mode="determinate", value=0)
                    self.btn_run.state(["!disabled"])
                    self._log(payload)
                    self.set_status("Errore nell'analisi.")
                    messagebox.showerror("Errore nell'analisi", payload.splitlines()[-1])
        except queue.Empty:
            pass
        if not once:
            self.after(100, self._poll_queue)

    def _finish(self, res: EQLResult):
        self._running = False
        self.pbar.stop()
        self.pbar.configure(mode="determinate", value=0)
        self.btn_run.state(["!disabled"])
        self.result = res
        self._log("CONVERGENZA RAGGIUNTA" if res.converged else
                  "ATTENZIONE: convergenza NON raggiunta (aumenta le iterazioni o la tolleranza)")
        self._log(f"Iterazioni: {res.n_iter}   errore finale = {res.errors[-1] * 100:.3f} %")
        for loc, a in res.outputs.items():
            self._log(f"  {loc}:  PGA = {np.max(np.abs(a)) / G_ACC:.4f} g")
        self.set_status("Analisi completata." if res.converged else "Analisi completata (NON convergita).")
        self._show_result()
        self.nb.select(3)

    # =============================================================== RISULTATI
    RCOLS = ("n", "name", "z", "h", "vs0", "vseq", "gg", "d", "gmax", "geff", "tau", "pgw", "pgo")

    def _build_tab_results(self):
        tab = ttk.Frame(self.nb, padding=6)
        self.nb.add(tab, text="4. Risultati")
        top = ttk.Frame(tab)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(top, text="Grafico:").pack(side="left")
        self.view_var = tk.StringVar(value=self.VIEWS[0])
        cb = ttk.Combobox(top, textvariable=self.view_var, values=self.VIEWS, state="readonly", width=40)
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self._draw_view())
        ttk.Label(top, text="Smorz. spettri [%]").pack(side="left", padx=(12, 2))
        e = ttk.Entry(top, textvariable=self.v_sa_damp, width=5)
        e.pack(side="left")
        e.bind("<Return>", lambda ev: self._draw_view())
        ttk.Button(top, text="Esporta tutto...", command=self.export_all).pack(side="right", padx=2)
        ttk.Button(top, text="Tabella CSV", command=self.export_table).pack(side="right", padx=2)
        ttk.Button(top, text="Storie temporali CSV", command=self.export_timeseries).pack(side="right", padx=2)
        ttk.Button(top, text="Grafico PNG", command=self.export_png).pack(side="right", padx=2)

        self.r_summary = tk.StringVar(value="Nessun risultato: esegui l'analisi (scheda 3).")
        ttk.Label(tab, textvariable=self.r_summary, font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        pw = ttk.PanedWindow(tab, orient="vertical")
        pw.pack(fill="both", expand=True, pady=4)
        self.res_plot = PlotFrame(pw, figsize=(9, 5))
        pw.add(self.res_plot, weight=4)
        fr, self.res_tree = make_tree(
            pw, self.RCOLS,
            ["#", "Nome", "z [m]", "h [m]", "Vs0", "Vs eq.", "G/Gmax", "D [%]", "g max [%]",
             "g eff [%]", "tau max [kPa]", "PGA within [g]", "PGA outcrop [g]"],
            [34, 130, 60, 55, 60, 60, 65, 55, 75, 75, 85, 95, 100], height=7, stretch_col="name")
        pw.add(fr, weight=2)

    def _show_result(self):
        res = self.result
        if res is None:
            return
        self.r_summary.set(
            f"Input: {res.input_loc} - {res.motion.name} (PGA {res.motion.pga / G_ACC:.3f} g)    "
            f"M={res.magnitude:g}  R_gamma={res.r_gamma:.3f}    "
            f"{'convergenza OK' if res.converged else 'NON convergita'} in {res.n_iter} iterazioni "
            f"(err. {res.errors[-1] * 100:.2f} %)")
        self.res_tree.delete(*self.res_tree.get_children())
        zt = res.profile.depths_top()
        N = res.profile.n
        for i, L in enumerate(res.profile.layers):
            last = i == N - 1
            self.res_tree.insert("", "end", values=(
                i + 1, L.name, f"{zt[i]:.2f}", "inf" if last else f"{L.thickness:.4g}",
                f"{L.vs:.0f}", f"{res.vs_eq[i]:.0f}", f"{res.g_gmax[i]:.3f}",
                f"{res.damping[i] * 100:.2f}",
                "-" if last else f"{res.gamma_max_pct[i]:.4f}",
                "-" if last else f"{res.gamma_eff_pct[i]:.4f}",
                "-" if last else f"{res.tau_max[i]:.1f}",
                f"{res.pga_within[i] / G_ACC:.4f}", f"{res.pga_outcrop[i] / G_ACC:.4f}"))
        self._draw_view()

    def _draw_view(self):
        res = self.result
        fig = self.res_plot.fig
        if res is None:
            fig.clear()
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Nessun risultato", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.set_axis_off()
            self.res_plot.draw()
            return
        self._render_view(fig, self.view_var.get(), res)
        self.res_plot.draw()

    def _render_view(self, fig, view: str, res: EQLResult):
        try:
            damp = parse_float(self.v_sa_damp.get(), "Smorzamento", 0.1, 50) / 100.0
        except ValueError:
            damp = 0.05
        table = {
            self.VIEWS[0]: lambda: plots.plot_accelerograms(fig, res),
            self.VIEWS[1]: lambda: plots.plot_spectra(fig, res, damp),
            self.VIEWS[2]: lambda: plots.plot_transfer(fig, res),
            self.VIEWS[3]: lambda: plots.plot_fourier(fig, res),
            self.VIEWS[4]: lambda: plots.plot_profiles(fig, res),
            self.VIEWS[5]: lambda: plots.plot_vs_profile(fig, res),
            self.VIEWS[6]: lambda: plots.plot_convergence(fig, res),
            self.VIEWS[7]: lambda: plots.plot_dashboard(fig, res),
        }
        table[view]()

    # -------- esportazioni
    def _need_result(self) -> bool:
        if self.result is None:
            messagebox.showinfo("Risultati", "Esegui prima l'analisi.")
            return False
        return True

    def export_png(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if p:
            self.res_plot.fig.savefig(p, dpi=150)
            self.set_status(f"Grafico salvato: {p}")

    def _write_timeseries(self, p: str):
        res = self.result
        locs = list(res.outputs.keys())
        cols = [res.time(), res.motion.acc / G_ACC] + [res.outputs[l] / G_ACC for l in locs]
        hdr = "t[s],input_" + res.input_loc.kind + f"_L{res.input_loc.layer}[g]," + \
              ",".join(f"L{l.layer}_{l.kind}[g]" for l in locs)
        np.savetxt(p, np.column_stack(cols), delimiter=",", header=hdr, comments="")

    def export_timeseries(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self._write_timeseries(p)
            self.set_status(f"Storie temporali salvate: {p}")

    def _write_table(self, p: str):
        import csv
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["#", "nome", "z_sommita_m", "h_m", "Vs0_m_s", "Vs_eq_m_s", "G/Gmax",
                        "D_pct", "gamma_max_pct", "gamma_eff_pct", "tau_max_kPa",
                        "PGA_within_g", "PGA_outcrop_g"])
            for i in self.res_tree.get_children():
                w.writerow(self.res_tree.item(i, "values"))

    def export_table(self):
        if not self._need_result():
            return
        p = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if p:
            self._write_table(p)
            self.set_status(f"Tabella salvata: {p}")

    def export_all(self):
        if not self._need_result():
            return
        d = filedialog.askdirectory(title="Cartella di destinazione")
        if not d:
            return
        res = self.result
        try:
            self._write_timeseries(os.path.join(d, "storie_temporali.csv"))
            self._write_table(os.path.join(d, "tabella_strati.csv"))
            # spettri di risposta
            per = plots.PERIODS
            cols = [per, response_spectrum(res.motion.acc, res.motion.dt, per) / G_ACC]
            hdr = ["T[s]", "Sa_input[g]"]
            for loc, a in res.outputs.items():
                cols.append(response_spectrum(a, res.motion.dt, per) / G_ACC)
                hdr.append(f"Sa_L{loc.layer}_{loc.kind}[g]")
            np.savetxt(os.path.join(d, "spettri_risposta.csv"), np.column_stack(cols),
                       delimiter=",", header=",".join(hdr), comments="")
            # funzioni di trasferimento
            cols = [res.freqs] + [np.abs(F) for F in res.transfer.values()]
            hdr = ["f[Hz]"] + [f"|F|_L{l.layer}_{l.kind}" for l in res.transfer]
            np.savetxt(os.path.join(d, "funzioni_trasferimento.csv"), np.column_stack(cols),
                       delimiter=",", header=",".join(hdr), comments="")
            # grafici
            fig = Figure(figsize=(11, 6.5), dpi=130)
            slugs = ["accelerogrammi", "spettri_risposta", "funzioni_trasferimento",
                     "spettri_fourier", "profili", "profilo_vs", "convergenza", "cruscotto"]
            for k, v in enumerate(self.VIEWS):
                self._render_view(fig, v, res)
                fig.savefig(os.path.join(d, f"grafico_{k + 1}_{slugs[k]}.png"))
        except Exception as e:
            messagebox.showerror("Errore di esportazione", str(e))
            return
        self.set_status(f"Risultati esportati in {d}")
        messagebox.showinfo("Esportazione", f"File salvati in:\n{d}")

    # ================================================================== CURVE
    def _build_tab_curves(self):
        tab = ttk.Frame(self.nb, padding=6)
        self.nb.add(tab, text="5. Curve")
        left = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=(0, 8))
        ttk.Label(left, text="Strati (selezione multipla):").pack(anchor="w")
        self.curve_list = tk.Listbox(left, selectmode="extended", width=30, height=25,
                                     exportselection=False)
        self.curve_list.pack(fill="y", expand=True)
        self.curve_list.bind("<<ListboxSelect>>", lambda e: self._draw_curves())
        ttk.Button(left, text="Seleziona tutti", command=self._curves_all).pack(fill="x", pady=4)
        ttk.Label(left, text="Le curve dipendono dalla tensione efficace\n"
                             "a meta' strato (Darendeli).", foreground="gray30").pack(anchor="w")
        self.curve_plot = PlotFrame(tab, figsize=(8, 5))
        self.curve_plot.pack(side="left", fill="both", expand=True)

    def _refresh_curve_list(self):
        if not hasattr(self, "curve_list"):
            return
        sel = set(self.curve_list.curselection())
        self.curve_list.delete(0, "end")
        for i, s in enumerate(self.specs[:-1]):
            ctype, _ = self._describe_curve(s)
            self.curve_list.insert("end", f"{i + 1}  {s['name']}  [{ctype}]")
        for i in sel:
            if i < self.curve_list.size():
                self.curve_list.selection_set(i)

    def _curves_all(self):
        self.curve_list.selection_set(0, "end")
        self._draw_curves()

    def _draw_curves(self):
        sel = list(self.curve_list.curselection())
        try:
            prof = profile_from_specs(copy.deepcopy(self.specs), self._water())
        except Exception as e:
            self.set_status(f"Profilo non valido: {e}")
            return
        plots.plot_layer_curves(self.curve_plot.fig, prof.layers, sel or None)
        self.curve_plot.draw()

    def _on_tab_changed(self, _=None):
        if self.nb.index("current") == 4:
            self._draw_curves()

    # ============================================================== PROGETTO
    def _analysis_dict(self) -> dict:
        return dict(
            input=dict(layer=self.in_layer.get(), kind=self.in_kind.get()),
            outputs=[[lo.layer, lo.kind] for lo in self._collect_outputs()],
            magnitude=self.v_mag.get(), use_rgamma=bool(self.use_rg.get()), rgamma=self.v_rg.get(),
            tol_pct=self.v_tol.get(), max_iter=self.v_maxit.get(), pad_time=self.v_pad.get())

    def _apply_analysis_dict(self, a: dict):
        if not a:
            self._set_default_analysis_points()
            return
        self.in_layer.set(str(a["input"]["layer"]))
        self.in_kind.set(a["input"]["kind"])
        self.out_tree.delete(*self.out_tree.get_children())
        n = len(self.specs)
        for layer, kind in a.get("outputs", []):
            if 1 <= int(layer) <= n:
                self._insert_output(int(layer), kind)
        self.v_mag.set(str(a.get("magnitude", "7.0")))
        self.use_rg.set(bool(a.get("use_rgamma", False)))
        self.v_rg.set(str(a.get("rgamma", "0.65")))
        self.v_tol.set(str(a.get("tol_pct", "1.0")))
        self.v_maxit.set(str(a.get("max_iter", "20")))
        self.v_pad.set(str(a.get("pad_time", "10")))

    def new_project(self):
        if not messagebox.askyesno("Nuovo progetto", "Scartare il progetto corrente?"):
            return
        self.specs = default_specs()
        self.water_var.set("2.0")
        self.use_water.set(True)
        self.motion = None
        self.result = None
        self.project_path = None
        self._on_profile_changed(0)
        self._set_default_analysis_points()
        self._refresh_motion()
        self._show_result()
        self._draw_view()
        self.r_summary.set("Nessun risultato: esegui l'analisi (scheda 3).")

    def save_project(self, save_as: bool = False):
        p = self.project_path
        if save_as or not p:
            p = filedialog.asksaveasfilename(title="Salva progetto", defaultextension=".json",
                                             filetypes=[("Progetto EQL", "*.json")])
            if not p:
                return
        try:
            wt = self._water()
            save_project(p, self.specs, None if wt >= NO_WATER else wt, self.motion,
                         self._analysis_dict())
        except Exception as e:
            messagebox.showerror("Errore di salvataggio", str(e))
            return
        self.project_path = p
        self.set_status(f"Progetto salvato: {p}")

    def open_project(self, path: Optional[str] = None):
        p = path or filedialog.askopenfilename(title="Apri progetto",
                                               filetypes=[("Progetto EQL", "*.json"), ("Tutti", "*.*")])
        if not p:
            return
        try:
            specs, wt, motion, analysis = load_project(p)
            err = validate_specs(specs)
            if err:
                raise ValueError("\n".join(err))
        except Exception as e:
            messagebox.showerror("Errore di apertura", str(e))
            return
        self.specs = specs
        self.use_water.set(wt is not None)
        self.water_var.set("2.0" if wt is None else f"{wt:g}")
        self.motion = motion
        self.result = None
        self.project_path = p
        self._on_profile_changed(0)
        self._apply_analysis_dict(analysis)
        self._refresh_motion()
        self._draw_view()
        self.r_summary.set("Nessun risultato: esegui l'analisi (scheda 3).")
        self.set_status(f"Progetto aperto: {p}")

    def load_demo(self):
        specs, wt = demo_specs(subdivided=True)
        self.specs = specs
        self.use_water.set(True)
        self.water_var.set(f"{wt:g}")
        self.motion = Motion.synthetic(pga_g=0.25, duration=30, dt=0.01)
        self.result = None
        self.project_path = None
        self._on_profile_changed(0)
        self._set_default_analysis_points()
        self.v_mag.set("6.5")
        self._refresh_motion()
        self._draw_view()
        self.set_status("Esempio caricato: premi 'ESEGUI ANALISI' nella scheda 3.")


def main():
    try:                                    # nitidezza su Windows ad alta risoluzione
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
