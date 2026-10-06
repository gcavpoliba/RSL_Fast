# -*- coding: utf-8 -*-
"""
eqlsite/project.py
Conversione fra "specifiche" (dizionari semplici, usati dalla GUI e salvati
in JSON) e oggetti del motore di calcolo; import/export CSV del profilo.

Specifica di uno strato (dict):
    {"name": "sabbia", "thickness": 5.0, "vs": 150.0, "unit_weight": 18.0,
     "curve": {"type": "darendeli", "pi": 0, "ocr": 1, "k0": 0.5, ...},
     "damping_pct": 2.0}                 # usato se curve.type == "linear"

Tipi di curva:  darendeli | hyperbolic | tabular | linear
"""
from __future__ import annotations

import copy
import csv
import json
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core import (DarendeliCurve, HyperbolicCurve, Layer, Motion, SoilCurve,
                   SoilProfile, TabularCurve, G_ACC)

CURVE_TYPES = ("darendeli", "hyperbolic", "tabular", "linear")
CURVE_LABELS = {
    "darendeli": "Darendeli (2001)",
    "hyperbolic": "Iperbolica",
    "tabular": "Tabellare",
    "linear": "Lineare",
}

DEFAULT_CURVES: Dict[str, dict] = {
    "darendeli": dict(type="darendeli", pi=0.0, ocr=1.0, k0=0.5, freq=1.0, n_cycles=10.0),
    "hyperbolic": dict(type="hyperbolic", gamma_ref_pct=0.05, d_min_pct=1.0, a=1.0),
    "tabular": dict(type="tabular", gamma_pct=[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                    g_gmax=[1.0, 0.99, 0.9, 0.5, 0.15, 0.03],
                    damping_pct=[1.0, 1.5, 3.0, 9.0, 18.0, 25.0]),
    "linear": dict(type="linear"),
}


def default_curve(kind: str = "darendeli") -> dict:
    return copy.deepcopy(DEFAULT_CURVES[kind])


def default_layer(name: str = "strato", thickness: float = 5.0, vs: float = 200.0,
                  unit_weight: float = 18.0) -> dict:
    return dict(name=name, thickness=thickness, vs=vs, unit_weight=unit_weight,
                curve=default_curve("darendeli"), damping_pct=2.0)


def default_halfspace() -> dict:
    return dict(name="bedrock", thickness=0.0, vs=800.0, unit_weight=22.0,
                curve=default_curve("linear"), damping_pct=1.0)


def default_specs() -> List[dict]:
    return [
        default_layer("strato 1", 5.0, 180.0, 18.0),
        default_layer("strato 2", 10.0, 260.0, 19.0),
        default_halfspace(),
    ]


# ----------------------------------------------------------------------------
#  spec -> oggetti
# ----------------------------------------------------------------------------
def curve_from_spec(c: dict) -> Optional[SoilCurve]:
    t = c.get("type", "linear")
    if t == "darendeli":
        return DarendeliCurve(pi=float(c.get("pi", 0)), ocr=float(c.get("ocr", 1)),
                              k0=float(c.get("k0", 0.5)), freq=float(c.get("freq", 1)),
                              n_cycles=float(c.get("n_cycles", 10)))
    if t == "hyperbolic":
        return HyperbolicCurve(gamma_ref_pct=float(c.get("gamma_ref_pct", 0.05)),
                               d_min_pct=float(c.get("d_min_pct", 1.0)),
                               a=float(c.get("a", 1.0)))
    if t == "tabular":
        return TabularCurve(c["gamma_pct"], c["g_gmax"], c["damping_pct"])
    return None


def layer_from_spec(s: dict, last: bool = False) -> Layer:
    curve = curve_from_spec(s.get("curve", {"type": "linear"}))
    if last:
        curve = None            # il semispazio e' sempre lineare
    return Layer(thickness=0.0 if last else float(s["thickness"]),
                 vs=float(s["vs"]), unit_weight=float(s["unit_weight"]),
                 curve=curve, damping=float(s.get("damping_pct", 2.0)) / 100.0,
                 name=str(s.get("name", "")))


def profile_from_specs(specs: List[dict], water_table: float = 1e9) -> SoilProfile:
    n = len(specs)
    if n < 2:
        raise ValueError("Servono almeno uno strato e il semispazio")
    layers = [layer_from_spec(s, last=(i == n - 1)) for i, s in enumerate(specs)]
    return SoilProfile(layers, water_table)


def validate_specs(specs: List[dict]) -> List[str]:
    """Restituisce l'elenco dei problemi (lista vuota = tutto ok)."""
    err = []
    if len(specs) < 2:
        err.append("Servono almeno uno strato e il semispazio (ultima riga).")
    for i, s in enumerate(specs):
        tag = f"Strato {i + 1}"
        try:
            if i < len(specs) - 1 and float(s["thickness"]) <= 0:
                err.append(f"{tag}: lo spessore deve essere > 0.")
            if float(s["vs"]) <= 0:
                err.append(f"{tag}: Vs deve essere > 0.")
            if float(s["unit_weight"]) <= 0:
                err.append(f"{tag}: il peso di volume deve essere > 0.")
            c = s.get("curve", {})
            if c.get("type") == "tabular":
                g = np.asarray(c["gamma_pct"], float)
                if len(g) < 2 or np.any(np.diff(g) <= 0):
                    err.append(f"{tag}: curva tabellare con gamma non crescente.")
            if c.get("type") == "linear" or i == len(specs) - 1:
                d = float(s.get("damping_pct", 2.0))
                if not (0.0 <= d < 50.0):
                    err.append(f"{tag}: smorzamento fuori range (0-50 %).")
        except (KeyError, ValueError, TypeError) as e:
            err.append(f"{tag}: dati non validi ({e}).")
    return err


# ----------------------------------------------------------------------------
#  Salvataggio / caricamento progetto (JSON)
# ----------------------------------------------------------------------------
def motion_to_dict(m: Motion) -> dict:
    return dict(name=m.name, dt=m.dt, acc_g=np.round(m.acc / G_ACC, 8).tolist())


def motion_from_dict(d: dict) -> Motion:
    return Motion(np.asarray(d["acc_g"], float) * G_ACC, float(d["dt"]), d.get("name", "input"))


def save_project(path: str, specs: List[dict], water_table: Optional[float],
                 motion: Optional[Motion], analysis: dict) -> None:
    data = dict(version=1, water_table=water_table, layers=specs,
                motion=motion_to_dict(motion) if motion is not None else None,
                analysis=analysis)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def load_project(path: str) -> Tuple[List[dict], Optional[float], Optional[Motion], dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    motion = motion_from_dict(data["motion"]) if data.get("motion") else None
    return (data["layers"], data.get("water_table"), motion, data.get("analysis", {}))


# ----------------------------------------------------------------------------
#  CSV del profilo (Darendeli / iperbolica / lineare; le tabellari solo in JSON)
# ----------------------------------------------------------------------------
CSV_COLS = ["name", "thickness", "vs", "unit_weight", "curve", "pi", "ocr", "k0",
            "gamma_ref_pct", "d_min_pct", "damping_pct"]


def export_profile_csv(path: str, specs: List[dict]) -> bool:
    """Ritorna True se tutte le curve sono state esportate (False se qualche
    curva tabellare e' stata sostituita da quella iperbolica di default)."""
    ok = True
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for s in specs:
            c = s.get("curve", {"type": "linear"})
            t = c.get("type", "linear")
            if t == "tabular":
                ok = False
                t = "linear"
            w.writerow([s["name"], s["thickness"], s["vs"], s["unit_weight"], t,
                        c.get("pi", 0), c.get("ocr", 1), c.get("k0", 0.5),
                        c.get("gamma_ref_pct", 0.05), c.get("d_min_pct", 1.0),
                        s.get("damping_pct", 2.0)])
    return ok


def import_profile_csv(path: str) -> List[dict]:
    specs = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        for row in rd:
            kind = (row.get("curve") or "linear").strip().lower()
            if kind not in CURVE_TYPES or kind == "tabular":
                kind = "linear"
            c = default_curve(kind)
            if kind == "darendeli":
                c.update(pi=float(row.get("pi") or 0), ocr=float(row.get("ocr") or 1),
                         k0=float(row.get("k0") or 0.5))
            elif kind == "hyperbolic":
                c.update(gamma_ref_pct=float(row.get("gamma_ref_pct") or 0.05),
                         d_min_pct=float(row.get("d_min_pct") or 1.0))
            specs.append(dict(name=row.get("name", ""), thickness=float(row["thickness"]),
                              vs=float(row["vs"]), unit_weight=float(row["unit_weight"]),
                              curve=c, damping_pct=float(row.get("damping_pct") or 2.0)))
    return specs


def subdivide_specs(specs: List[dict], fmax: float, n_per_wavelength: int,
                    max_thickness: Optional[float] = None) -> List[dict]:
    """Suddivide gli strati (come SoilProfile.subdivide) lavorando sulle spec."""
    import math
    new = []
    for s in specs[:-1]:
        h_max = float(s["vs"]) / (n_per_wavelength * fmax)
        if max_thickness:
            h_max = min(h_max, max_thickness)
        n_sub = max(1, int(math.ceil(float(s["thickness"]) / h_max)))
        for j in range(n_sub):
            t = copy.deepcopy(s)
            t["thickness"] = float(s["thickness"]) / n_sub
            t["name"] = f"{s['name']}#{j + 1}" if n_sub > 1 else s["name"]
            new.append(t)
    new.append(copy.deepcopy(specs[-1]))
    return new


def demo_specs(subdivided: bool = True) -> Tuple[List[dict], float]:
    """Profilo dimostrativo: 4 strati + bedrock (falda a 3 m)."""
    def layer(name, h, vs, uw, pi):
        c = default_curve("darendeli")
        c["pi"] = pi
        return dict(name=name, thickness=h, vs=vs, unit_weight=uw, curve=c, damping_pct=2.0)
    specs = [
        layer("sabbia lenta", 4.0, 150.0, 17.5, 0.0),
        layer("sabbia", 6.0, 200.0, 18.0, 0.0),
        layer("argilla", 10.0, 250.0, 18.5, 30.0),
        layer("argilla rigida", 15.0, 350.0, 19.0, 30.0),
        dict(name="bedrock", thickness=0.0, vs=800.0, unit_weight=22.0,
             curve=default_curve("linear"), damping_pct=1.0),
    ]
    if subdivided:
        specs = subdivide_specs(specs, fmax=20.0, n_per_wavelength=5)
    return specs, 3.0
