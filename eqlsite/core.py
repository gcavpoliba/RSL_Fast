#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eqlsite/core.py  (motore di calcolo)
====================
Analisi di risposta sismica locale 1D con metodo LINEARE EQUIVALENTE nel
dominio della frequenza (propagazione di onde SH in mezzo stratificato
visco-elastico), sullo schema di EERA (Bardet et al., 2000) e STRATA
(Kottke & Rathje, 2008).

Riferimenti teorici
-------------------
 - Kramer (1996), "Geotechnical Earthquake Engineering", cap. 7
 - Schnabel, Lysmer, Seed (1972), SHAKE
 - Bardet, Ichii, Lin (2000), EERA
 - Darendeli (2001), curve G/Gmax e D in funzione della deformazione

Convenzioni (identiche a EERA)
------------------------------
 * Gli strati sono numerati da 1 (in superficie) a N (ultimo = semispazio,
   o "bedrock", di spessore infinito).
 * Un moto "within" allo strato m e' il moto alla SOMMITA' dello strato m
   (somma delle onde ascendente e discendente): u = A_m + B_m.
 * Un moto "outcrop" allo strato m e' il moto che si avrebbe alla sommita'
   dello strato m se questa fosse una superficie libera: u = 2 A_m.
   (Nello strato 1 within e outcrop coincidono.)
 * L'input puo' essere assegnato come within o outcrop a QUALSIASI strato,
   e l'output puo' essere richiesto come within o outcrop a qualsiasi strato.
 * La deformazione di taglio e' calcolata a META' di ogni strato; la
   deformazione efficace e' gamma_eff = R_gamma * gamma_max, con
   R_gamma = (M - 1) / 10 (default EERA/SHAKE).

Dipendenze: numpy, scipy  (matplotlib opzionale per i grafici)

Uso rapido
----------
    from eqlsite import *
    profilo = SoilProfile([...], water_table=2.0)
    moto    = Motion.from_file("record.AT2", units="g")
    analisi = EquivalentLinearAnalysis(profilo)
    ris = analisi.run(moto,
                      input_loc=Location(layer=6, kind="outcrop"),
                      output_locs=[Location(1, "within"),
                                   Location(3, "within"),
                                   Location(6, "outcrop")],
                      magnitude=6.5)
    ris.print_summary()
    ris.plot()

L'interfaccia grafica e' in eqlsite/gui.py (python run_gui.py).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal

G_ACC = 9.80665      # m/s^2
GAMMA_W = 9.81       # kN/m^3
P_ATM = 101.325      # kPa


# =============================================================================
#  CURVE DI DEGRADO G/Gmax e SMORZAMENTO D(gamma)
# =============================================================================
class SoilCurve:
    """Interfaccia: restituisce (G/Gmax, D) in funzione di gamma [%].

    D e' restituito come FRAZIONE (0.05 = 5 %).
    """

    def evaluate(self, gamma_pct: np.ndarray, sigma_m: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class TabularCurve(SoilCurve):
    """Curve definite per punti (come i fogli 'Curves' di EERA).

    gamma_pct : deformazioni di taglio [%] crescenti
    g_gmax    : G/Gmax
    damping_pct : smorzamento [%]
    Interpolazione lineare in log(gamma); valori costanti oltre gli estremi.
    """
    gamma_pct: Sequence[float]
    g_gmax: Sequence[float]
    damping_pct: Sequence[float]

    def __post_init__(self):
        self.gamma_pct = np.asarray(self.gamma_pct, float)
        self.g_gmax = np.asarray(self.g_gmax, float)
        self.damping_pct = np.asarray(self.damping_pct, float)
        if not (len(self.gamma_pct) == len(self.g_gmax) == len(self.damping_pct)):
            raise ValueError("TabularCurve: vettori di lunghezza diversa")
        if np.any(np.diff(self.gamma_pct) <= 0):
            raise ValueError("TabularCurve: gamma deve essere crescente")

    def evaluate(self, gamma_pct, sigma_m=0.0):
        lg = np.log10(np.maximum(np.asarray(gamma_pct, float), 1e-12))
        lx = np.log10(self.gamma_pct)
        gg = np.interp(lg, lx, self.g_gmax)
        d = np.interp(lg, lx, self.damping_pct) / 100.0
        return gg, d

    @classmethod
    def from_csv(cls, path: str, delimiter: str = ",", skip_header: int = 1):
        """CSV a 3 colonne: gamma[%], G/Gmax, D[%]."""
        data = np.genfromtxt(path, delimiter=delimiter, skip_header=skip_header)
        return cls(data[:, 0], data[:, 1], data[:, 2])


@dataclass
class HyperbolicCurve(SoilCurve):
    """Modello iperbolico G/Gmax = 1/(1+(gamma/gamma_ref)^a) con smorzamento
    di Masing (a=1) + smorzamento minimo.  Utile per prove rapide.
    """
    gamma_ref_pct: float = 0.05
    d_min_pct: float = 1.0
    a: float = 1.0

    def evaluate(self, gamma_pct, sigma_m=0.0):
        g = np.maximum(np.asarray(gamma_pct, float), 1e-8)
        gg = 1.0 / (1.0 + (g / self.gamma_ref_pct) ** self.a)
        d_mas = _masing_damping_pct(g, self.gamma_ref_pct) / 100.0
        return gg, d_mas * gg ** 0.1 + self.d_min_pct / 100.0


def _masing_damping_pct(g_pct: np.ndarray, g_ref_pct: float) -> np.ndarray:
    """Smorzamento di Masing per curva iperbolica (a=1), in %."""
    g = np.maximum(g_pct, 1e-10)
    x = g / g_ref_pct
    # forma numericamente stabile di:  (100/pi)[4(g - gr ln((g+gr)/gr))/(g^2/(g+gr)) - 2]
    d = (100.0 / math.pi) * (4.0 * (g - g_ref_pct * np.log1p(x)) * (g + g_ref_pct) / g ** 2 - 2.0)
    return np.maximum(d, 0.0)


@dataclass
class DarendeliCurve(SoilCurve):
    """Curve di Darendeli (2001) in funzione di PI, OCR e tensione efficace.

    pi        : indice di plasticita' [%] (0 per sabbie/ghiaie)
    ocr       : grado di sovraconsolidazione
    k0        : coeff. di spinta a riposo (per la tensione media efficace)
    freq      : frequenza di carico [Hz] (influenza D_min)
    n_cycles  : numero di cicli di carico (influenza b)
    """
    pi: float = 0.0
    ocr: float = 1.0
    k0: float = 0.5
    freq: float = 1.0
    n_cycles: float = 10.0

    # coefficienti del modello (Darendeli 2001, Tab. 8.12)
    _phi = dict(p1=0.0352, p2=0.0010, p3=0.3246, p4=0.3483, p5=0.9190,
                p6=0.8005, p7=0.0129, p8=-0.1069, p9=-0.2889, p10=0.2919,
                p11=0.6329, p12=-0.0057)

    def evaluate(self, gamma_pct, sigma_m):
        """sigma_m: tensione media efficace [kPa] a meta' strato."""
        p = self._phi
        g = np.maximum(np.asarray(gamma_pct, float), 1e-8)
        s_atm = max(sigma_m, 1.0) / P_ATM

        g_ref = (p["p1"] + p["p2"] * self.pi * self.ocr ** p["p3"]) * s_atm ** p["p4"]
        a = p["p5"]
        gg = 1.0 / (1.0 + (g / g_ref) ** a)

        d_min = (p["p6"] + p["p7"] * self.pi * self.ocr ** p["p8"]) \
            * s_atm ** p["p9"] * (1.0 + p["p10"] * math.log(self.freq))
        d_mas1 = _masing_damping_pct(g, g_ref)
        c1 = -1.1143 * a ** 2 + 1.8618 * a + 0.2523
        c2 = 0.0805 * a ** 2 - 0.0710 * a - 0.0095
        c3 = -0.0005 * a ** 2 + 0.0002 * a + 0.0003
        d_mas = c1 * d_mas1 + c2 * d_mas1 ** 2 + c3 * d_mas1 ** 3
        b = p["p11"] + p["p12"] * math.log(self.n_cycles)
        d = b * gg ** 0.1 * d_mas + d_min
        return gg, d / 100.0


# =============================================================================
#  PROFILO STRATIGRAFICO
# =============================================================================
@dataclass
class Layer:
    """Singolo strato.

    thickness   : spessore [m] (ignorato per l'ultimo strato = semispazio)
    vs          : velocita' onde S a piccole deformazioni [m/s]
    unit_weight : peso di volume totale [kN/m3]
    curve       : SoilCurve; None => strato lineare con smorzamento costante
    damping     : smorzamento [frazione] usato se curve e' None
    """
    thickness: float
    vs: float
    unit_weight: float
    curve: Optional[SoilCurve] = None
    damping: float = 0.02
    name: str = ""
    sigma_m: float = field(default=0.0, init=False)   # tens. media efficace [kPa]

    @property
    def rho(self) -> float:             # Mg/m3 (= kPa s2 / m2)
        return self.unit_weight / G_ACC

    @property
    def gmax(self) -> float:            # kPa
        return self.rho * self.vs ** 2


class SoilProfile:
    """Colonna di terreno: strati 1..N (l'ultimo e' il semispazio)."""

    def __init__(self, layers: Sequence[Layer], water_table: float = 1e9):
        if len(layers) < 2:
            raise ValueError("Servono almeno uno strato e il semispazio")
        self.layers: List[Layer] = list(layers)
        self.water_table = water_table
        self.update_stresses()

    def __len__(self):
        return len(self.layers)

    @property
    def n(self):
        return len(self.layers)

    def update_stresses(self) -> None:
        """Tensione verticale e media efficace a meta' di ogni strato."""
        sv = 0.0
        z_top = 0.0
        for i, L in enumerate(self.layers):
            h = L.thickness if i < self.n - 1 else 10.0
            z_mid = z_top + h / 2.0
            sv_mid = sv + L.unit_weight * h / 2.0
            u = max(z_mid - self.water_table, 0.0) * GAMMA_W
            sv_eff = max(sv_mid - u, 1.0)
            k0 = getattr(L.curve, "k0", 0.5)
            L.sigma_m = sv_eff * (1.0 + 2.0 * k0) / 3.0
            sv += L.unit_weight * h
            z_top += h

    def depths_top(self) -> np.ndarray:
        h = np.array([L.thickness for L in self.layers[:-1]], float)
        return np.concatenate([[0.0], np.cumsum(h)])

    def subdivide(self, fmax: float = 25.0, n_per_wavelength: int = 5,
                  max_thickness: Optional[float] = None) -> "SoilProfile":
        """Suddivide gli strati in sottostrati con
        h <= Vs / (n_per_wavelength * fmax)   (criterio EERA/SHAKE).
        Il semispazio non viene toccato.  Restituisce un NUOVO profilo."""
        new: List[Layer] = []
        for L in self.layers[:-1]:
            h_max = L.vs / (n_per_wavelength * fmax)
            if max_thickness:
                h_max = min(h_max, max_thickness)
            n_sub = max(1, int(math.ceil(L.thickness / h_max)))
            for j in range(n_sub):
                s = copy.deepcopy(L)
                s.thickness = L.thickness / n_sub
                s.name = f"{L.name}#{j + 1}" if n_sub > 1 else L.name
                new.append(s)
        new.append(copy.deepcopy(self.layers[-1]))
        return SoilProfile(new, self.water_table)

    def table(self) -> str:
        z = self.depths_top()
        rows = [f"{'#':>3} {'z_top[m]':>9} {'h[m]':>7} {'Vs[m/s]':>8} {'g[kN/m3]':>9} "
                f"{'sig_m[kPa]':>10}  nome"]
        for i, L in enumerate(self.layers):
            h = f"{L.thickness:7.2f}" if i < self.n - 1 else "    inf"
            rows.append(f"{i + 1:3d} {z[i]:9.2f} {h} {L.vs:8.1f} {L.unit_weight:9.2f} "
                        f"{L.sigma_m:10.1f}  {L.name}")
        return "\n".join(rows)


# =============================================================================
#  MOTO SISMICO
# =============================================================================
@dataclass
class Motion:
    """Accelerogramma: acc in m/s^2, passo dt in s."""
    acc: np.ndarray
    dt: float
    name: str = "input"

    def __post_init__(self):
        self.acc = np.asarray(self.acc, float)

    @property
    def time(self):
        return np.arange(len(self.acc)) * self.dt

    @property
    def pga(self):
        return float(np.max(np.abs(self.acc)))

    def scaled(self, factor: float) -> "Motion":
        return Motion(self.acc * factor, self.dt, self.name)

    def scaled_to_pga(self, pga_g: float) -> "Motion":
        return self.scaled(pga_g * G_ACC / self.pga)

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_file(cls, path: str, dt: Optional[float] = None,
                  units: str = "g", name: Optional[str] = None) -> "Motion":
        """Legge:
          - file PEER NGA (.AT2): header con 'NPTS=..., DT=...'
          - testo/CSV a 1 colonna (serve dt) o a 2 colonne (tempo, acc)
        units: 'g', 'm/s2', 'cm/s2'
        """
        with open(path, "r", errors="ignore") as f:
            lines = f.readlines()
        acc = None
        for i, ln in enumerate(lines[:8]):
            up = ln.upper()
            if "NPTS" in up and "DT" in up:
                txt = up.replace(",", " ").replace("=", " ").split()
                dt = float(txt[txt.index("DT") + 1])
                vals = []
                for l2 in lines[i + 1:]:
                    vals += [float(x) for x in l2.replace(",", " ").split()]
                acc = np.array(vals)
                break
        if acc is None:
            rows = []
            for ln in lines:
                parts = ln.replace(",", " ").replace(";", " ").split()
                try:
                    rows.append([float(x) for x in parts])
                except ValueError:
                    continue            # salta intestazioni
            data = [r for r in rows if r]
            if all(len(r) >= 2 for r in data):
                arr = np.array([r[:2] for r in data])
                dt = float(np.mean(np.diff(arr[:, 0])))
                acc = arr[:, 1]
            else:
                if dt is None:
                    raise ValueError("File a 1 colonna: specificare dt")
                acc = np.array([x for r in data for x in r])
        fac = {"g": G_ACC, "m/s2": 1.0, "cm/s2": 0.01}[units.lower()]
        return cls(acc * fac, float(dt), name or path)

    @classmethod
    def synthetic(cls, pga_g: float = 0.2, duration: float = 30.0, dt: float = 0.01,
                  f_lo: float = 0.3, f_hi: float = 15.0, seed: int = 1) -> "Motion":
        """Moto sintetico (rumore filtrato + inviluppo) per prove e test."""
        rng = np.random.default_rng(seed)
        t = np.arange(0, duration, dt)
        w = rng.standard_normal(len(t))
        sos = signal.butter(4, [f_lo, f_hi], btype="band", fs=1.0 / dt, output="sos")
        a = signal.sosfilt(sos, w)
        env = (t / 3.0) ** 2 * np.exp(2.0 * (1.0 - t / 3.0))   # inviluppo tipo Saragoni-Hart
        a *= env
        a *= pga_g * G_ACC / np.max(np.abs(a))
        return cls(a, dt, "synthetic")


# =============================================================================
#  PUNTO DI INPUT / OUTPUT
# =============================================================================
@dataclass(frozen=True)
class Location:
    """layer: 1..N (1 = superficie, N = semispazio); kind: 'within' | 'outcrop'."""
    layer: int
    kind: str = "within"

    def __post_init__(self):
        if self.kind not in ("within", "outcrop"):
            raise ValueError("kind deve essere 'within' oppure 'outcrop'")

    def __str__(self):
        return f"strato {self.layer} ({self.kind})"


# =============================================================================
#  RISPOSTA DI SPETTRO ELASTICO
# =============================================================================
def response_spectrum(acc: np.ndarray, dt: float, periods: Sequence[float],
                      damping: float = 0.05) -> np.ndarray:
    """Pseudo-accelerazione spettrale Sa(T) [stesse unita' di acc].

    Sistema SDOF integrato con discretizzazione esatta 'first-order hold'
    (equivalente a Nigam-Jennings) tramite scipy.signal.cont2discrete.
    """
    out = np.zeros(len(periods))
    for i, T in enumerate(periods):
        if T <= 0:
            out[i] = np.max(np.abs(acc))
            continue
        w = 2.0 * math.pi / T
        num, den, _ = signal.cont2discrete(([-1.0], [1.0, 2 * damping * w, w * w]),
                                           dt, method="foh")
        x = signal.lfilter(np.squeeze(num), den, acc)
        out[i] = w * w * np.max(np.abs(x))
    return out


# =============================================================================
#  RISULTATI
# =============================================================================
@dataclass
class EQLResult:
    profile: SoilProfile
    motion: Motion
    input_loc: Location
    magnitude: float
    r_gamma: float
    converged: bool
    n_iter: int
    errors: List[float]
    # proprieta' finali per strato
    g_gmax: np.ndarray
    G: np.ndarray
    damping: np.ndarray
    vs_eq: np.ndarray
    gamma_max_pct: np.ndarray        # strati 1..N-1
    gamma_eff_pct: np.ndarray
    tau_max: np.ndarray              # kPa, strati 1..N-1
    # storia iterazioni
    history: List[Dict[str, np.ndarray]]
    # serie temporali in uscita {Location: acc[m/s2]}
    outputs: Dict[Location, np.ndarray]
    # tabelle su tutti gli strati
    pga_within: np.ndarray           # m/s2, sommita' di ogni strato
    pga_outcrop: np.ndarray
    freqs: np.ndarray
    transfer: Dict[Location, np.ndarray]   # funzione di trasferimento ampiezza complessa

    # ---------------------------------------------------------------- utilita'
    def acc(self, layer: int, kind: str = "within") -> np.ndarray:
        return self.outputs[Location(layer, kind)]

    def time(self) -> np.ndarray:
        return self.motion.time

    def spectrum(self, loc: Union[Location, Tuple[int, str]],
                 periods: Optional[Sequence[float]] = None, damping: float = 0.05):
        loc = loc if isinstance(loc, Location) else Location(*loc)
        if periods is None:
            periods = np.logspace(-2, 1, 80)
        sa = response_spectrum(self.outputs[loc], self.motion.dt, periods, damping)
        return np.asarray(periods), sa / G_ACC

    def fourier_amplitude(self, loc: Union[Location, Tuple[int, str]]):
        loc = loc if isinstance(loc, Location) else Location(*loc)
        a = self.outputs[loc]
        f = np.fft.rfftfreq(len(a), self.motion.dt)
        return f, np.abs(np.fft.rfft(a)) * self.motion.dt

    def to_csv(self, path: str, locs: Optional[Sequence[Location]] = None) -> None:
        """Salva t + tutte le storie temporali richieste (in g)."""
        locs = list(locs or self.outputs.keys())
        cols = [self.time()] + [self.outputs[l] / G_ACC for l in locs]
        hdr = "t[s]," + ",".join(f"L{l.layer}_{l.kind}[g]" for l in locs)
        np.savetxt(path, np.column_stack(cols), delimiter=",", header=hdr, comments="")

    def print_summary(self) -> None:
        N = self.profile.n
        z = self.profile.depths_top()
        print("=" * 92)
        print(f"ANALISI LINEARE EQUIVALENTE  -  input: {self.motion.name}, "
              f"{self.input_loc}")
        print(f"M = {self.magnitude:.1f}   R_gamma = {self.r_gamma:.3f}   "
              f"iterazioni = {self.n_iter}   "
              f"{'CONVERGENZA OK' if self.converged else 'NON CONVERGIUTA'}  "
              f"(err. finale = {self.errors[-1] * 100:.2f} %)")
        print("-" * 92)
        print(f"{'#':>3} {'z[m]':>7} {'h[m]':>6} {'Vs0':>7} {'Vs_eq':>7} {'G/Gmax':>7} "
              f"{'D[%]':>6} {'g_max[%]':>9} {'g_eff[%]':>9} {'tau_max':>8} "
              f"{'PGA_w[g]':>9} {'PGA_o[g]':>9}")
        for i, L in enumerate(self.profile.layers):
            last = i == N - 1
            h = "   inf" if last else f"{L.thickness:6.2f}"
            gm = "     -" if last else f"{self.gamma_max_pct[i]:9.4f}"
            ge = "     -" if last else f"{self.gamma_eff_pct[i]:9.4f}"
            tm = "     -" if last else f"{self.tau_max[i]:8.1f}"
            print(f"{i + 1:3d} {z[i]:7.2f} {h} {L.vs:7.1f} {self.vs_eq[i]:7.1f} "
                  f"{self.g_gmax[i]:7.3f} {self.damping[i] * 100:6.2f} {gm:>9} {ge:>9} "
                  f"{tm:>8} {self.pga_within[i] / G_ACC:9.4f} "
                  f"{self.pga_outcrop[i] / G_ACC:9.4f}")
        print("-" * 92)
        print(f"Input PGA = {self.motion.pga / G_ACC:.4f} g")
        for loc, a in self.outputs.items():
            print(f"Output {str(loc):<22} PGA = {np.max(np.abs(a)) / G_ACC:.4f} g")
        print("=" * 92)

    # ----------------------------------------------------------------- grafici
    def plot(self, show: bool = True, save: Optional[str] = None):
        """Cruscotto riassuntivo (matplotlib)."""
        import matplotlib.pyplot as plt
        from . import plots
        fig = plt.figure(figsize=(14, 9))
        plots.plot_dashboard(fig, self)
        if save:
            fig.savefig(save, dpi=130)
        if show:
            plt.show()
        return fig


# =============================================================================
#  MOTORE DI CALCOLO
# =============================================================================
class EquivalentLinearAnalysis:
    """Analisi lineare equivalente 1D nel dominio della frequenza."""

    def __init__(self, profile: SoilProfile):
        self.profile = profile
        self.profile.update_stresses()

    # ------------------------------------------------------------------
    #  Propagazione ondosa (Kramer 1996, eq. 7.x) con riscalamento
    #  per evitare overflow ad alta frequenza.
    # ------------------------------------------------------------------
    def _propagate(self, omega: np.ndarray, G: np.ndarray, beta: np.ndarray):
        """Calcola ampiezze normalizzate (a, b) alla sommita' di ogni strato,
        il logaritmo del fattore di scala L e i numeri d'onda complessi k.

        Ampiezze fisiche:  A_m = a_m * exp(L_m),  B_m = b_m * exp(L_m)
        con condizione di superficie libera A_1 = B_1 = 1.
        """
        P = self.profile
        N = P.n
        nf = len(omega)
        rho = np.array([L.rho for L in P.layers])
        h = np.array([L.thickness for L in P.layers[:-1]] + [0.0])

        # modulo complesso:  G* = G (1 - 2 b^2 + 2 i b sqrt(1-b^2))   (EERA)
        Gs = G * (1.0 - 2.0 * beta ** 2 + 2j * beta * np.sqrt(1.0 - beta ** 2))
        vs_c = np.sqrt(Gs / rho)                       # velocita' complessa
        imp = rho * vs_c                               # impedenza complessa

        k = omega[None, :] / vs_c[:, None]             # (N, nf)
        a = np.ones((N, nf), complex)
        b = np.ones((N, nf), complex)
        Ls = np.zeros((N, nf))

        for m in range(N - 1):
            alpha = imp[m] / imp[m + 1]
            ea = np.exp(1j * k[m] * h[m])
            eb = 1.0 / ea
            a_new = 0.5 * a[m] * (1 + alpha) * ea + 0.5 * b[m] * (1 - alpha) * eb
            b_new = 0.5 * a[m] * (1 - alpha) * ea + 0.5 * b[m] * (1 + alpha) * eb
            s = np.maximum(np.abs(a_new), np.abs(b_new))
            s[s == 0] = 1.0
            a[m + 1] = a_new / s
            b[m + 1] = b_new / s
            Ls[m + 1] = Ls[m] + np.log(s)
        return a, b, Ls, k, Gs

    @staticmethod
    def _disp(a, b, kind: str, m: int):
        """Spostamento normalizzato alla sommita' dello strato m (0-based)."""
        return (a[m] + b[m]) if kind == "within" else 2.0 * a[m]

    def _ratio(self, a, b, Ls, loc_ref: Location, loc_out: Location, eps=1e-12):
        """Funzione di trasferimento complessa  u_out / u_ref  (spostamento
        = accelerazione).  loc.layer e' 1-based."""
        j, i = loc_ref.layer - 1, loc_out.layer - 1
        d_ref = self._disp(a, b, loc_ref.kind, j)
        d_out = self._disp(a, b, loc_out.kind, i)
        d_ref = np.where(np.abs(d_ref) < eps, eps, d_ref)
        return d_out / d_ref * np.exp(Ls[i] - Ls[j])

    # ------------------------------------------------------------------
    def transfer_function(self, freqs: np.ndarray, in_loc: Location, out_loc: Location,
                          G: Optional[np.ndarray] = None,
                          damping: Optional[np.ndarray] = None) -> np.ndarray:
        """Funzione di trasferimento complessa tra due punti per il profilo
        con G e D dati (default: valori a piccole deformazioni)."""
        P = self.profile
        if G is None:
            G = np.array([L.gmax for L in P.layers])
        if damping is None:
            damping = np.array([self._small_strain_damping(L) for L in P.layers])
        w = 2 * np.pi * np.asarray(freqs, float)
        a, b, Ls, _, _ = self._propagate(w, G, damping)
        return self._ratio(a, b, Ls, in_loc, out_loc)

    @staticmethod
    def _small_strain_damping(L: Layer) -> float:
        if L.curve is None:
            return L.damping
        return float(L.curve.evaluate(np.array([1e-6]), L.sigma_m)[1][0])

    # ------------------------------------------------------------------
    def run(self, motion: Motion, input_loc: Location,
            output_locs: Sequence[Location] = (), magnitude: float = 7.0,
            r_gamma: Optional[float] = None, tol: float = 0.01, max_iter: int = 20,
            pad_time: float = 10.0, verbose: bool = True,
            progress=None) -> EQLResult:
        """Esegue l'analisi lineare equivalente.

        motion      : accelerogramma di input (assegnato in input_loc)
        input_loc   : Location(layer, 'within'|'outcrop')  -- tipico: (N, 'outcrop')
        output_locs : elenco di Location per cui restituire l'accelerogramma
        magnitude   : M per R_gamma = (M-1)/10   (ignorato se r_gamma assegnato)
        tol         : tolleranza relativa su G e D tra iterazioni successive
        pad_time    : secondi di zeri aggiunti in coda (evita aliasing temporale)
        progress    : funzione opzionale progress(iterazione, errore) a ogni passo
        """
        P = self.profile
        P.update_stresses()
        N = P.n
        if not (1 <= input_loc.layer <= N):
            raise ValueError(f"Strato di input fuori range 1..{N}")
        for lo in output_locs:
            if not (1 <= lo.layer <= N):
                raise ValueError(f"Strato di output {lo.layer} fuori range 1..{N}")
        R = r_gamma if r_gamma is not None else (magnitude - 1.0) / 10.0

        dt = motion.dt
        n = len(motion.acc)
        nfft = 1 << int(math.ceil(math.log2(n + int(round(pad_time / dt)))))
        freqs = np.fft.rfftfreq(nfft, dt)
        omega = 2 * np.pi * freqs
        acc_f = np.fft.rfft(motion.acc, nfft)
        with np.errstate(divide="ignore", invalid="ignore"):
            disp_f = np.where(omega > 0, -acc_f / omega ** 2, 0.0)   # u = -a/w^2

        gmax = np.array([L.gmax for L in P.layers])
        h = np.array([L.thickness for L in P.layers[:-1]])
        rho = np.array([L.rho for L in P.layers])

        # stato iniziale: G = Gmax, D = D a piccole deformazioni
        gg = np.ones(N)
        D = np.array([self._small_strain_damping(L) for L in P.layers])
        errors: List[float] = []
        history: List[Dict[str, np.ndarray]] = []
        converged = False

        for it in range(1, max_iter + 1):
            G = gg * gmax
            a, b, Ls, k, Gs = self._propagate(omega, G, D)

            # deformazione a meta' strato: gamma = du/dz  (z verso il basso)
            gam_max = np.zeros(N - 1)
            tau_max = np.zeros(N - 1)
            j = input_loc.layer - 1
            d_ref = self._disp(a, b, input_loc.kind, j)
            d_ref = np.where(np.abs(d_ref) < 1e-12, 1e-12, d_ref)
            for m in range(N - 1):
                zm = h[m] / 2.0
                dudz = 1j * k[m] * (a[m] * np.exp(1j * k[m] * zm)
                                    - b[m] * np.exp(-1j * k[m] * zm))
                r = dudz / d_ref * np.exp(Ls[m] - Ls[j])
                gam_f = r * disp_f
                gam_f[0] = 0.0
                gam_t = np.fft.irfft(gam_f, nfft)[:n]
                gam_max[m] = np.max(np.abs(gam_t))
                tau_t = np.fft.irfft(Gs[m] * gam_f, nfft)[:n]
                tau_max[m] = np.max(np.abs(tau_t))

            gam_eff_pct = R * gam_max * 100.0

            history.append(dict(g_gmax=gg.copy(), damping=D.copy(),
                                gamma_max_pct=gam_max * 100.0,
                                gamma_eff_pct=gam_eff_pct.copy()))

            # nuovi G/Gmax e D dalle curve
            gg_new, D_new = gg.copy(), D.copy()
            for m in range(N - 1):
                L = P.layers[m]
                if L.curve is not None:
                    g1, d1 = L.curve.evaluate(np.array([max(gam_eff_pct[m], 1e-7)]),
                                              L.sigma_m)
                    gg_new[m], D_new[m] = float(g1[0]), float(d1[0])
            err = max(np.max(np.abs(gg_new - gg) / gg), np.max(np.abs(D_new - D) / D))
            errors.append(float(err))
            if verbose:
                print(f"  iter {it:2d}: errore max = {err * 100:7.3f} %")
            if progress is not None:
                progress(it, float(err))
            gg, D = gg_new, D_new
            if err < tol:
                converged = True
                break

        # ---- analisi finale con le proprieta' convergenti (G, D finali)
        G = gg * gmax
        a, b, Ls, k, Gs = self._propagate(omega, G, D)

        outputs: Dict[Location, np.ndarray] = {}
        transfer: Dict[Location, np.ndarray] = {}
        for lo in output_locs:
            F = self._ratio(a, b, Ls, input_loc, lo)
            transfer[lo] = F
            outputs[lo] = np.fft.irfft(F * acc_f, nfft)[:n]

        pga_w = np.zeros(N)
        pga_o = np.zeros(N)
        for i in range(N):
            for kind, arr in (("within", pga_w), ("outcrop", pga_o)):
                F = self._ratio(a, b, Ls, input_loc, Location(i + 1, kind))
                arr[i] = np.max(np.abs(np.fft.irfft(F * acc_f, nfft)[:n]))

        # deformazione/tensione finali (ricalcolate con G, D convergenti)
        j = input_loc.layer - 1
        d_ref = self._disp(a, b, input_loc.kind, j)
        d_ref = np.where(np.abs(d_ref) < 1e-12, 1e-12, d_ref)
        gam_max = np.zeros(N - 1)
        tau_max = np.zeros(N - 1)
        for m in range(N - 1):
            zm = h[m] / 2.0
            dudz = 1j * k[m] * (a[m] * np.exp(1j * k[m] * zm) - b[m] * np.exp(-1j * k[m] * zm))
            gam_f = dudz / d_ref * np.exp(Ls[m] - Ls[j]) * disp_f
            gam_f[0] = 0.0
            gam_max[m] = np.max(np.abs(np.fft.irfft(gam_f, nfft)[:n]))
            tau_max[m] = np.max(np.abs(np.fft.irfft(Gs[m] * gam_f, nfft)[:n]))

        vs_eq = np.sqrt(G / rho)
        return EQLResult(
            profile=P, motion=motion, input_loc=input_loc, magnitude=magnitude,
            r_gamma=R, converged=converged, n_iter=len(errors), errors=errors,
            g_gmax=gg, G=G, damping=D, vs_eq=vs_eq,
            gamma_max_pct=gam_max * 100.0, gamma_eff_pct=R * gam_max * 100.0,
            tau_max=tau_max, history=history, outputs=outputs,
            pga_within=pga_w, pga_outcrop=pga_o, freqs=freqs, transfer=transfer)
