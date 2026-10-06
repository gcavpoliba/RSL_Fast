# -*- coding: utf-8 -*-
"""
eqlsite/plots.py
Funzioni di grafico (matplotlib) condivise da GUI e uso da script.
Ogni funzione riceve una Figure e la ridisegna da zero.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .core import G_ACC, EQLResult, Layer, Motion, response_spectrum

PERIODS = np.logspace(-2, 1, 90)


def _label(loc) -> str:
    return f"strato {loc.layer} ({loc.kind})"


def _legend(ax, n: int, inside_max: int = 8, fontsize: int = 8):
    """Legenda compatta: dentro il grafico se poche curve, fuori a destra se molte."""
    if n <= inside_max:
        ax.legend(fontsize=fontsize)
    else:
        ax.legend(fontsize=6, ncol=1 if n <= 22 else 2, loc="upper left",
                  bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)


# ----------------------------------------------------------------------------
#  Moto di input
# ----------------------------------------------------------------------------
def plot_motion(fig, motion: Optional[Motion]):
    fig.clear()
    if motion is None:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Nessun accelerogramma caricato", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
        ax.set_axis_off()
        return
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(motion.time, motion.acc / G_ACC, lw=0.7, color="C0")
    ax.set(xlabel="t [s]", ylabel="a [g]", title=f"Accelerogramma di input - {motion.name}")
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    f = np.fft.rfftfreq(len(motion.acc), motion.dt)
    A = np.abs(np.fft.rfft(motion.acc)) * motion.dt
    ax.loglog(f[1:], A[1:] / G_ACC, lw=0.7)
    ax.set(xlabel="f [Hz]", ylabel="FAS [g s]", title="Spettro di Fourier")
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 1])
    sa = response_spectrum(motion.acc, motion.dt, PERIODS) / G_ACC
    ax.semilogx(PERIODS, sa)
    ax.set(xlabel="T [s]", ylabel="Sa (5%) [g]", title="Spettro di risposta elastico")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()


# ----------------------------------------------------------------------------
#  Risultati
# ----------------------------------------------------------------------------
def plot_accelerograms(fig, res: EQLResult):
    fig.clear()
    locs = list(res.outputs.keys())
    if len(locs) > 6:                       # molti output: un solo asse con curve sovrapposte
        ax = fig.add_subplot(111)
        ax.plot(res.time(), res.motion.acc / G_ACC, color="0.5", lw=1.0,
                label=f"input {_label(res.input_loc)}")
        for loc in locs:
            ax.plot(res.time(), res.outputs[loc] / G_ACC, lw=0.6, label=_label(loc))
        ax.set(xlabel="t [s]", ylabel="a [g]", title="Accelerogrammi")
        ax.grid(alpha=0.3)
        _legend(ax, len(locs) + 1, inside_max=0)
        fig.tight_layout()
        return
    n = len(locs) + 1
    axes = fig.subplots(n, 1, sharex=True, squeeze=False)[:, 0]
    axes[0].plot(res.time(), res.motion.acc / G_ACC, color="0.4", lw=0.7)
    axes[0].set_ylabel("a [g]")
    axes[0].set_title(f"INPUT - {_label(res.input_loc)}   PGA = {res.motion.pga / G_ACC:.4f} g",
                      fontsize=9, loc="left")
    for ax, loc in zip(axes[1:], locs):
        a = res.outputs[loc]
        ax.plot(res.time(), a / G_ACC, lw=0.7, color="C0" if loc.kind == "within" else "C3")
        ax.set_ylabel("a [g]")
        ax.set_title(f"{_label(loc)}   PGA = {np.max(np.abs(a)) / G_ACC:.4f} g",
                     fontsize=9, loc="left")
    for ax in axes:
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()


def plot_spectra(fig, res: EQLResult, damping: float = 0.05):
    fig.clear()
    ax = fig.add_subplot(111)
    sa_in = response_spectrum(res.motion.acc, res.motion.dt, PERIODS, damping) / G_ACC
    ax.semilogx(PERIODS, sa_in, color="0.5", lw=1.6, label=f"input {_label(res.input_loc)}")
    for loc, a in res.outputs.items():
        sa = response_spectrum(a, res.motion.dt, PERIODS, damping) / G_ACC
        ax.semilogx(PERIODS, sa, lw=1.3, label=_label(loc))
    ax.set(xlabel="T [s]", ylabel=f"Sa ({damping * 100:.0f}% smorz.) [g]",
           title="Spettri di risposta elastici")
    _legend(ax, len(res.outputs) + 1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()


def plot_transfer(fig, res: EQLResult):
    fig.clear()
    ax = fig.add_subplot(111)
    fmax = min(30.0, res.freqs[-1])
    for loc, F in res.transfer.items():
        ax.semilogx(res.freqs[1:], np.abs(F[1:]), lw=1.1,
                    label=f"{_label(res.input_loc)} -> {_label(loc)}")
    ax.set(xlabel="f [Hz]", ylabel="|F| = out / in", xlim=(0.1, fmax),
           title="Funzioni di trasferimento (proprieta' strain-compatibili)")
    _legend(ax, len(res.transfer))
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()


def plot_fourier(fig, res: EQLResult):
    fig.clear()
    ax = fig.add_subplot(111)
    f = np.fft.rfftfreq(len(res.motion.acc), res.motion.dt)
    A = np.abs(np.fft.rfft(res.motion.acc)) * res.motion.dt / G_ACC
    ax.loglog(f[1:], A[1:], color="0.5", lw=1.0, label=f"input {_label(res.input_loc)}")
    for loc in res.outputs:
        f, A = res.fourier_amplitude(loc)
        ax.loglog(f[1:], A[1:] / G_ACC, lw=0.9, label=_label(loc))
    ax.set(xlabel="f [Hz]", ylabel="FAS [g s]", xlim=(0.1, 50),
           title="Spettri di ampiezza di Fourier")
    _legend(ax, len(res.outputs) + 1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()


def _layer_mid_depths(res: EQLResult):
    zt = res.profile.depths_top()
    h = np.array([L.thickness for L in res.profile.layers[:-1]])
    return zt[:-1] + h / 2.0, zt


def plot_profiles(fig, res: EQLResult):
    fig.clear()
    z_mid, zt = _layer_mid_depths(res)
    axes = fig.subplots(1, 5, sharey=True)
    axes[0].plot(res.gamma_max_pct, z_mid, "o-", ms=3)
    axes[0].set_xlabel("gamma max [%]")
    axes[0].set_ylabel("profondita' [m]")
    axes[1].plot(res.g_gmax[:-1], z_mid, "o-", ms=3)
    axes[1].set_xlabel("G/Gmax")
    axes[2].plot(res.damping[:-1] * 100, z_mid, "o-", ms=3)
    axes[2].set_xlabel("D [%]")
    axes[3].plot(res.tau_max, z_mid, "o-", ms=3)
    axes[3].set_xlabel("tau max [kPa]")
    axes[4].step(res.pga_within / G_ACC, zt, where="post", label="within")
    axes[4].step(res.pga_outcrop / G_ACC, zt, where="post", ls="--", label="outcrop")
    axes[4].set_xlabel("PGA [g]")
    axes[4].legend(fontsize=7)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Profili con la profondita'", fontsize=10)
    fig.tight_layout()


def plot_vs_profile(fig, res: EQLResult):
    fig.clear()
    ax = fig.add_subplot(111)
    zt = res.profile.depths_top()
    vs0 = np.array([L.vs for L in res.profile.layers])
    zb = np.concatenate([zt[1:], [zt[-1] + max(5.0, 0.2 * zt[-1])]])
    for i in range(len(vs0)):
        ax.plot([vs0[i], vs0[i]], [zt[i], zb[i]], "C0-", lw=2,
                label="Vs iniziale" if i == 0 else None)
        ax.plot([res.vs_eq[i], res.vs_eq[i]], [zt[i], zb[i]], "C3--", lw=2,
                label="Vs strain-compatibile" if i == 0 else None)
    ax.invert_yaxis()
    ax.set(xlabel="Vs [m/s]", ylabel="profondita' [m]", title="Profilo di Vs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()


def plot_convergence(fig, res: EQLResult):
    fig.clear()
    ax1 = fig.add_subplot(121)
    it = np.arange(1, len(res.errors) + 1)
    ax1.semilogy(it, np.maximum(np.array(res.errors) * 100, 1e-6), "o-")
    ax1.axhline(1.0, color="0.6", ls=":")
    ax1.set(xlabel="iterazione", ylabel="errore max [%]", title="Convergenza")
    ax1.grid(alpha=0.3, which="both")
    ax2 = fig.add_subplot(122)
    z_mid, _ = _layer_mid_depths(res)
    for k, h in enumerate(res.history):
        ax2.plot(h["g_gmax"][:-1], z_mid, lw=0.9, color=f"C{k % 10}",
                 alpha=0.35 + 0.65 * (k + 1) / len(res.history),
                 label=f"it. {k + 1}" if k in (0, len(res.history) - 1) else None)
    ax2.invert_yaxis()
    ax2.set(xlabel="G/Gmax", ylabel="profondita' [m]", title="G/Gmax per iterazione")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()


def plot_dashboard(fig, res: EQLResult):
    """Cruscotto riassuntivo (usato da EQLResult.plot)."""
    fig.clear()
    gs = fig.add_gridspec(3, 4)
    zt = res.profile.depths_top()
    z_mid, _ = _layer_mid_depths(res)

    ax = fig.add_subplot(gs[0, :3])
    ax.plot(res.time(), res.motion.acc / G_ACC, color="0.6", lw=0.8,
            label=f"input {_label(res.input_loc)}")
    for loc, a in res.outputs.items():
        ax.plot(res.time(), a / G_ACC, lw=0.8, label=_label(loc))
    ax.set(xlabel="t [s]", ylabel="a [g]", title="Accelerogrammi")
    if len(res.outputs) <= 8:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 3])
    ax.step(res.pga_within / G_ACC, zt, where="post", label="within")
    ax.step(res.pga_outcrop / G_ACC, zt, where="post", label="outcrop", ls="--")
    ax.invert_yaxis()
    ax.set(xlabel="PGA [g]", ylabel="z [m]", title="PGA")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, :2])
    sa_in = response_spectrum(res.motion.acc, res.motion.dt, PERIODS) / G_ACC
    ax.loglog(PERIODS, sa_in, color="0.6", label="input")
    for loc, a in res.outputs.items():
        ax.loglog(PERIODS, response_spectrum(a, res.motion.dt, PERIODS) / G_ACC,
                  label=_label(loc))
    ax.set(xlabel="T [s]", ylabel="Sa (5%) [g]", title="Spettri di risposta")
    if len(res.outputs) <= 8:
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = fig.add_subplot(gs[1, 2:])
    for loc, F in res.transfer.items():
        ax.semilogx(res.freqs[1:], np.abs(F[1:]), lw=0.8, label=_label(loc))
    ax.set(xlabel="f [Hz]", ylabel="|F|", title="Funzioni di trasferimento",
           xlim=(0.1, min(30, res.freqs[-1])))
    if len(res.transfer) <= 8:
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    for col, (y, xl, tt) in enumerate([
            (res.gamma_max_pct, "gamma_max [%]", "Deformazione max"),
            (res.g_gmax[:-1], "G/Gmax", "Degrado di rigidezza"),
            (res.damping[:-1] * 100, "D [%]", "Smorzamento"),
            (res.tau_max, "tau_max [kPa]", "Taglio massimo")]):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(y, z_mid, "o-", ms=3)
        ax.invert_yaxis()
        ax.set(xlabel=xl, title=tt, ylabel="z [m]" if col == 0 else "")
        ax.grid(alpha=0.3)
    fig.tight_layout()


# ----------------------------------------------------------------------------
#  Curve di un singolo strato
# ----------------------------------------------------------------------------
def plot_layer_curves(fig, layers: Sequence[Layer], selected: Optional[Sequence[int]] = None):
    """G/Gmax e D in funzione di gamma per gli strati indicati (0-based)."""
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    g = np.logspace(-4, 1, 200)
    idx = list(range(len(layers) - 1)) if selected is None else list(selected)
    shown = 0
    for i in idx:
        L = layers[i]
        if L.curve is None:
            continue
        gg, d = L.curve.evaluate(g, L.sigma_m)
        lab = f"{i + 1} {L.name}".strip()
        ax1.semilogx(g, gg, label=lab)
        ax2.semilogx(g, d * 100, label=lab)
        shown += 1
    ax1.set(xlabel="gamma [%]", ylabel="G/Gmax", ylim=(0, 1.05), title="Degrado di rigidezza")
    ax2.set(xlabel="gamma [%]", ylabel="D [%]", title="Smorzamento")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3, which="both")
        if shown:
            ax.legend(fontsize=7)
    fig.tight_layout()
