# -*- coding: utf-8 -*-
"""Test di validazione del motore di calcolo.   Esecuzione:  python tests/test_core.py
(oppure:  pytest tests)"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eqlsite import *                                              # noqa: E402
from eqlsite.project import (demo_specs, load_project, profile_from_specs,   # noqa: E402
                             save_project, subdivide_specs, validate_specs)

F = np.linspace(0.01, 20, 2000)


def test_strato_su_base_rigida_soluzione_analitica():
    H, vs, xi = 20.0, 200.0, 0.05
    lay = [Layer(H, vs, 18.0, None, damping=xi), Layer(0, 1e5, 25.0, None, damping=0.0)]
    an = EquivalentLinearAnalysis(SoilProfile(lay))
    Fn = an.transfer_function(F, Location(2, "within"), Location(1, "within"))
    Gs = lay[0].gmax * (1 - 2 * xi ** 2 + 2j * xi * np.sqrt(1 - xi ** 2))
    k = 2 * np.pi * F / np.sqrt(Gs / lay[0].rho)
    Fa = 1 / np.cos(k * H)
    assert np.max(np.abs(np.abs(Fn) - np.abs(Fa)) / np.abs(Fa)) < 1e-9
    assert abs(F[np.argmax(np.abs(Fn))] - vs / (4 * H)) < 0.02          # picco a Vs/4H


def test_stessa_impedenza_outcrop_superficie_unitaria():
    lay = [Layer(10, 300, 20, None, damping=0.0), Layer(0, 300, 20, None, damping=0.0)]
    an = EquivalentLinearAnalysis(SoilProfile(lay))
    Fn = an.transfer_function(F, Location(2, "outcrop"), Location(1, "within"))
    assert np.max(np.abs(np.abs(Fn) - 1)) < 1e-9


def test_within_outcrop_coincidono_in_superficie():
    lay = [Layer(10, 150, 18, None, damping=0.03), Layer(5, 250, 19, None, damping=0.03),
           Layer(0, 700, 22, None, damping=0.01)]
    an = EquivalentLinearAnalysis(SoilProfile(lay))
    Fw = an.transfer_function(F, Location(3, "outcrop"), Location(1, "within"))
    Fo = an.transfer_function(F, Location(3, "outcrop"), Location(1, "outcrop"))
    assert np.max(np.abs(Fw - Fo)) < 1e-12


def test_reciprocita_input_output():
    lay = [Layer(10, 150, 18, None, damping=0.03), Layer(5, 250, 19, None, damping=0.03),
           Layer(0, 700, 22, None, damping=0.01)]
    an = EquivalentLinearAnalysis(SoilProfile(lay))
    a = an.transfer_function(F, Location(1, "within"), Location(3, "outcrop"))
    b = an.transfer_function(F, Location(3, "outcrop"), Location(1, "within"))
    assert np.max(np.abs(a * b - 1)) < 1e-9


def test_alta_frequenza_stabile():
    lay = [Layer(2, 80, 17, None, damping=0.1) for _ in range(30)] + \
          [Layer(0, 800, 22, None, damping=0.01)]
    an = EquivalentLinearAnalysis(SoilProfile(lay))
    Fn = an.transfer_function(np.linspace(0, 500, 5000), Location(31, "outcrop"), Location(1, "within"))
    assert np.all(np.isfinite(Fn))


def test_spettro_risposta_risonanza():
    dt = 0.005
    t = np.arange(0, 60, dt)
    sa = response_spectrum(np.sin(2 * np.pi * t / 0.5), dt, [0.5, 0.1, 2.0], 0.05)
    assert 8.0 < sa[0] <= 10.5 and sa[1] < sa[0] and sa[2] < sa[0]       # amplificazione ~ 1/(2 xi)


def test_eql_convergenza_e_coerenza():
    specs, wt = demo_specs()
    prof = profile_from_specs(specs, wt)
    N = prof.n
    moto = Motion.synthetic(pga_g=0.25, duration=30, dt=0.01)
    ris = EquivalentLinearAnalysis(prof).run(
        moto, Location(N, "outcrop"), [Location(1, "within"), Location(N, "outcrop")],
        magnitude=6.5, verbose=False)
    assert ris.converged
    # l'outcrop ricalcolato sul bedrock deve coincidere con l'input
    assert np.allclose(ris.outputs[Location(N, "outcrop")], moto.acc, atol=1e-6)
    assert np.all(ris.g_gmax[:-1] < 1.0) and np.all(ris.damping[:-1] > 0)
    assert ris.pga_within[0] > 0


def test_progetto_roundtrip():
    specs, wt = demo_specs()
    moto = Motion.synthetic(duration=5, dt=0.01)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p.json")
        save_project(p, specs, wt, moto, {"x": 1})
        s2, wt2, m2, a2 = load_project(p)
    assert s2 == specs and wt2 == wt and a2 == {"x": 1}
    assert np.allclose(m2.acc, moto.acc, atol=1e-6) and m2.dt == moto.dt
    assert validate_specs(s2) == []


def test_validazione_profilo():
    specs, _ = demo_specs(subdivided=False)
    specs[0]["thickness"] = -1
    assert any("spessore" in e for e in validate_specs(specs))
    assert len(subdivide_specs(demo_specs(subdivided=False)[0], 20, 5)) > 5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print(f"\n{len(tests)} test superati.")
