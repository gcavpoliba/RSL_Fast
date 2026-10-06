#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esempio di uso da script (senza interfaccia grafica).

    python examples/esempio_script.py

Profilo: 4 strati + bedrock; input = moto sintetico assegnato come OUTCROP sul
bedrock; output = superficie (within), strato intermedio (within) e bedrock
(within e outcrop).  Salva grafici e CSV nella cartella 'risultati_esempio'.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eqlsite import *                                    # noqa: E402
from eqlsite.project import demo_specs, profile_from_specs  # noqa: E402

specs, falda = demo_specs(subdivided=True)               # strati gia' suddivisi (fmax=20 Hz)
profilo = profile_from_specs(specs, water_table=falda)
print(profilo.table(), "\n")

N = profilo.n                                            # l'ultimo strato e' il semispazio
moto = Motion.synthetic(pga_g=0.25, duration=30, dt=0.01)
# moto = Motion.from_file("record.AT2", units="g")       # <- per usare un record reale

analisi = EquivalentLinearAnalysis(profilo)
ris = analisi.run(
    moto,
    input_loc=Location(N, "outcrop"),                    # input: affioramento sul bedrock
    output_locs=[Location(1, "within"),                  # superficie libera
                 Location(N // 2, "within"),             # profondita' intermedia
                 Location(N, "within"),                  # bedrock, moto interno
                 Location(N, "outcrop")],                # bedrock, affioramento
    magnitude=6.5)

ris.print_summary()

out = "risultati_esempio"
os.makedirs(out, exist_ok=True)
ris.to_csv(os.path.join(out, "storie_temporali.csv"))
ris.plot(show=False, save=os.path.join(out, "cruscotto.png"))
print(f"\nRisultati salvati in '{out}/'")
