#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Avvio dell'interfaccia grafica:  python run_gui.py"""
import sys

try:
    import tkinter  # noqa: F401
except ImportError:
    sys.exit("Tkinter non e' installato.\n"
             "  Linux (Debian/Ubuntu):  sudo apt install python3-tk\n"
             "  Windows/macOS: reinstallare Python da python.org includendo 'tcl/tk'.")

from eqlsite.gui import main

if __name__ == "__main__":
    main()
