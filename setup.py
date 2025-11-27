#!/usr/bin/env python3
"""
Setup script per retrocompatibilità.

Usa pyproject.toml per la configurazione moderna.
Questo file esiste solo per supportare:
    pip install .
    
su sistemi che non supportano ancora PEP 517/518.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()

