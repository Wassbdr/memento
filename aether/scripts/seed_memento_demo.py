#!/usr/bin/env python3
"""
Initialise la démo « raisonnement proactif » (vision + mémoire relationnelle).

Pré-requis :
  - Photo `known_faces/Wassim.jpg` (ou `wassim.jpg`) pour l’encodage facial — le nom
    affiché sera le nom de fichier (ex. Wassim).
  - MEM0_API_KEY + mode réel, ou mock local.

Usage (depuis le dossier aether) :
  source ../.venv/bin/activate
  PYTHONPATH=. python scripts/seed_memento_demo.py
"""

from __future__ import annotations

import os
import sys

# aether/ as cwd for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import DEMO_USER_ID
from app.services.memory_agent import _get_mem0_client


RELATION_FACT = (
    "Wassim est le père de Léa, la petite-fille de Jean."
)


def main() -> None:
    client = _get_mem0_client()
    messages = [{"role": "user", "content": RELATION_FACT}]
    try:
        client.add(
            messages,
            user_id=DEMO_USER_ID,
            metadata={"category": "family"},
        )
    except TypeError:
        client.add(messages, user_id=DEMO_USER_ID)

    print("OK — fait relationnel enregistré dans Mem0 pour", DEMO_USER_ID)
    print("  →", RELATION_FACT)
    print()
    print("Visages : placez une photo de Wassim sous known_faces/ sous le nom « Wassim.jpg »")
    print("  (ou « wassim.jpg » → libellé Wassim). Puis POST /api/v1/faces/reload ou redémarrage API.")
    print()
    print("Étape vocale (exemple) : « Memento, rappelle-moi que Léa adore les pivoines. »")


if __name__ == "__main__":
    main()
