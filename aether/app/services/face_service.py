"""
Service de reconnaissance faciale pour Memento Guardian.

Identifie les visages depuis un dossier known_faces/ et retourne
les correspondances avec niveau de confiance.

Requires: pip install face-recognition numpy Pillow
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import logger

# Lazy imports for optional dependencies
_numpy: Any = None
_face_recognition: Any = None
_PIL_Image: Any = None
FACE_RECOGNITION_AVAILABLE = False


def _init_face_libs() -> bool:
    """Lazy-load face recognition libraries."""
    global _numpy, _face_recognition, _PIL_Image, FACE_RECOGNITION_AVAILABLE

    if FACE_RECOGNITION_AVAILABLE:
        return True

    try:
        import numpy as np
        import face_recognition
        from PIL import Image

        _numpy = np
        _face_recognition = face_recognition
        _PIL_Image = Image
        FACE_RECOGNITION_AVAILABLE = True
        return True
    except ImportError as e:
        logger.warning(f"face_recognition non disponible: {e}. Reconnaissance desactivee.")
        return False


def _default_known_faces_dir() -> Path:
    """Prefer repo-root ``known_faces/`` (sibling of ``aether/``), else ``aether/known_faces``."""
    aether_root = Path(__file__).resolve().parents[2]
    repo_candidate = aether_root.parent / "known_faces"
    if repo_candidate.is_dir():
        return repo_candidate
    return aether_root / "known_faces"


# Dossier contenant les visages connus (format: nom_personne.jpg)
KNOWN_FACES_DIR = Path(os.getenv("KNOWN_FACES_DIR", str(_default_known_faces_dir())))

_known_faces: dict[str, Any] = {}  # name -> encoding (numpy array)
_faces_lock = Lock()
_initialized = False


def _load_known_faces() -> None:
    """Charge tous les visages connus depuis le dossier."""
    global _initialized, _known_faces

    if not _init_face_libs():
        return

    if not KNOWN_FACES_DIR.exists():
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dossier {KNOWN_FACES_DIR} cree. Ajoutez des photos nom_personne.jpg")
        _initialized = True
        return

    with _faces_lock:
        _known_faces.clear()

        for image_path in KNOWN_FACES_DIR.glob("*"):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            try:
                # Le nom du fichier devient le nom de la personne
                person_name = image_path.stem.replace("_", " ").title()

                image = _face_recognition.load_image_file(str(image_path))
                encodings = _face_recognition.face_encodings(image)

                if encodings:
                    _known_faces[person_name] = encodings[0]
                    logger.info(f"Visage charge: {person_name}")
                else:
                    logger.warning(f"Aucun visage detecte dans {image_path}")

            except Exception as e:
                logger.error(f"Erreur chargement {image_path}: {e}")

        _initialized = True
        logger.info(f"Reconnaissance faciale initialisee avec {len(_known_faces)} visages")


def reload_known_faces() -> int:
    """Force le rechargement des visages connus. Retourne le nombre charge."""
    global _initialized
    _initialized = False
    _load_known_faces()
    return len(_known_faces)


def detect_faces_in_frame(image_data: bytes | str) -> list[dict[str, Any]]:
    """
    Detecte et identifie les visages dans une image.

    Args:
        image_data: Image en bytes ou base64 string (avec ou sans data URL prefix)

    Returns:
        Liste de visages detectes avec:
        - identified: bool
        - name: str
        - confidence: float (0-1)
        - bounding_box: dict avec top, right, bottom, left
    """
    global _initialized

    if not _init_face_libs():
        return []

    if not _initialized:
        _load_known_faces()

    # Decode l'image
    if isinstance(image_data, str):
        # Gere le format data URL (data:image/jpeg;base64,...)
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"Erreur decodage base64: {e}")
            return []
    else:
        image_bytes = image_data

    try:
        pil_image = _PIL_Image.open(io.BytesIO(image_bytes))

        # Convertit en RGB si necessaire (RGBA, grayscale, etc.)
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        # Reduit la taille pour performance (max 640px)
        max_size = 640
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size))

        image_array = _numpy.array(pil_image)

    except Exception as e:
        logger.error(f"Erreur decodage image: {e}")
        return []

    # Detection des visages (model="hog" est plus rapide, "cnn" plus precis)
    try:
        face_locations = _face_recognition.face_locations(image_array, model="hog")
    except Exception as e:
        logger.error(f"Erreur detection visages: {e}")
        return []

    if not face_locations:
        return []

    try:
        face_encodings = _face_recognition.face_encodings(image_array, face_locations)
    except Exception as e:
        logger.error(f"Erreur encodage visages: {e}")
        return []

    results: list[dict[str, Any]] = []

    with _faces_lock:
        known_names = list(_known_faces.keys())
        known_encodings = list(_known_faces.values())

    for location, encoding in zip(face_locations, face_encodings):
        top, right, bottom, left = location

        detected: dict[str, Any] = {
            "identified": False,
            "name": "Inconnu",
            "confidence": 0.0,
            "bounding_box": {
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "left": int(left),
            },
        }

        if known_encodings:
            # Compare avec les visages connus
            distances = _face_recognition.face_distance(known_encodings, encoding)

            if len(distances) > 0:
                best_match_idx = _numpy.argmin(distances)
                best_distance = float(distances[best_match_idx])

                # Seuil de reconnaissance (0.6 = bon equilibre precision/rappel)
                if best_distance < 0.6:
                    detected["identified"] = True
                    detected["name"] = known_names[best_match_idx]
                    detected["confidence"] = round(1 - best_distance, 2)

        results.append(detected)

    return results


def add_known_face(name: str, image_data: bytes | str) -> bool:
    """
    Ajoute un nouveau visage a la base de connaissances.

    Args:
        name: Nom de la personne
        image_data: Image en bytes ou base64

    Returns:
        True si succes, False sinon
    """
    if not _init_face_libs():
        return False

    # Decode l'image
    if isinstance(image_data, str):
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logger.error(f"Erreur decodage base64 pour {name}: {e}")
            return False
    else:
        image_bytes = image_data

    try:
        pil_image = _PIL_Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        image_array = _numpy.array(pil_image)
        encodings = _face_recognition.face_encodings(image_array)

        if not encodings:
            logger.warning(f"Aucun visage detecte pour {name}")
            return False

        # Cree le dossier si necessaire
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

        # Sauvegarde l'image
        safe_name = name.lower().replace(" ", "_")
        image_path = KNOWN_FACES_DIR / f"{safe_name}.jpg"
        pil_image.save(str(image_path), "JPEG", quality=90)

        # Met a jour le cache en memoire
        with _faces_lock:
            _known_faces[name] = encodings[0]

        logger.info(f"Nouveau visage ajoute: {name} -> {image_path}")
        return True

    except Exception as e:
        logger.error(f"Erreur ajout visage {name}: {e}")
        return False


def remove_known_face(name: str) -> bool:
    """Supprime un visage de la base de connaissances."""
    safe_name = name.lower().replace(" ", "_")

    # Supprime du cache
    with _faces_lock:
        # Cherche la cle avec correspondance insensible a la casse
        key_to_remove = None
        for key in _known_faces:
            if key.lower().replace(" ", "_") == safe_name:
                key_to_remove = key
                break

        if key_to_remove:
            del _known_faces[key_to_remove]

    # Supprime le fichier
    for ext in (".jpg", ".jpeg", ".png"):
        image_path = KNOWN_FACES_DIR / f"{safe_name}{ext}"
        if image_path.exists():
            try:
                image_path.unlink()
                logger.info(f"Visage supprime: {name}")
                return True
            except Exception as e:
                logger.error(f"Erreur suppression {image_path}: {e}")
                return False

    return key_to_remove is not None


def get_known_faces_list() -> list[str]:
    """Retourne la liste des personnes connues."""
    if not _initialized:
        _load_known_faces()
    with _faces_lock:
        return list(_known_faces.keys())


def is_available() -> bool:
    """Verifie si le service de reconnaissance faciale est disponible."""
    return _init_face_libs()
