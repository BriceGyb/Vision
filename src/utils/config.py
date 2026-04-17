"""
Utilitaires — Configuration et reproductibilité
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026
"""

import os
import random
import yaml
import torch
import numpy as np


def load_config(config_path: str) -> dict:
    """Charge la configuration depuis un fichier YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    """
    Fixe tous les seeds pour la reproductibilité complète.
    Seed 42 — standard dans le roadmap.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[INFO] Seed fixé à {seed} pour la reproductibilité")


def get_device() -> torch.device:
    """Retourne le device disponible (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] GPU détecté : {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Apple MPS détecté")
    else:
        device = torch.device("cpu")
        print("[WARN] Aucun GPU détecté — entraînement sur CPU (très lent pour IRM 3D)")
    return device
