"""
Script d'évaluation — SE-RES-CNN 3D BraTS-Africa
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Évalue un checkpoint sauvegardé sur le jeu de test.
Génère le tableau de résultats pour le paper.

Usage :
    python scripts/evaluate.py \
        --config configs/config.yaml \
        --checkpoint checkpoints/best_model_fold0.pth
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SE_RES_CNN_3D
from src.data import BraTSAfricaDataset, get_val_transforms
from src.evaluation import SegmentationMetrics
from src.utils import load_config, set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation SE-RES-CNN 3D — BraTS-Africa")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Chemin vers le checkpoint .pth")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="results/test_results.json")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("training", {}).get("seed", 42))
    device = get_device()

    data_dir = args.data_dir or cfg.get("data", {}).get("root_dir", "./data/brats_africa")

    # Dataset test
    dataset = BraTSAfricaDataset(
        root_dir=data_dir, cfg=cfg,
        val_transform=get_val_transforms(cfg=cfg),
    )
    test_loader = DataLoader(
        dataset.get_test_dataset(), batch_size=1, shuffle=False,
        num_workers=2, collate_fn=BraTSAfricaDataset.collate_fn,
    )

    # Charger le modele
    model_cfg = cfg.get("model", {})
    model = SE_RES_CNN_3D(
        in_channels=model_cfg.get("in_channels", 4),
        num_classes=model_cfg.get("num_classes", 3),
        encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128]),
        bottleneck_ch=model_cfg.get("bottleneck_channels", 256),
        se_reduction=model_cfg.get("se_reduction", 16),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[INFO] Checkpoint charge — Epoch {checkpoint.get('epoch', '?')} | "
          f"Val Dice enregistre: {checkpoint.get('val_dice', 0.0):.4f}")
    print(f"[INFO] Parametres du modele : {model.count_parameters():,}")

    # Evaluation
    metrics = SegmentationMetrics()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            metrics.update(logits, labels)

    results = metrics.compute()
    report = metrics.format_report(results)
    print(report)

    # Sauvegarde
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Resultats sauvegardes dans {args.output}")


if __name__ == "__main__":
    main()
