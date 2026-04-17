"""
Script d'entraînement principal — SE-RES-CNN 3D BraTS-Africa
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Usage :
    # Entraînement simple (un seul split)
    python scripts/train.py --config configs/config.yaml

    # K-fold cross-validation (recommandé avec 146 cas)
    python scripts/train.py --config configs/config.yaml --kfold

    # Sur Compute Canada (SLURM) — voir scripts/compute_canada.sh
    python scripts/train.py --config configs/config.yaml --kfold --wandb
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import SE_RES_CNN_3D
from src.data import BraTSAfricaDataset, get_train_transforms, get_val_transforms
from src.training import Trainer
from src.utils import load_config, set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement SE-RES-CNN 3D — BraTS-Africa")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="Chemin vers le fichier de configuration YAML")
    parser.add_argument("--kfold", action="store_true",
                        help="Utiliser K-fold cross-validation (recommandé)")
    parser.add_argument("--wandb", action="store_true",
                        help="Activer le logging Weights & Biases")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override du chemin vers le dataset")
    return parser.parse_args()


def build_model(cfg: dict) -> SE_RES_CNN_3D:
    model_cfg = cfg.get("model", {})
    return SE_RES_CNN_3D(
        in_channels=model_cfg.get("in_channels", 4),
        num_classes=model_cfg.get("num_classes", 3),
        encoder_channels=model_cfg.get("encoder_channels", [32, 64, 128]),
        bottleneck_ch=model_cfg.get("bottleneck_channels", 256),
        se_reduction=model_cfg.get("se_reduction", 16),
        dropout=model_cfg.get("dropout", 0.1),
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    seed = cfg.get("training", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()

    # Chemins
    data_dir = args.data_dir or cfg.get("data", {}).get("root_dir", "./data/brats_africa")
    results_dir = cfg.get("paths", {}).get("results", "./results")
    os.makedirs(results_dir, exist_ok=True)

    # Transforms
    patch_size = tuple(cfg.get("preprocessing", {}).get("patch_size", [128, 128, 128]))
    train_tf = get_train_transforms(patch_size=patch_size, cfg=cfg)
    val_tf = get_val_transforms(cfg=cfg)

    # Dataset
    dataset = BraTSAfricaDataset(
        root_dir=data_dir,
        cfg=cfg,
        train_transform=train_tf,
        val_transform=val_tf,
        seed=seed,
    )

    batch_size = cfg.get("training", {}).get("batch_size", 2)
    num_workers = cfg.get("preprocessing", {}).get("num_workers", 4)
    epochs = cfg.get("training", {}).get("epochs", 200)

    # ──────────────────────────────────────────────
    # MODE K-FOLD (recommandé avec 146 cas)
    # ──────────────────────────────────────────────
    if args.kfold:
        n_folds = cfg.get("cross_validation", {}).get("n_folds", 5)
        all_fold_results = []

        for fold, train_ds, val_ds in dataset.get_kfold_datasets(n_folds):
            print(f"\n{'#'*60}")
            print(f"# FOLD {fold + 1}/{n_folds}")
            print(f"{'#'*60}")

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True,
                collate_fn=BraTSAfricaDataset.collate_fn,
            )
            val_loader = DataLoader(
                val_ds, batch_size=1, shuffle=False,
                num_workers=num_workers, pin_memory=True,
                collate_fn=BraTSAfricaDataset.collate_fn,
            )

            model = build_model(cfg)
            trainer = Trainer(model, cfg, device, use_wandb=args.wandb, fold=fold)
            history = trainer.fit(train_loader, val_loader, epochs)

            best_val_dice = max(history["val_dice"])
            all_fold_results.append({"fold": fold, "best_val_dice": best_val_dice})
            print(f"Fold {fold+1} — Meilleur Dice Val: {best_val_dice:.4f}")

        # Résumé K-fold
        import numpy as np
        dices = [r["best_val_dice"] for r in all_fold_results]
        print(f"\n{'='*60}")
        print(f"RÉSULTATS K-FOLD ({n_folds} folds)")
        print(f"Mean Dice: {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        for r in all_fold_results:
            print(f"  Fold {r['fold']+1}: {r['best_val_dice']:.4f}")
        print(f"{'='*60}")

        with open(os.path.join(results_dir, "kfold_results.json"), "w") as f:
            json.dump({
                "mean_dice": float(np.mean(dices)),
                "std_dice": float(np.std(dices)),
                "folds": all_fold_results,
            }, f, indent=2)

    # ──────────────────────────────────────────────
    # MODE SIMPLE (split unique)
    # ──────────────────────────────────────────────
    else:
        train_loader = DataLoader(
            dataset.get_train_dataset(), batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            collate_fn=BraTSAfricaDataset.collate_fn,
        )
        val_loader = DataLoader(
            dataset.get_val_dataset(), batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=BraTSAfricaDataset.collate_fn,
        )

        model = build_model(cfg)
        trainer = Trainer(model, cfg, device, use_wandb=args.wandb, fold=0)
        trainer.fit(train_loader, val_loader, epochs)


if __name__ == "__main__":
    main()
