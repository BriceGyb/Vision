"""
Test rapide de l'architecture SE-RES-CNN 3D — sans données réelles
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Vérifie que l'architecture est correcte (dimensions, forward pass)
avant d'avoir accès au dataset.

Usage : python scripts/test_architecture.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.models import SE_RES_CNN_3D
from src.training.losses import CombinedDiceCELoss
from src.evaluation.metrics import SegmentationMetrics


def test_forward_pass():
    print("="*60)
    print("TEST DE L'ARCHITECTURE SE-RES-CNN 3D")
    print("="*60)

    # Dimensions réduites pour test CPU (vraies : 240×240×155)
    B, C, D, H, W = 1, 4, 64, 64, 64

    model = SE_RES_CNN_3D(
        in_channels=4,
        num_classes=3,
        encoder_channels=[32, 64, 128],
        bottleneck_ch=256,
        se_reduction=16,
        dropout=0.1,
    )

    print(f"\nArchitecture créée avec succès")
    print(f"Paramètres entraînables : {model.count_parameters():,}")

    # Forward pass
    x = torch.randn(B, C, D, H, W)
    print(f"\nForward pass : {list(x.shape)} → ", end="", flush=True)

    model.eval()
    with torch.no_grad():
        out = model(x)

    print(f"{list(out.shape)}")
    assert out.shape == (B, 3, D, H, W), f"Forme de sortie incorrecte : {out.shape}"
    print("Forme de sortie correcte : (B, 3, D, H, W)")

    # Test de la loss
    print("\nTest de la loss (Dice + BCE)...", end=" ", flush=True)
    targets = (torch.rand(B, 3, D, H, W) > 0.7).float()
    criterion = CombinedDiceCELoss(dice_weight=0.5, ce_weight=0.5)
    loss_dict = criterion(out, targets)
    print(f"Loss totale = {loss_dict['loss'].item():.4f} | "
          f"Dice = {loss_dict['dice_loss'].item():.4f} | "
          f"BCE = {loss_dict['bce_loss'].item():.4f}")

    # Test backward
    print("Test backward pass...", end=" ", flush=True)
    model.train()
    out = model(x)
    loss_dict = criterion(out, targets)
    loss_dict["loss"].backward()
    print("OK — Gradients calculés correctement")

    # Test métriques
    print("\nTest des métriques BraTS...", end=" ", flush=True)
    metrics = SegmentationMetrics()
    model.eval()
    with torch.no_grad():
        out = model(x)
    metrics.update(out, targets)
    results = metrics.compute()
    print(f"OK — Mean Dice = {results['mean_dice']:.4f}")

    # Résumé architecture
    print("\n" + "="*60)
    print("RÉSUMÉ ARCHITECTURE")
    print("="*60)
    print(f"Input     : (B, 4, D, H, W)")
    print(f"Encoder 1 : → (B, 32, D/2, H/2, W/2)   [SE + Résiduel + MaxPool]")
    print(f"Encoder 2 : → (B, 64, D/4, H/4, W/4)   [SE + Résiduel + MaxPool]")
    print(f"Encoder 3 : → (B, 128, D/8, H/8, W/8)  [SE + Résiduel + MaxPool]")
    print(f"Bottleneck: → (B, 256, D/8, H/8, W/8)  [SE + Résiduel]")
    print(f"Decoder 1 : → (B, 128, D/4, H/4, W/4)  [UpSample + SE + Résiduel]")
    print(f"Decoder 2 : → (B, 64, D/2, H/2, W/2)   [UpSample + SE + Résiduel]")
    print(f"Decoder 3 : → (B, 32, D, H, W)          [UpSample + SE + Résiduel]")
    print(f"Output    : → (B, 3, D, H, W)           [Conv 1×1×1]")
    print(f"\nClasses sorties : [WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)]")
    print(f"\nTotal paramètres : {model.count_parameters():,}")

    print("\n" + "="*60)
    print("TOUS LES TESTS PASSENT — Architecture prête pour l'entraînement")
    print("="*60)
    print("\nProchaine étape : Télécharger BraTS-Africa")
    print("  python scripts/download_dataset.py")


if __name__ == "__main__":
    test_forward_pass()
