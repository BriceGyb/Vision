"""
Fonctions de perte — BraTS-Africa
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Combinaison Dice Loss + Cross-Entropy Loss (standard BraTS).
Dice Loss : optimise directement le score de chevauchement (métrique principale).
CE Loss   : stabilise l'entraînement, pénalise les erreurs pixel-à-pixel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss pour segmentation médicale multi-label.

    Formule : DiceLoss = 1 - (2 * |P ∩ G|) / (|P| + |G|)

    Args:
        smooth: Terme de lissage pour éviter la division par zéro
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  (B, C, D, H, W) — logits bruts du modèle
            targets: (B, C, D, H, W) — masques binaires par région
        """
        probs = torch.sigmoid(logits)   # Multi-label → sigmoid (pas softmax)

        # Calcul par région, moyenne sur batch
        dims = (0, 2, 3, 4)             # Moyenne sur batch et dimensions spatiales
        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_per_class.mean()


class CombinedDiceCELoss(nn.Module):
    """
    Perte combinée Dice + Binary Cross-Entropy (standard BraTS).

    Formule : L = alpha * L_Dice + beta * L_BCE

    Args:
        dice_weight: Poids de la Dice Loss (défaut 0.5)
        ce_weight:   Poids de la BCE Loss (défaut 0.5)
        smooth:      Terme de lissage pour Dice
    """
    def __init__(
        self,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict:
        """
        Args:
            logits:  (B, C, D, H, W)
            targets: (B, C, D, H, W) — masques binaires multi-label

        Returns:
            dict avec loss totale et composantes individuelles
        """
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets.float())
        total = self.dice_weight * dice + self.ce_weight * bce
        return {"loss": total, "dice_loss": dice, "bce_loss": bce}
