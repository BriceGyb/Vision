"""
Métriques d'évaluation — BraTS-Africa
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Métriques standardisées BraTS (requis pour comparaison à la littérature) :
  - Dice Score (DSC) — chevauchement prédiction / ground truth
  - Hausdorff Distance 95% (HD95) — précision des frontières tumorales
  - Sensibilité (Recall) — taux de vrais positifs
  - Spécificité — taux de vrais négatifs

Calculées par région : ET (Enhancing Tumor), TC (Tumor Core), WT (Whole Tumor).
"""

import torch
import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial.distance import directed_hausdorff


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Calcule le Dice Score (DSC) entre prédiction et ground truth binaires.

    DSC = 2 * |P ∩ G| / (|P| + |G|)

    Args:
        pred:   Masque prédit (binaire, numpy)
        target: Masque ground truth (binaire, numpy)
        smooth: Terme de lissage

    Returns:
        Score Dice entre 0 et 1
    """
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = (pred & target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calcule la Hausdorff Distance au 95e percentile (HD95).

    Mesure la précision des frontières tumorales — plus faible = meilleur.
    Retourne inf si l'un des masques est vide (cas sans tumeur).

    Args:
        pred:   Masque prédit (binaire, numpy)
        target: Masque ground truth (binaire, numpy)

    Returns:
        HD95 en voxels
    """
    if not pred.any() or not target.any():
        return float("inf")

    # Extraction des surfaces (voxels de bordure)
    pred_surface = _get_surface_points(pred)
    target_surface = _get_surface_points(target)

    if len(pred_surface) == 0 or len(target_surface) == 0:
        return float("inf")

    # Distances de Hausdorff dirigées
    d_pt = _directed_hausdorff_percentile(pred_surface, target_surface, 95)
    d_tp = _directed_hausdorff_percentile(target_surface, pred_surface, 95)

    return max(d_pt, d_tp)


def sensitivity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Sensibilité (Recall, True Positive Rate) = TP / (TP + FN)"""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    tp = (pred & target).sum()
    fn = (~pred & target).sum()
    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Spécificité (True Negative Rate) = TN / (TN + FP)"""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    tn = (~pred & ~target).sum()
    fp = (pred & ~target).sum()
    return (tn + smooth) / (tn + fp + smooth)


def _get_surface_points(mask: np.ndarray) -> np.ndarray:
    """Extrait les coordonnées des voxels de surface d'un masque 3D."""
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(mask, struct)
    surface = mask & ~eroded
    return np.argwhere(surface)


def _directed_hausdorff_percentile(points_a: np.ndarray, points_b: np.ndarray, pct: int) -> float:
    """Calcule le percentile `pct` des distances minimales de A vers B."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points_b)
    dists, _ = tree.query(points_a)
    return np.percentile(dists, pct)


class SegmentationMetrics:
    """
    Calculateur de métriques BraTS pour une passe d'évaluation complète.

    Usage :
        metrics = SegmentationMetrics()
        metrics.update(pred_logits, targets)
        results = metrics.compute()
        metrics.reset()
    """

    REGIONS = ["whole_tumor", "tumor_core", "enhancing_tumor"]
    REGION_IDX = {"whole_tumor": 0, "tumor_core": 1, "enhancing_tumor": 2}

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()

    def reset(self):
        self._results = {region: {"dice": [], "hd95": [], "sensitivity": [], "specificity": []}
                         for region in self.REGIONS}

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Calcule et accumule les métriques pour un batch.

        Args:
            logits:  (B, 3, D, H, W) — logits bruts du modèle
            targets: (B, 3, D, H, W) — masques ground truth multi-label
        """
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()

        for b in range(probs.shape[0]):
            for region, idx in self.REGION_IDX.items():
                pred = (probs[b, idx] > self.threshold)
                gt = targets_np[b, idx].astype(bool)

                self._results[region]["dice"].append(dice_score(pred, gt))
                self._results[region]["hd95"].append(hausdorff_distance_95(pred, gt))
                self._results[region]["sensitivity"].append(sensitivity(pred, gt))
                self._results[region]["specificity"].append(specificity(pred, gt))

    def compute(self) -> dict:
        """
        Retourne les métriques moyennées sur tous les cas accumulés.

        Returns:
            dict avec moyennes par région et moyenne globale Dice
        """
        results = {}
        all_dice = []

        for region in self.REGIONS:
            r = self._results[region]
            # Filtre les HD95 inf pour la moyenne (cas sans tumeur)
            hd95_finite = [v for v in r["hd95"] if np.isfinite(v)]

            results[region] = {
                "dice": float(np.mean(r["dice"])) if r["dice"] else 0.0,
                "hd95": float(np.mean(hd95_finite)) if hd95_finite else float("inf"),
                "sensitivity": float(np.mean(r["sensitivity"])) if r["sensitivity"] else 0.0,
                "specificity": float(np.mean(r["specificity"])) if r["specificity"] else 0.0,
            }
            all_dice.append(results[region]["dice"])

        results["mean_dice"] = float(np.mean(all_dice))
        return results

    def format_report(self, results: dict) -> str:
        """Génère un rapport texte lisible des métriques."""
        lines = ["\n" + "="*60, "MÉTRIQUES DE SEGMENTATION — BraTS-Africa", "="*60]
        for region in self.REGIONS:
            r = results[region]
            label = region.replace("_", " ").upper()
            lines.append(f"\n{label}:")
            lines.append(f"  Dice Score (DSC) : {r['dice']:.4f}")
            hd95_str = f"{r['hd95']:.2f}" if np.isfinite(r['hd95']) else "N/A"
            lines.append(f"  HD95             : {hd95_str} mm")
            lines.append(f"  Sensibilité      : {r['sensitivity']:.4f}")
            lines.append(f"  Spécificité      : {r['specificity']:.4f}")
        lines.append(f"\nMean Dice (WT/TC/ET) : {results['mean_dice']:.4f}")
        lines.append("="*60)
        return "\n".join(lines)
