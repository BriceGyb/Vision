"""
Boucle d'entraînement — SE-RES-CNN 3D BraTS-Africa
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Gère :
  - Entraînement avec mixed precision (AMP) — économise mémoire GPU
  - Validation après chaque epoch
  - Early stopping (patience configurable)
  - Sauvegarde du meilleur modèle
  - Logging Weights & Biases (optionnel)
  - Gradient clipping pour stabilité
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .losses import CombinedDiceCELoss
from ..evaluation.metrics import SegmentationMetrics


class EarlyStopping:
    """Arrêt anticipé basé sur la métrique de validation."""

    def __init__(self, patience: int = 30, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Retourne True si le modèle s'est amélioré."""
        improved = (self.mode == "max" and value > self.best_value) or \
                   (self.mode == "min" and value < self.best_value)
        if improved:
            self.best_value = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


class Trainer:
    """
    Classe d'entraînement principale pour SE-RES-CNN 3D.

    Args:
        model:       Modèle PyTorch
        cfg:         Configuration (dict depuis config.yaml)
        device:      Device d'entraînement
        use_wandb:   Activer le logging W&B
        fold:        Numéro de fold (pour K-fold cross-validation)
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        use_wandb: bool = False,
        fold: int = 0,
    ):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.fold = fold
        self.use_wandb = use_wandb

        train_cfg = cfg.get("training", {})

        # Optimizer — AdamW (robuste au weight decay pour IRM)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get("optimizer", {}).get("lr", 1e-4),
            weight_decay=train_cfg.get("optimizer", {}).get("weight_decay", 1e-5),
        )

        # Scheduler — CosineAnnealingLR
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg.get("scheduler", {}).get("T_max", 200),
            eta_min=train_cfg.get("scheduler", {}).get("eta_min", 1e-6),
        )

        # Loss combinée Dice + BCE
        loss_cfg = train_cfg.get("loss", {})
        self.criterion = CombinedDiceCELoss(
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            ce_weight=loss_cfg.get("ce_weight", 0.5),
        )

        # Mixed precision (AMP) — économise ~50% VRAM
        self.use_amp = train_cfg.get("amp", True) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.grad_clip = train_cfg.get("gradient_clip", 1.0)
        self.patience = train_cfg.get("early_stopping", {}).get("patience", 30)
        self.early_stopping = EarlyStopping(patience=self.patience, mode="max")
        self.metrics = SegmentationMetrics()

        # Dossiers de sauvegarde
        self.checkpoint_dir = cfg.get("paths", {}).get("checkpoints", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}

    def train_epoch(self, loader: DataLoader) -> dict:
        """Effectue une epoch d'entraînement."""
        self.model.train()
        total_loss = 0.0
        total_dice = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc="  Train", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss_dict = self.criterion(logits, labels)
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            dice_val = 1.0 - loss_dict["dice_loss"].item()
            total_loss += loss.item()
            total_dice += dice_val
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{dice_val:.4f}"})

        return {
            "train_loss": total_loss / n_batches,
            "train_dice": total_dice / n_batches,
        }

    @torch.no_grad()
    def val_epoch(self, loader: DataLoader) -> dict:
        """Effectue une epoch de validation."""
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc="  Val  ", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss_dict = self.criterion(logits, labels)

            self.metrics.update(logits, labels)
            total_loss += loss_dict["loss"].item()
            n_batches += 1

        metric_results = self.metrics.compute()
        return {
            "val_loss": total_loss / n_batches,
            "val_dice": metric_results["mean_dice"],
            "metrics": metric_results,
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> dict:
        """
        Boucle d'entraînement principale.

        Args:
            train_loader: DataLoader d'entraînement
            val_loader:   DataLoader de validation
            epochs:       Nombre maximum d'epochs

        Returns:
            Historique d'entraînement
        """
        print(f"\n{'='*60}")
        print(f"Entraînement SE-RES-CNN 3D — Fold {self.fold} | {epochs} epochs max")
        print(f"Paramètres: {self.model.count_parameters():,}")
        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"{'='*60}\n")

        if self.use_wandb:
            import wandb
            wandb.init(
                project=self.cfg.get("logging", {}).get("project", "SE-RES-CNN-BraTS-Africa"),
                config=self.cfg,
                name=f"fold_{self.fold}",
            )

        best_dice = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.val_epoch(val_loader)

            self.scheduler.step()
            elapsed = time.time() - t0

            # Affichage
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {train_metrics['train_loss']:.4f} → {val_metrics['val_loss']:.4f} | "
                  f"Dice Val: {val_metrics['val_dice']:.4f} | "
                  f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                  f"{elapsed:.1f}s")

            # Détail métriques par région toutes les 10 epochs
            if epoch % self.cfg.get("logging", {}).get("log_interval", 10) == 0:
                report = self.metrics.format_report(val_metrics["metrics"])
                print(report)

            # W&B logging
            if self.use_wandb:
                import wandb
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics["train_loss"],
                    "train/dice": train_metrics["train_dice"],
                    "val/loss": val_metrics["val_loss"],
                    "val/mean_dice": val_metrics["val_dice"],
                    "lr": self.scheduler.get_last_lr()[0],
                }
                for region in ["whole_tumor", "tumor_core", "enhancing_tumor"]:
                    for metric in ["dice", "hd95", "sensitivity"]:
                        log_dict[f"val/{region}/{metric}"] = val_metrics["metrics"][region][metric]
                wandb.log(log_dict)

            # Sauvegarde du meilleur modèle
            improved = self.early_stopping.step(val_metrics["val_dice"])
            if improved:
                best_dice = val_metrics["val_dice"]
                path = os.path.join(self.checkpoint_dir, f"best_model_fold{self.fold}.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_dice": best_dice,
                    "cfg": self.cfg,
                }, path)
                print(f"  ✓ Meilleur modèle sauvegardé (Dice={best_dice:.4f})")

            # Historique
            self.history["train_loss"].append(train_metrics["train_loss"])
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_dice"].append(val_metrics["val_dice"])

            # Early stopping
            if self.early_stopping.should_stop:
                print(f"\nEarly stopping déclenché à l'epoch {epoch} "
                      f"(patience={self.patience}, meilleur Dice={best_dice:.4f})")
                break

        if self.use_wandb:
            import wandb
            wandb.finish()

        print(f"\nEntraînement terminé — Meilleur Dice Val: {best_dice:.4f}")
        return self.history
