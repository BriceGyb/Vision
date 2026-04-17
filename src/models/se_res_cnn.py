"""
SE-RES-CNN 3D — Architecture complète pour segmentation de tumeurs cérébrales
BraTS-Africa | Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Architecture U-Net avec blocs Squeeze-and-Excitation + Résiduel à chaque niveau.
Inspiré de Li et al. (2024) — SE-RES-CNN for Sports Image Classification —
adapté à l'imagerie médicale 3D.

Références:
  - He et al. (2015) — Deep Residual Learning. CVPR 2016.
  - Hu et al. (2018) — Squeeze-and-Excitation Networks. CVPR 2018.
  - Ronneberger et al. (2015) — U-Net. MICCAI 2015.
  - Li et al. (2024) — SE-RES-CNN. Scientific Reports 14, 19087.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# BRIQUE 2 — Module Squeeze-and-Excitation 3D
# Hu et al. (2018) — adapté aux features 3D
# ─────────────────────────────────────────────────────────────
class SEModule3D(nn.Module):
    """
    Module Squeeze-and-Excitation pour tenseurs 3D.

    Comprime (squeeze) les features spatiales par Global Average Pooling,
    puis recalibre (excitation) l'importance de chaque canal via MLP.

    Args:
        channels: Nombre de canaux d'entrée C
        reduction: Ratio de réduction (défaut 16 → C/16 canaux cachés)
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool3d(1)        # (B, C, D, H, W) → (B, C, 1, 1, 1)
        self.excitation = nn.Sequential(
            nn.Flatten(),                              # (B, C)
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        scale = self.squeeze(x)                        # (B, C, 1, 1, 1)
        scale = self.excitation(scale)                 # (B, C)
        scale = scale.view(b, c, 1, 1, 1)             # broadcast spatial
        return x * scale


# ─────────────────────────────────────────────────────────────
# BRIQUE 1 — Bloc Résiduel 3D (ResNet-style)
# He et al. (2015)
# ─────────────────────────────────────────────────────────────
class ResidualBlock3D(nn.Module):
    """
    Bloc résiduel 3D avec connexion skip.

    Chemin principal : Conv3D(3×3×3) → BN → ReLU → Conv3D(3×3×3) → BN
    Skip connection  : projection 1×1×1 si nécessaire (changement de canaux)
    Fusion           : addition + ReLU

    Args:
        in_channels:  Canaux d'entrée
        out_channels: Canaux de sortie
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_path = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        # Projection skip si changement de dimension
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv_path(x) + self.skip(x))


# ─────────────────────────────────────────────────────────────
# Bloc combiné SE + Résiduel (brique principale du modèle)
# ─────────────────────────────────────────────────────────────
class SEResBlock3D(nn.Module):
    """
    Bloc combiné : Résiduel 3D → SE Module 3D

    L'ordre résiduel → SE permet au module d'attention de pondérer
    les features déjà enrichies par la connexion skip.
    """
    def __init__(self, in_channels: int, out_channels: int, se_reduction: int = 16):
        super().__init__()
        self.residual = ResidualBlock3D(in_channels, out_channels)
        self.se = SEModule3D(out_channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.residual(x))


# ─────────────────────────────────────────────────────────────
# Bloc Encoder : SEResBlock + MaxPool
# ─────────────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, se_reduction: int = 16):
        super().__init__()
        self.block = SEResBlock3D(in_channels, out_channels, se_reduction)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        features = self.block(x)   # Sauvegardé pour skip connection
        pooled = self.pool(features)
        return features, pooled


# ─────────────────────────────────────────────────────────────
# Bloc Decoder : Upsample + Concat skip + SEResBlock
# ─────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, se_reduction: int = 16):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.block = SEResBlock3D(in_channels + skip_channels, out_channels, se_reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Ajustement taille si nécessaire (dimensions impaires après pooling)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


# ─────────────────────────────────────────────────────────────
# ARCHITECTURE COMPLÈTE — SE-RES-CNN 3D (U-Net style)
# ─────────────────────────────────────────────────────────────
class SE_RES_CNN_3D(nn.Module):
    """
    SE-RES-CNN 3D : Architecture U-Net avec blocs SE + Résiduel.

    Architecture complète selon le roadmap :
        Input  240×240×155×4
        Enc1   120×120×77×32   (Conv3D + SE + Résiduel + MaxPool)
        Enc2   60×60×38×64     (Conv3D + SE + Résiduel + MaxPool)
        Enc3   30×30×19×128    (Conv3D + SE + Résiduel + MaxPool)
        BN     30×30×19×256    (Conv3D + SE + Résiduel)
        Dec1   60×60×38×128    (UpSample + Conv3D + SE + Résiduel)
        Dec2   120×120×77×64   (UpSample + Conv3D + SE + Résiduel)
        Dec3   240×240×155×32  (UpSample + Conv3D + SE + Résiduel)
        Out    240×240×155×num_classes (Conv3D 1×1×1)

    Args:
        in_channels:      Nombre de modalités IRM (défaut 4)
        num_classes:      Nombre de classes de sortie (défaut 3 : ET, TC, WT)
        encoder_channels: Liste des canaux par niveau encoder
        bottleneck_ch:    Canaux bottleneck
        se_reduction:     Ratio réduction SE (défaut 16)
        dropout:          Dropout avant couche finale
    """
    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 3,
        encoder_channels: list = None,
        bottleneck_ch: int = 256,
        se_reduction: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [32, 64, 128]

        # Couche d'entrée initiale
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, encoder_channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(encoder_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder (3 niveaux)
        self.enc1 = EncoderBlock(encoder_channels[0], encoder_channels[0], se_reduction)
        self.enc2 = EncoderBlock(encoder_channels[0], encoder_channels[1], se_reduction)
        self.enc3 = EncoderBlock(encoder_channels[1], encoder_channels[2], se_reduction)

        # Bottleneck
        self.bottleneck = SEResBlock3D(encoder_channels[2], bottleneck_ch, se_reduction)

        # Decoder (3 niveaux — symétrique à l'encoder)
        self.dec1 = DecoderBlock(bottleneck_ch, encoder_channels[2], encoder_channels[2], se_reduction)
        self.dec2 = DecoderBlock(encoder_channels[2], encoder_channels[1], encoder_channels[1], se_reduction)
        self.dec3 = DecoderBlock(encoder_channels[1], encoder_channels[0], encoder_channels[0], se_reduction)

        self.dropout = nn.Dropout3d(p=dropout)

        # Couche de sortie — Conv 1×1×1 + segmentation
        self.output_conv = nn.Conv3d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Entrée
        x = self.input_conv(x)

        # Encoder — on garde les feature maps pour les skip connections
        skip1, x = self.enc1(x)   # skip1: 32 ch, x: 32 ch poolé
        skip2, x = self.enc2(x)   # skip2: 64 ch, x: 64 ch poolé
        skip3, x = self.enc3(x)   # skip3: 128 ch, x: 128 ch poolé

        # Bottleneck
        x = self.bottleneck(x)    # 256 ch

        # Decoder
        x = self.dec1(x, skip3)   # 128 ch
        x = self.dec2(x, skip2)   # 64 ch
        x = self.dec3(x, skip1)   # 32 ch

        x = self.dropout(x)

        # Sortie — logits (pas de softmax, géré dans la loss)
        return self.output_conv(x)

    def count_parameters(self) -> int:
        """Retourne le nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
