#!/bin/bash
# ============================================================
# Script SLURM — Compute Canada (Alliance de Recherche)
# SE-RES-CNN 3D BraTS-Africa | Brice Gyebre | UQAC | 2026
#
# Soumettre avec : sbatch scripts/compute_canada.sh
# Statut avec   : squeue -u $USER
# ============================================================

#SBATCH --job-name=se_res_cnn_brats
#SBATCH --account=def-<TON_SUPERVISEUR>    # <-- Remplacer par ton compte PI
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1                  # GPU A100 80GB
#SBATCH --mem=32G
#SBATCH --time=12:00:00                    # 12h max par job
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=ton.email@uqac.ca     # <-- Ton email UQAC
#SBATCH --mail-type=BEGIN,END,FAIL

# ── Setup environnement ──
module load python/3.10
module load cuda/12.1
source ~/envs/brats_env/bin/activate

mkdir -p logs checkpoints results

# ── Entraînement K-fold avec W&B ──
python scripts/train.py \
    --config configs/config.yaml \
    --kfold \
    --wandb \
    --data_dir /scratch/$USER/brats_africa

echo "Job terminé : $(date)"
