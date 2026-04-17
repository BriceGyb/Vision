"""
Aide au téléchargement — BraTS-Africa (TCIA)
Auteur: Brice Gyebre | MSc Informatique, UQAC | 2026

Instructions :
    1. Créer un compte sur https://www.cancerimagingarchive.net
    2. Accepter la licence CC BY 4.0 pour BraTS-Africa
    3. Installer : pip install tcia-utils
    4. Exécuter ce script

Note : ~15-20 GB de données — prévoir espace disque et bande passante.
"""

import os
import sys


def check_tcia_utils():
    try:
        import tcia_utils
        return True
    except ImportError:
        print("[ERREUR] tcia-utils non installé.")
        print("         Installer avec : pip install tcia-utils")
        return False


def download_brats_africa(output_dir: str = "./data/brats_africa"):
    """
    Télécharge le dataset BraTS-Africa depuis TCIA.

    Args:
        output_dir: Dossier de destination
    """
    if not check_tcia_utils():
        sys.exit(1)

    from tcia_utils import nbia

    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("Téléchargement BraTS-Africa — TCIA/NIH")
    print("="*60)
    print(f"Destination : {os.path.abspath(output_dir)}")
    print()

    # Collection officielle
    collection = "BraTS-Africa"

    print(f"[1/3] Récupération de la liste des patients...")
    try:
        patients = nbia.getPatients(collection=collection)
        print(f"      {len(patients)} patients trouvés")
    except Exception as e:
        print(f"[ERREUR] Impossible de contacter TCIA : {e}")
        print()
        print("Solutions :")
        print("  - Vérifier votre connexion internet")
        print("  - Vérifier que vous avez un compte TCIA et accepté la licence")
        print("  - Télécharger manuellement depuis :")
        print("    https://www.cancerimagingarchive.net/collection/brats-africa/")
        sys.exit(1)

    print(f"[2/3] Téléchargement des séries IRM...")
    try:
        nbia.downloadSeries(
            series_data=nbia.getSeries(collection=collection),
            path=output_dir,
            nThreads=4,
        )
    except Exception as e:
        print(f"[ERREUR] Téléchargement échoué : {e}")
        sys.exit(1)

    print(f"[3/3] Téléchargement terminé !")
    print(f"      Données dans : {os.path.abspath(output_dir)}")
    print()
    print("Prochaine étape :")
    print("  python scripts/train.py --config configs/config.yaml --kfold")


def verify_dataset(data_dir: str = "./data/brats_africa"):
    """Vérifie l'intégrité du dataset téléchargé."""
    from pathlib import Path

    root = Path(data_dir)
    if not root.exists():
        print(f"[ERREUR] Dossier introuvable : {data_dir}")
        return

    modalities = ["t1n", "t1c", "t2w", "t2f", "seg"]
    patients = [d for d in root.iterdir() if d.is_dir()]

    print(f"\nVérification du dataset : {len(patients)} dossiers patients")
    ok = 0
    errors = []

    for patient_dir in patients:
        pid = patient_dir.name
        missing = []
        for mod in modalities:
            f = patient_dir / f"{pid}-{mod}.nii.gz"
            if not f.exists():
                missing.append(mod)
        if missing:
            errors.append(f"  {pid} — manquant: {missing}")
        else:
            ok += 1

    print(f"Patients complets : {ok}/{len(patients)}")
    if errors:
        print(f"Patients avec erreurs ({len(errors)}) :")
        for e in errors[:10]:
            print(e)
        if len(errors) > 10:
            print(f"  ... et {len(errors) - 10} autres")
    else:
        print("Tous les fichiers sont présents !")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/brats_africa")
    parser.add_argument("--verify", action="store_true",
                        help="Vérifier l'intégrité du dataset existant")
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output)
    else:
        download_brats_africa(args.output)
