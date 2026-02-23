"""
POLCO : chargement des prix contraints nationaux (INTERMARCHE).

Les fichiers PRICING_FFL_YYYYMMDD.csv definissent les PVC obligatoires
(negocies ou stickers). Ces prix sont une contrainte dure : le modele
ne predit PAS le PVC pour ces produits.

Format CSV (sans en-tete, separateur ;, encoding windows-1252) :
    Type;Enseigne;Code_CPS;Zone;Prix_brut
    055;IM;00039798;02;00001,99

Parametres lus depuis config/business_rules.yaml (section produits.polco).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import POLCO_ENSEIGNE, POLCO_FORMAT

logger = logging.getLogger(__name__)


def _parse_polco_file(filepath: Path) -> pd.DataFrame:
    """Parse un fichier PRICING_FFL_YYYYMMDD.csv.

    Returns
    -------
    pd.DataFrame
        Colonnes : CODE_PRODUIT_STANDARD, pvc_polco, date_fichier
    """
    stem = filepath.stem
    date_str = stem.split("_")[-1]
    date_fichier = pd.Timestamp(
        year=int(date_str[:4]),
        month=int(date_str[4:6]),
        day=int(date_str[6:8]),
    )

    col_names = POLCO_FORMAT["colonnes"]

    df = pd.read_csv(
        filepath,
        sep=POLCO_FORMAT["separateur"],
        header=None,
        names=col_names,
        encoding=POLCO_FORMAT["encoding"],
        dtype={c: str for c in col_names},
    )

    # Parser le prix : "00001,99" → 1.99
    df["pvc_polco"] = (
        df["Prix_brut"]
        .str.strip()
        .str.replace(",", ".", regex=False)
        .apply(lambda x: float(x) if x else None)
    )

    # Nettoyer Code_CPS : strip zeros en tete
    stripped = df["Code_CPS"].str.strip().str.lstrip("0")
    df["CODE_PRODUIT_STANDARD"] = stripped.where(stripped != "", "0")

    df["date_fichier"] = date_fichier

    # Filtrer lignes valides
    df = df[df["pvc_polco"].notna() & (df["pvc_polco"] > 0)].copy()

    # Dedoublonner par produit : prix moyen si plusieurs zones
    df = df.groupby("CODE_PRODUIT_STANDARD", as_index=False).agg(
        pvc_polco=("pvc_polco", "mean"),
        date_fichier=("date_fichier", "first"),
    )

    return df


def _date_to_semaine_ffl(date: pd.Timestamp) -> str:
    """Convertit une date en identifiant semaine FFL (ISO)."""
    iso = date.isocalendar()
    return f"SEMAINE FFL S{iso[1]:02d}-{iso[0] % 100}"


def load_polco(polco_dir: Optional[str | Path] = None) -> pd.DataFrame:
    """Charge tous les fichiers POLCO et construit le referentiel prix contraints.

    Parameters
    ----------
    polco_dir : Path, optional
        Repertoire contenant les PRICING_FFL_*.csv.

    Returns
    -------
    pd.DataFrame
        Colonnes : CODE_PRODUIT_STANDARD, Enseigne, Semaine_FFL, pvc_polco, is_polco
    """
    empty_cols = [
        "CODE_PRODUIT_STANDARD", "Enseigne", "Semaine_FFL", "pvc_polco", "is_polco",
    ]

    if polco_dir is None:
        logger.warning("Repertoire POLCO non specifie.")
        return pd.DataFrame(columns=empty_cols)

    polco_dir = Path(polco_dir)
    if not polco_dir.exists():
        logger.warning("Repertoire POLCO inexistant : %s", polco_dir)
        return pd.DataFrame(columns=empty_cols)

    files = sorted(polco_dir.glob("PRICING_FFL_*.csv"))
    if not files:
        logger.warning("Aucun fichier POLCO dans %s", polco_dir)
        return pd.DataFrame(columns=empty_cols)

    logger.info("Chargement de %d fichiers POLCO...", len(files))

    frames = []
    for f in files:
        df_f = _parse_polco_file(f)
        if df_f.empty:
            continue
        df_f["Semaine_FFL"] = _date_to_semaine_ffl(df_f["date_fichier"].iloc[0])
        frames.append(df_f)
        logger.info("  %s : %d produits → %s", f.name, len(df_f), df_f["Semaine_FFL"].iloc[0])

    if not frames:
        return pd.DataFrame(columns=empty_cols)

    df = pd.concat(frames, ignore_index=True)

    # Garder un seul prix par produit x semaine (dernier fichier)
    df = df.sort_values("date_fichier", ascending=False)
    df = df.drop_duplicates(
        subset=["CODE_PRODUIT_STANDARD", "Semaine_FFL"],
        keep="first",
    )

    df["Enseigne"] = POLCO_ENSEIGNE
    df["is_polco"] = 1

    output_cols = ["CODE_PRODUIT_STANDARD", "Enseigne", "Semaine_FFL", "pvc_polco", "is_polco"]
    df = df[output_cols].reset_index(drop=True)

    logger.info(
        "POLCO charge : %d lignes, %d produits, %d semaines.",
        len(df), df["CODE_PRODUIT_STANDARD"].nunique(), df["Semaine_FFL"].nunique(),
    )
    return df
