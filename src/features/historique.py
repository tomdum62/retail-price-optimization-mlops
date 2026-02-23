"""
Chargement des donnees financieres historiques (VF SCAFLF).

Source : tables Databricks ou fichiers parquet locaux.
Colonnes cles : CODBAS, Enseigne, CODE_PRODUIT_STANDARD, Date,
                CA_TTC, CA_HT, Val_Cession (PC), Val_Achat (PA), Qte_Reelle.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import BASES, TVA_DEFAULT

logger = logging.getLogger(__name__)


def load_historique(
    source: str | Path,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> pd.DataFrame:
    """Charge les donnees financieres historiques.

    Parameters
    ----------
    source : str or Path
        Chemin parquet ou nom de table Databricks.
    date_min, date_max : str, optional
        Filtres dates (format YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame
        Donnees financieres nettoyees.
    """
    source = str(source)

    if source.endswith(".parquet"):
        df = pd.read_parquet(source)
    elif source.endswith(".csv"):
        df = pd.read_csv(source, sep=";", encoding="utf-8")
    else:
        # Databricks — utilise spark si disponible
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark is None:
                raise RuntimeError("Pas de session Spark active.")
            df = spark.table(source).toPandas()
        except ImportError:
            raise ImportError(
                f"Source '{source}' n'est ni un fichier parquet/csv "
                "ni une table Spark accessible."
            )

    # Normalisation types
    df["CODBAS"] = df["CODBAS"].astype(str)
    df["Date"] = pd.to_datetime(df["Date"])

    # Filtres dates
    if date_min:
        df = df[df["Date"] >= pd.Timestamp(date_min)]
    if date_max:
        df = df[df["Date"] <= pd.Timestamp(date_max)]

    # Filtrer aux bases connues
    bases_connues = set(BASES.keys())
    n_before = len(df)
    df = df[df["CODBAS"].isin(bases_connues)].copy()
    n_filtered = n_before - len(df)
    if n_filtered > 0:
        logger.warning("Historique : %d lignes exclues (bases inconnues).", n_filtered)

    logger.info(
        "Historique charge : %d lignes, %s → %s.",
        len(df),
        df["Date"].min().strftime("%Y-%m-%d") if len(df) else "N/A",
        df["Date"].max().strftime("%Y-%m-%d") if len(df) else "N/A",
    )
    return df


def compute_marges(df: pd.DataFrame, tva: float = TVA_DEFAULT) -> pd.DataFrame:
    """Ajoute les colonnes de marges calculees.

    Colonnes ajoutees : CA_HT, MADH_Val, MFIL_Val, MLNI_Val,
                        MADH_Taux, MFIL_Taux, MLNI_Taux.
    """
    coef_tva = 1.0 + tva / 100.0

    # Valeurs
    df["CA_HT"] = df["CA_TTC"] / coef_tva
    df["MADH_Val"] = df["CA_HT"] - df["Val_Cession"]
    df["MFIL_Val"] = df["Val_Cession"] - df["Val_Achat"]
    df["MLNI_Val"] = df["CA_HT"] - df["Val_Achat"]

    # Taux (denominateurs SCAFLF)
    df["MADH_Taux"] = df["MADH_Val"] / df["CA_TTC"] * 100  # denom CA_TTC
    df["MFIL_Taux"] = df["MFIL_Val"] / df["Val_Cession"] * 100  # denom PC
    df["MLNI_Taux"] = df["MLNI_Val"] / df["CA_TTC"] * 100  # denom CA_TTC

    return df
