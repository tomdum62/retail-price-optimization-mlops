"""
Chargement et calcul des indices prix concurrents.

L'indice CA comparable est le ratio pondere par les quantites x coefficients :
    Indice = sum(PVC * Qte * Coef) / sum(PrixConc * Qte * Coef) * 100

IMPORTANT : ce n'est PAS une moyenne simple des ratios individuels.
(erreur corrigee en V5 : 3.98 pts d'ecart sur NETTO).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import OBJECTIFS, FILIERE_HORTICOLE_CODE

logger = logging.getLogger(__name__)


def load_indices(source: str) -> pd.DataFrame:
    """Charge les donnees d'indices prix concurrents.

    Parameters
    ----------
    source : str
        Chemin parquet/csv ou table Databricks.

    Returns
    -------
    pd.DataFrame
        Colonnes : CODE_PRODUIT_STANDARD, CODBAS, Enseigne,
                   Prix_Concurrent, Indice_Brut, Coef_Ponderation.
    """
    if source.endswith(".parquet"):
        df = pd.read_parquet(source)
    elif source.endswith(".csv"):
        df = pd.read_csv(source, sep=";")
    else:
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            df = spark.table(source).toPandas()
        except ImportError:
            raise ImportError(f"Source indices inaccessible : {source}")

    df["CODBAS"] = df["CODBAS"].astype(str)

    # Exclure filiere horticole (pas d'indices prix)
    if "Code_Filiere" in df.columns:
        n_before = len(df)
        df = df[df["Code_Filiere"] != FILIERE_HORTICOLE_CODE].copy()
        logger.info("Filiere horticole exclue : %d lignes.", n_before - len(df))

    logger.info("Indices charges : %d lignes.", len(df))
    return df


def compute_indice_ca_comparable(
    df: pd.DataFrame,
    enseigne: str,
    col_pvc: str = "PVC",
    col_qte: str = "Qte_Reelle",
    col_coef: str = "Coef_Ponderation",
) -> float:
    """Calcule l'indice CA comparable pour une enseigne.

    Formule V5 (corrigee) :
        Indice = sum(PVC * Qte * Coef) / sum(PrixConc * Qte * Coef) * 100

    Parameters
    ----------
    df : pd.DataFrame
        Donnees produit avec PVC, prix concurrent, quantites et coefficients.
    enseigne : str
        "INTERMARCHE" ou "NETTO".

    Returns
    -------
    float
        Indice CA comparable.
    """
    obj = OBJECTIFS.get(enseigne)
    if obj is None:
        raise ValueError(f"Enseigne inconnue : {enseigne}")

    col_conc = obj["colonne_prix_concurrent"]

    if col_conc not in df.columns:
        logger.warning("Colonne %s absente, indice non calculable.", col_conc)
        return np.nan

    # Filtrer lignes avec prix concurrent valide
    mask = df[col_conc].notna() & (df[col_conc] > 0)
    df_valid = df[mask]

    if df_valid.empty:
        return np.nan

    # Si pas de coefficient, on met 1
    if col_coef not in df_valid.columns:
        coef = 1.0
    else:
        coef = df_valid[col_coef].fillna(1.0)

    numerateur = (df_valid[col_pvc] * df_valid[col_qte] * coef).sum()
    denominateur = (df_valid[col_conc] * df_valid[col_qte] * coef).sum()

    if denominateur == 0:
        return np.nan

    indice = numerateur / denominateur * 100
    return round(indice, 2)


def flag_produits_avec_indices(
    df_produits: pd.DataFrame,
    df_indices: pd.DataFrame,
) -> pd.DataFrame:
    """Marque les produits qui ont des indices concurrents disponibles.

    Seuls ces produits seront optimises (regle metier).

    Returns
    -------
    pd.DataFrame
        df_produits avec colonne 'has_indice' (bool).
    """
    clefs_indices = set(
        zip(
            df_indices["CODE_PRODUIT_STANDARD"].astype(str),
            df_indices["CODBAS"].astype(str),
        )
    )

    df_produits["has_indice"] = [
        (str(row["CODE_PRODUIT_STANDARD"]), str(row["CODBAS"])) in clefs_indices
        for _, row in df_produits.iterrows()
    ]

    n_with = df_produits["has_indice"].sum()
    logger.info(
        "Produits avec indices : %d / %d (%.0f%%).",
        n_with, len(df_produits), n_with / len(df_produits) * 100 if len(df_produits) else 0,
    )
    return df_produits
