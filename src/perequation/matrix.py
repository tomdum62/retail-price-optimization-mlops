"""
Matrice de perequation SUIVI / NON SUIVI.

Construit le tableau de bord a la maille :
    Semaine_FFL x CODBAS x Enseigne x Maille_Suivi

Pivot wide : colonnes *_SUIVI, *_NS, *_TOTAL.
Compatible DAX / Power BI pour affichage interactif.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from src.config import (
    PEREQUATION_CONFIG,
    PEREQUATION_CHUNKSIZE,
    TVA_DEFAULT,
)

logger = logging.getLogger(__name__)


def build_perequation_matrix(
    df: pd.DataFrame,
    group_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Construit la matrice de perequation historique ou previsionnelle.

    Parameters
    ----------
    df : pd.DataFrame
        Donnees avec Maille_Suivi, CA_TTC, Val_Cession, Val_Achat, Qte.
    group_keys : list, optional
        Colonnes de regroupement.
        Defaut : [Semaine_FFL, CODBAS, Enseigne].

    Returns
    -------
    pd.DataFrame
        Matrice pivotee : colonnes *_SUIVI, *_NS, *_TOTAL.
    """
    config = PEREQUATION_CONFIG["colonnes_matrice"]

    if group_keys is None:
        group_keys = ["Semaine_FFL", "CODBAS", "Enseigne"]

    value_cols = config["value_cols"]
    pivot_col = config["pivot_col"]

    # Verifier la presence des colonnes
    required = set(group_keys + value_cols + [pivot_col])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la perequation : {missing}")

    # Aggreation par group_keys x Maille_Suivi
    agg_dict = {col: "sum" for col in value_cols}
    df_agg = df.groupby(group_keys + [pivot_col], as_index=False).agg(agg_dict)

    # Pivot : SUIVI / NON SUIVI en colonnes
    df_pivot = df_agg.pivot_table(
        index=group_keys,
        columns=pivot_col,
        values=value_cols,
        aggfunc="sum",
        fill_value=0,
    )

    # Aplatir les colonnes multi-index
    df_pivot.columns = [f"{val}_{maille}" for val, maille in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    # Normaliser les noms : "SUIVI" → "_SUIVI", "NON SUIVI" → "_NS"
    df_pivot.columns = [
        c.replace("_NON SUIVI", "_NS").replace("_SUIVI", "_SUIVI")
        for c in df_pivot.columns
    ]

    # Ajouter les totaux
    for col in value_cols:
        col_s = f"{col}_SUIVI"
        col_ns = f"{col}_NS"
        col_tot = f"{col}_TOTAL"
        if col_s in df_pivot.columns and col_ns in df_pivot.columns:
            df_pivot[col_tot] = df_pivot[col_s] + df_pivot[col_ns]

    # Taux de marge portefeuille
    coef_tva = 1.0 + TVA_DEFAULT / 100.0

    for suffix in ["_SUIVI", "_NS", "_TOTAL"]:
        ca_ttc = f"CA_TTC{suffix}"
        mlni_val = f"MLNI_Val{suffix}"
        if ca_ttc in df_pivot.columns and mlni_val in df_pivot.columns:
            df_pivot[f"MLNI_Taux{suffix}"] = np.where(
                df_pivot[ca_ttc] > 0,
                df_pivot[mlni_val] / df_pivot[ca_ttc] * 100,
                0,
            )

    # Part NS dans le CA total
    if "CA_TTC_TOTAL" in df_pivot.columns and "CA_TTC_NS" in df_pivot.columns:
        df_pivot["Part_NS_Pct"] = np.where(
            df_pivot["CA_TTC_TOTAL"] > 0,
            df_pivot["CA_TTC_NS"] / df_pivot["CA_TTC_TOTAL"] * 100,
            0,
        )

    logger.info(
        "Matrice perequation : %d lignes, %d colonnes.",
        len(df_pivot), len(df_pivot.columns),
    )
    return df_pivot


def build_perequation_from_chunks(
    source: str,
    group_keys: Optional[List[str]] = None,
    chunksize: int = PEREQUATION_CHUNKSIZE,
) -> pd.DataFrame:
    """Construit la matrice en lecture chunked (gros volumes).

    Parameters
    ----------
    source : str
        Chemin vers un fichier CSV volumineux.

    Returns
    -------
    pd.DataFrame
        Matrice de perequation agregee.
    """
    config = PEREQUATION_CONFIG["colonnes_matrice"]
    if group_keys is None:
        group_keys = ["Semaine_FFL", "CODBAS", "Enseigne"]

    pivot_col = config["pivot_col"]
    value_cols = config["value_cols"]
    agg_dict = {col: "sum" for col in value_cols}

    frames = []
    for chunk in pd.read_csv(source, sep=";", chunksize=chunksize):
        chunk_agg = chunk.groupby(group_keys + [pivot_col], as_index=False).agg(agg_dict)
        frames.append(chunk_agg)

    if not frames:
        return pd.DataFrame()

    # Re-aggreger tous les chunks
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.groupby(group_keys + [pivot_col], as_index=False).agg(agg_dict)

    return build_perequation_matrix(df_all, group_keys=group_keys)
