"""
Assemblage du dataset produit final pour le forecast.

Fusionne : historique financier + indices + POLCO + calendrier + meteo.
Produit un dataset par produit x base x jour avec toutes les features.
"""

import logging
from typing import Optional

import pandas as pd

from src.config import PRODUITS_CONFIG
from src.features.calendrier import add_calendar_features
from src.features.historique import compute_marges

logger = logging.getLogger(__name__)


def build_product_dataset(
    df_historique: pd.DataFrame,
    df_indices: pd.DataFrame,
    df_polco: pd.DataFrame,
    df_meteo: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble le dataset produit pour le forecast.

    Parameters
    ----------
    df_historique : pd.DataFrame
        Donnees financieres (CA, PC, PA, Qte).
    df_indices : pd.DataFrame
        Indices prix concurrents.
    df_polco : pd.DataFrame
        Prix contraints POLCO.
    df_meteo : pd.DataFrame, optional
        Features meteo par base x jour.

    Returns
    -------
    pd.DataFrame
        Dataset complet pret pour le forecast.
    """
    df = df_historique.copy()

    # --- Marges ---
    df = compute_marges(df)

    # --- Calendrier ---
    df = add_calendar_features(df)

    # --- Maille Suivi ---
    col_suivi = PRODUITS_CONFIG["suivis"]["filtre_colonne"]
    val_suivi = PRODUITS_CONFIG["suivis"]["valeur_filtre"]
    val_ns = PRODUITS_CONFIG["non_suivis"]["valeur_filtre"]

    if col_suivi in df.columns:
        df["is_suivi"] = (df[col_suivi] == val_suivi).astype(int)
        df["is_non_suivi"] = (df[col_suivi] == val_ns).astype(int)
    else:
        logger.warning("Colonne '%s' absente. Tous les produits seront traites comme SUIVI.", col_suivi)
        df["is_suivi"] = 1
        df["is_non_suivi"] = 0

    # --- POLCO flag ---
    if not df_polco.empty:
        polco_keys = set(df_polco["CODE_PRODUIT_STANDARD"].astype(str))
        df["is_polco"] = df["CODE_PRODUIT_STANDARD"].astype(str).isin(polco_keys).astype(int)
    else:
        df["is_polco"] = 0

    # --- Indices ---
    if not df_indices.empty:
        merge_cols = ["CODE_PRODUIT_STANDARD", "CODBAS"]
        # Prendre le dernier indice disponible
        idx_latest = df_indices.sort_values("Date" if "Date" in df_indices.columns else merge_cols[0])
        idx_latest = idx_latest.drop_duplicates(subset=merge_cols, keep="last")

        suffix_cols = [c for c in idx_latest.columns if c not in merge_cols]
        df = df.merge(
            idx_latest[merge_cols + ["Prix_Concurrent"]],
            on=merge_cols,
            how="left",
        )
    else:
        df["Prix_Concurrent"] = None

    # --- Meteo ---
    if df_meteo is not None and not df_meteo.empty:
        df = df.merge(
            df_meteo,
            on=["CODBAS", "Date"],
            how="left",
        )

    logger.info(
        "Dataset assemble : %d lignes, %d colonnes, "
        "%d SUIVI, %d NS, %d POLCO.",
        len(df), len(df.columns),
        df["is_suivi"].sum() if "is_suivi" in df.columns else 0,
        df["is_non_suivi"].sum() if "is_non_suivi" in df.columns else 0,
        df["is_polco"].sum(),
    )
    return df
