"""
Prediction sur horizon J+N.

Genere les predictions Qte et CA pour un horizon de jours
a partir du modele LightGBM entraine.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import FORECAST_CONFIG

logger = logging.getLogger(__name__)


def predict_horizon(
    model,
    df_history: pd.DataFrame,
    horizon_days: int = 7,
    feature_names: Optional[list] = None,
) -> pd.DataFrame:
    """Genere les predictions sur l'horizon demande.

    Pour chaque jour de l'horizon, construit les features a partir
    de l'historique et predit les quantites.

    Parameters
    ----------
    model : LGBMRegressor
        Modele entraine.
    df_history : pd.DataFrame
        Historique recent (60 jours min).
    horizon_days : int
        Nombre de jours a predire.
    feature_names : list, optional
        Noms des features (si None, utilise model.feature_name_).

    Returns
    -------
    pd.DataFrame
        Predictions avec colonnes : CODE_PRODUIT_STANDARD, CODBAS,
        Enseigne, Date, horizon, Qte_Pred, CA_Pred, PVC.
    """
    if feature_names is None:
        feature_names = model.feature_name_

    max_date = df_history["Date"].max()
    predictions = []

    for h in range(1, horizon_days + 1):
        target_date = max_date + pd.Timedelta(days=h)

        # Construire les features pour cette date
        df_features = _build_features_for_date(df_history, target_date, feature_names)

        if df_features.empty:
            continue

        # Predire
        X = df_features[feature_names]
        qte_pred = model.predict(X)

        # Assembler le resultat
        df_pred = df_features[["CODE_PRODUIT_STANDARD", "CODBAS", "Enseigne"]].copy()
        df_pred["Date"] = target_date
        df_pred["horizon"] = h
        df_pred["Qte_Pred"] = np.maximum(qte_pred, 0)  # pas de quantite negative

        # CA predit = Qte * PVC actuel
        if "PVC" in df_features.columns:
            df_pred["PVC"] = df_features["PVC"].values
            df_pred["CA_Pred"] = df_pred["Qte_Pred"] * df_pred["PVC"]
        else:
            df_pred["PVC"] = np.nan
            df_pred["CA_Pred"] = np.nan

        predictions.append(df_pred)

    if not predictions:
        logger.warning("Aucune prediction generee.")
        return pd.DataFrame()

    df_all = pd.concat(predictions, ignore_index=True)

    logger.info(
        "Predictions : %d lignes, horizon J+1 a J+%d.",
        len(df_all), horizon_days,
    )
    return df_all


def _build_features_for_date(
    df_history: pd.DataFrame,
    target_date: pd.Timestamp,
    feature_names: list,
) -> pd.DataFrame:
    """Construit les features pour une date future.

    Utilise les dernieres valeurs connues de chaque produit x base
    et met a jour les features calendaires/temporelles.
    """
    from src.features.calendrier import add_calendar_features

    # Dernier enregistrement par produit x base
    df_latest = df_history.sort_values("Date").drop_duplicates(
        subset=["CODE_PRODUIT_STANDARD", "CODBAS"], keep="last",
    ).copy()

    # Mettre a jour la date
    df_latest["Date"] = target_date

    # Recalculer les features calendaires
    df_latest = add_calendar_features(df_latest)

    # S'assurer que toutes les features sont presentes
    for col in feature_names:
        if col not in df_latest.columns:
            df_latest[col] = 0

    return df_latest
