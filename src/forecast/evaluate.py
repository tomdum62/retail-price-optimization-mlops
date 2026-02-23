"""
Evaluation du modele de forecast.

Metriques principales :
  - MAPE (Mean Absolute Percentage Error) au niveau base x CA
  - MAE (Mean Absolute Error) par produit
  - RMSE (Root Mean Squared Error)
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Calcule les metriques d'evaluation.

    Parameters
    ----------
    y_true : pd.Series
        Valeurs reelles.
    y_pred : np.ndarray
        Valeurs predites.

    Returns
    -------
    dict
        MAE, RMSE, MAPE.
    """
    mask = y_true > 0  # eviter div/0 pour MAPE
    y_t = y_true[mask].values
    y_p = y_pred[mask] if isinstance(y_pred, pd.Series) else y_pred[mask.values]

    mae = np.mean(np.abs(y_t - y_p))
    rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
    mape = np.mean(np.abs((y_t - y_p) / y_t)) * 100

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE_pct": round(mape, 2)}


def evaluate_by_base(
    df: pd.DataFrame,
    col_actual: str = "Qte_Reelle",
    col_pred: str = "Qte_Pred",
) -> pd.DataFrame:
    """Evalue les predictions par base logistique.

    Returns
    -------
    pd.DataFrame
        Metriques par CODBAS.
    """
    results = []
    for codbas, grp in df.groupby("CODBAS"):
        mask = grp[col_actual] > 0
        if mask.sum() == 0:
            continue

        y_t = grp.loc[mask, col_actual].values
        y_p = grp.loc[mask, col_pred].values

        metrics = compute_metrics(pd.Series(y_t), y_p)
        metrics["CODBAS"] = codbas
        metrics["n_produits"] = len(grp)
        results.append(metrics)

    df_eval = pd.DataFrame(results)
    if not df_eval.empty:
        logger.info(
            "Evaluation : MAPE moyen = %.2f%%, MAE moyen = %.2f.",
            df_eval["MAPE_pct"].mean(), df_eval["MAE"].mean(),
        )
    return df_eval


def evaluate_ca_base(
    df: pd.DataFrame,
    col_ca_actual: str = "CA_TTC",
    col_ca_pred: str = "CA_Pred",
) -> Dict[str, float]:
    """Evalue le CA agrege par base (metrique principale V2 : 1.3% MAPE).

    Returns
    -------
    dict
        MAPE_CA_base, MAE_CA_base.
    """
    agg_actual = df.groupby("CODBAS")[col_ca_actual].sum()
    agg_pred = df.groupby("CODBAS")[col_ca_pred].sum()

    common = agg_actual.index.intersection(agg_pred.index)
    if len(common) == 0:
        return {"MAPE_CA_base": np.nan, "MAE_CA_base": np.nan}

    actual = agg_actual[common]
    pred = agg_pred[common]

    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    mae = np.mean(np.abs(actual - pred))

    logger.info("CA base : MAPE = %.2f%%, MAE = %.0f.", mape, mae)
    return {"MAPE_CA_base": round(mape, 2), "MAE_CA_base": round(mae, 2)}
