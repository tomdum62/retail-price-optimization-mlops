"""
Entrainement du modele LightGBM de forecast quantites/CA.

Le modele predit Qte_Reelle par produit x base x jour.
Le CA est derive : CA_Pred = Qte_Pred * PVC.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import FORECAST_CONFIG, LGBM_PARAMS, PROJECT_ROOT

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "data" / "models"

# Features cibles et identifiants (pas des features d'entree)
NON_FEATURES = {
    "Date", "Qte_Reelle", "CA_TTC", "CA_HT",
    "CODE_PRODUIT_STANDARD", "CODBAS", "Enseigne",
    "Semaine_FFL", "Maille_Suivi",
    "is_polco", "is_suivi", "is_non_suivi",
}


def prepare_train_data(
    df: pd.DataFrame,
    target: str = "Qte_Reelle",
    exclude_cols: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Prepare X, y pour l'entrainement.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset complet (output de build_dataset).
    target : str
        Colonne cible.
    exclude_cols : set, optional
        Colonnes additionnelles a exclure.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    feature_names : list[str]
    """
    if target not in df.columns:
        raise ValueError(f"Colonne cible '{target}' absente du dataset.")

    exclude = NON_FEATURES.copy()
    if exclude_cols:
        exclude |= exclude_cols

    feature_cols = [c for c in df.columns if c not in exclude and c != target]

    # Garder seulement les colonnes numeriques
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target]

    feature_names = list(X.columns)

    # Supprimer les lignes avec target NaN
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    logger.info(
        "Train data : %d lignes, %d features. Target: %s.",
        len(X), len(feature_names), target,
    )
    return X, y, feature_names


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict] = None,
) -> "lightgbm.LGBMRegressor":
    """Entraine un modele LightGBM.

    Parameters
    ----------
    X_train, y_train : train split.
    X_val, y_val : validation split (pour early stopping).
    params : dict, optional
        Hyperparametres LightGBM (defaut depuis config).

    Returns
    -------
    lightgbm.LGBMRegressor
        Modele entraine.
    """
    import lightgbm as lgb

    if params is None:
        params = LGBM_PARAMS.copy()

    early_stopping = FORECAST_CONFIG.get("early_stopping_rounds", 50)

    model = lgb.LGBMRegressor(**params)

    fit_kwargs = {}
    if X_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=True),
            lgb.log_evaluation(period=100),
        ]

    model.fit(X_train, y_train, **fit_kwargs)

    logger.info(
        "Modele entraine : %d estimators, %d features.",
        model.n_estimators_, len(model.feature_name_),
    )
    return model


def save_model(model, path: Optional[Path] = None) -> Path:
    """Sauvegarde le modele au format joblib."""
    import joblib

    if path is None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        path = MODELS_DIR / "forecast_lgbm.joblib"

    joblib.dump(model, path)
    logger.info("Modele sauvegarde : %s", path)
    return path


def load_model(path: Optional[Path] = None):
    """Charge un modele sauvegarde."""
    import joblib

    if path is None:
        path = MODELS_DIR / "forecast_lgbm.joblib"

    if not path.exists():
        raise FileNotFoundError(f"Modele introuvable : {path}")

    model = joblib.load(path)
    logger.info("Modele charge : %s", path)
    return model
