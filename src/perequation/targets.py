"""
Calcul des cibles de perequation.

IMPORTANT : ce module se calcule AVANT le Monte Carlo.
Il fournit la cible MLNI NS requise pour atteindre l'objectif global.
Cette cible alimente ensuite le Monte Carlo comme contrainte.

Formule centrale :
    MLNI_total = (CA_suivi * MLNI_suivi + CA_ns * MLNI_ns) / CA_total

Inversion :
    MLNI_ns_requis = (MLNI_cible * CA_total - MLNI_suivi * CA_suivi) / CA_ns
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import OBJECTIFS, TVA_DEFAULT

logger = logging.getLogger(__name__)


def compute_mlni_ns_requis(
    ca_suivi: float,
    ca_ns: float,
    mlni_suivi_taux: float,
    mlni_cible_taux: float,
) -> float:
    """Calcule le taux MLNI NS requis pour atteindre l'objectif global.

    Parameters
    ----------
    ca_suivi : float
        CA TTC des produits SUIVI.
    ca_ns : float
        CA TTC des produits NON SUIVI.
    mlni_suivi_taux : float
        Taux MLNI actuel/prevu des SUIVIS (%).
    mlni_cible_taux : float
        Objectif MLNI global de l'enseigne (%).

    Returns
    -------
    float
        Taux MLNI NS requis (%).
        Peut etre negatif si l'objectif est trop ambitieux.
    """
    ca_total = ca_suivi + ca_ns

    if ca_total <= 0 or ca_ns <= 0:
        logger.warning("CA total ou CA NS <= 0 : perequation impossible.")
        return np.nan

    mlni_ns_requis = (
        (mlni_cible_taux / 100 * ca_total - mlni_suivi_taux / 100 * ca_suivi)
        / ca_ns * 100
    )

    return round(mlni_ns_requis, 4)


def compute_perequation_targets(
    df_pereq_matrix: pd.DataFrame,
    enseigne: str,
) -> pd.DataFrame:
    """Calcule les cibles de perequation pour chaque base x semaine.

    Parameters
    ----------
    df_pereq_matrix : pd.DataFrame
        Matrice de perequation (output de build_perequation_matrix).
    enseigne : str
        "INTERMARCHE" ou "NETTO".

    Returns
    -------
    pd.DataFrame
        Avec colonnes additionnelles :
        - MLNI_NS_Requis : taux MLNI NS requis
        - Ecart_MLNI_Total : ecart vs objectif
        - Sensibilite_1pt_NS : impact de +1pt NS sur le total
        - NS_Requis_Realiste : flag (MLNI NS requis < 50%)
    """
    obj = OBJECTIFS.get(enseigne)
    if obj is None:
        raise ValueError(f"Enseigne inconnue : {enseigne}")

    mlni_cible = obj["MLNI_TAUX"]

    df = df_pereq_matrix.copy()

    # Filtrer pour l'enseigne
    if "Enseigne" in df.columns:
        df = df[df["Enseigne"] == enseigne].copy()

    # MLNI NS requis
    df["MLNI_NS_Requis"] = df.apply(
        lambda row: compute_mlni_ns_requis(
            ca_suivi=row.get("CA_TTC_SUIVI", 0),
            ca_ns=row.get("CA_TTC_NS", 0),
            mlni_suivi_taux=row.get("MLNI_Taux_SUIVI", 0),
            mlni_cible_taux=mlni_cible,
        ),
        axis=1,
    )

    # Ecart vs objectif
    df["Ecart_MLNI_Total"] = df.get("MLNI_Taux_TOTAL", pd.Series(0, index=df.index)) - mlni_cible

    # Sensibilite : impact de +1pt MLNI NS sur le MLNI total
    # d(MLNI_total)/d(MLNI_ns) = CA_ns / CA_total
    df["Sensibilite_1pt_NS"] = np.where(
        df.get("CA_TTC_TOTAL", pd.Series(0, index=df.index)) > 0,
        df.get("CA_TTC_NS", pd.Series(0, index=df.index))
        / df.get("CA_TTC_TOTAL", pd.Series(1, index=df.index)),
        0,
    )

    # Flag realiste : MLNI NS requis < 50% (au-dela c'est intenable)
    df["NS_Requis_Realiste"] = (df["MLNI_NS_Requis"] < 50).astype(int)

    logger.info(
        "Perequation %s : %d lignes, MLNI NS requis moyen = %.2f%%.",
        enseigne, len(df),
        df["MLNI_NS_Requis"].mean() if not df.empty else 0,
    )
    return df


def compute_prix_bornes_from_mlni(
    pa: float,
    mlni_cible_taux: float,
    tva: float = TVA_DEFAULT,
) -> Dict[str, float]:
    """Derive les bornes de prix (PVC_min, PC_min, PC_max) depuis un objectif MLNI.

    Utilise par le Monte Carlo comme contrainte de marge.

    Parameters
    ----------
    pa : float
        Prix d'achat.
    mlni_cible_taux : float
        Taux MLNI cible (%).
    tva : float
        TVA applicable.

    Returns
    -------
    dict
        PVC_min, PC_min (=PA), PC_max, feasible (bool).
    """
    coef_tva = 1.0 + tva / 100.0
    denominateur = 1.0 / coef_tva - mlni_cible_taux / 100.0

    if denominateur <= 0:
        # Marge cible trop elevee, pas de PVC faisable
        return {
            "PVC_min": np.inf,
            "PC_min": pa,
            "PC_max": np.inf,
            "feasible": False,
        }

    pvc_min = pa / denominateur

    return {
        "PVC_min": round(pvc_min, 4),
        "PC_min": pa,
        "PC_max": round(pvc_min / coef_tva, 4),
        "feasible": True,
    }


def compute_mlni_suivi_prevu(
    df_reco: pd.DataFrame,
    enseigne: str,
) -> float:
    """Calcule le MLNI SUIVI prevu apres optimisation (pour la boucle iterative).

    Utilise apres le Monte Carlo pour verifier la convergence.

    Parameters
    ----------
    df_reco : pd.DataFrame
        Recommandations MC (avec PVC_Optimal, PA, CA prevu).
    enseigne : str
        Enseigne a evaluer.

    Returns
    -------
    float
        MLNI taux SUIVI prevu (%).
    """
    df = df_reco[df_reco["Enseigne"] == enseigne] if "Enseigne" in df_reco.columns else df_reco

    if df.empty:
        return 0.0

    coef_tva = 1.0 + TVA_DEFAULT / 100.0

    # CA TTC = PVC * Qte
    if "Qte_Pred" in df.columns and "PVC_Optimal" in df.columns:
        ca_ttc = (df["PVC_Optimal"] * df["Qte_Pred"]).sum()
        ca_ht = ca_ttc / coef_tva
        pa_total = (df["PA"] * df["Qte_Pred"]).sum()
        mlni_val = ca_ht - pa_total
        if ca_ttc > 0:
            return round(mlni_val / ca_ttc * 100, 4)

    # Fallback sur colonne existante
    if "MLNI_Prevu" in df.columns:
        return round(df["MLNI_Prevu"].mean(), 4)

    return 0.0
