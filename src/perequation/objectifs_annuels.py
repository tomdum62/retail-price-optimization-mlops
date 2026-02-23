"""
Decomposition et pilotage des objectifs annuels MFIL / MADH / MLNI par base.

PRINCIPES :
  1. Les objectifs sont ANNUELS — on ne regarde pas le reste a faire.
  2. Chaque base a sa propre cible AJUSTEE dynamiquement.
  3. Anti "coup de volant" : les corrections sont amorties.
     Si une base est a +5pts sur MFIL, on ne corrige pas de -5pts d'un coup.
     On applique :  correction = min(ecart * coef_amortissement, ecart_max_pts)

FORMULES DE DECOMPOSITION :
  MLNI_Val = MADH_Val + MFIL_Val       (toujours vrai en valeur)
  MLNI_Val = CA_HT - PA
  MADH_Val = CA_HT - PC
  MFIL_Val = PC - PA

  En TAUX, les denominateurs different :
    MADH_Taux = MADH_Val / CA_TTC
    MFIL_Taux = MFIL_Val / Val_Cession (= PC)
    MLNI_Taux = MLNI_Val / CA_TTC

  Donc MADH_Taux + MFIL_Taux != MLNI_Taux (denominateurs differents).
  La validation porte sur les VALEURS (additives) et les TAUX (chaque cible).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    OBJECTIFS,
    OBJECTIFS_ANNUELS_ACTIF,
    AMORTISSEMENT_COEF,
    AMORTISSEMENT_ECART_MAX,
    OBJECTIFS_ANNUELS_CONFIG,
    TOLERANCE_CONFIG,
    TVA_DEFAULT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Validation de la decomposition (MFIL_Val + MADH_Val = MLNI_Val)
# ---------------------------------------------------------------------------

def validate_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """Verifie que MFIL_Val + MADH_Val = MLNI_Val pour chaque ligne.

    Parameters
    ----------
    df : pd.DataFrame
        Doit contenir : MADH_Val, MFIL_Val, MLNI_Val.

    Returns
    -------
    pd.DataFrame
        Avec colonnes ajoutees :
        - decomposition_ok : bool (ecart < 0.01)
        - decomposition_ecart : float (MADH_Val + MFIL_Val - MLNI_Val)
    """
    required = {"MADH_Val", "MFIL_Val", "MLNI_Val"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("validate_decomposition : colonnes manquantes %s", missing)
        out = df.copy()
        out["decomposition_ok"] = False
        out["decomposition_ecart"] = np.nan
        return out

    out = df.copy()
    somme = out["MADH_Val"] + out["MFIL_Val"]
    out["decomposition_ecart"] = somme - out["MLNI_Val"]
    out["decomposition_ok"] = out["decomposition_ecart"].abs() < 0.01

    n_ko = (~out["decomposition_ok"]).sum()
    if n_ko > 0:
        logger.warning(
            "Decomposition marge : %d/%d lignes incoherentes "
            "(MADH_Val + MFIL_Val != MLNI_Val).",
            n_ko, len(out),
        )

    return out


# ---------------------------------------------------------------------------
# 2. Calcul de la cible ajustee par base (mobile, avec amortissement)
# ---------------------------------------------------------------------------

def compute_cible_ajustee(
    cible_annuelle: float,
    taux_observe: float,
    coef_amortissement: float = AMORTISSEMENT_COEF,
    ecart_max_pts: float = AMORTISSEMENT_ECART_MAX,
    cible_min: Optional[float] = None,
    cible_max: Optional[float] = None,
) -> Dict[str, float]:
    """Calcule la cible ajustee pour un run, avec anti coup de volant.

    Si la base est en avance (taux_observe > cible), on relache legerement.
    Si la base est en retard (taux_observe < cible), on durcit legerement.
    Jamais plus de ecart_max_pts de correction par run.

    Parameters
    ----------
    cible_annuelle : float
        Objectif annuel de l'enseigne pour cette marge (%).
    taux_observe : float
        Taux observe actuellement sur cette base (%).
    coef_amortissement : float
        Fraction de l'ecart a corriger par run (0-1). Default 0.40.
    ecart_max_pts : float
        Correction max en points par run. Default 3.0 pts.
    cible_min : float, optional
        Borne min de la cible ajustee.
    cible_max : float, optional
        Borne max de la cible ajustee.

    Returns
    -------
    dict
        cible_ajustee : float
            Cible pour ce run.
        ecart_vs_annuel : float
            taux_observe - cible_annuelle (>0 = en avance).
        correction_brute : float
            Correction avant amortissement.
        correction_effective : float
            Correction apres amortissement et cap.
        situation : str
            "en_avance" | "en_retard" | "sur_cible".
    """
    ecart = taux_observe - cible_annuelle  # >0 = en avance, <0 = en retard

    # Tolerance
    tol = TOLERANCE_CONFIG.get("MLNI_pts", 1.0)
    if abs(ecart) <= tol:
        return {
            "cible_ajustee": cible_annuelle,
            "ecart_vs_annuel": round(ecart, 4),
            "correction_brute": 0.0,
            "correction_effective": 0.0,
            "situation": "sur_cible",
        }

    # Correction brute = ramener vers la cible
    correction_brute = -ecart  # si en avance (+), on relache (-) => cible baisse

    # Amortissement
    correction_amortie = correction_brute * coef_amortissement

    # Cap
    correction_effective = np.clip(correction_amortie, -ecart_max_pts, ecart_max_pts)

    cible_ajustee = cible_annuelle + correction_effective

    # Bornes de securite
    if cible_min is not None:
        cible_ajustee = max(cible_ajustee, cible_min)
    if cible_max is not None:
        cible_ajustee = min(cible_ajustee, cible_max)

    situation = "en_avance" if ecart > 0 else "en_retard"

    return {
        "cible_ajustee": round(cible_ajustee, 4),
        "ecart_vs_annuel": round(ecart, 4),
        "correction_brute": round(correction_brute, 4),
        "correction_effective": round(correction_effective, 4),
        "situation": situation,
    }


# ---------------------------------------------------------------------------
# 3. Decomposition des objectifs par base
# ---------------------------------------------------------------------------

def compute_objectifs_par_base(
    df_marges_base: pd.DataFrame,
    enseigne: str,
) -> pd.DataFrame:
    """Calcule les cibles ajustees MFIL, MADH, MLNI pour chaque base.

    Parameters
    ----------
    df_marges_base : pd.DataFrame
        Marges observees par base. Colonnes requises :
        CODBAS, MLNI_Taux, MADH_Taux, MFIL_Taux.
    enseigne : str
        "INTERMARCHE" ou "NETTO".

    Returns
    -------
    pd.DataFrame
        Une ligne par base, avec :
        - *_cible_annuelle, *_taux_observe, *_cible_ajustee, *_situation
          pour MLNI, MADH, MFIL.
        - ecart_alerte : bool (ecart > seuil d'alerte sur au moins une marge).
    """
    obj = OBJECTIFS.get(enseigne)
    if obj is None:
        raise ValueError(f"Enseigne inconnue : {enseigne}")

    bornes = OBJECTIFS_ANNUELS_CONFIG.get("bornes_cible", {})
    alerte_pts = OBJECTIFS_ANNUELS_CONFIG.get("amortissement", {}).get(
        "ecart_alerte_pts", 5.0,
    )

    results = []
    for _, row in df_marges_base.iterrows():
        codbas = row["CODBAS"]
        r = {"CODBAS": codbas, "Enseigne": enseigne}

        # --- MLNI ---
        mlni_obs = row.get("MLNI_Taux", 0.0)
        mlni_adj = compute_cible_ajustee(
            cible_annuelle=obj["MLNI_TAUX"],
            taux_observe=mlni_obs,
        )
        r["MLNI_cible_annuelle"] = obj["MLNI_TAUX"]
        r["MLNI_taux_observe"] = mlni_obs
        r["MLNI_cible_ajustee"] = mlni_adj["cible_ajustee"]
        r["MLNI_situation"] = mlni_adj["situation"]
        r["MLNI_correction_pts"] = mlni_adj["correction_effective"]

        # --- MADH ---
        madh_obs = row.get("MADH_Taux", 0.0)
        madh_adj = compute_cible_ajustee(
            cible_annuelle=obj["MADH_TAUX_CIBLE"],
            taux_observe=madh_obs,
            cible_min=bornes.get("MADH_TAUX_MIN"),
            cible_max=bornes.get("MADH_TAUX_MAX"),
        )
        r["MADH_cible_annuelle"] = obj["MADH_TAUX_CIBLE"]
        r["MADH_taux_observe"] = madh_obs
        r["MADH_cible_ajustee"] = madh_adj["cible_ajustee"]
        r["MADH_situation"] = madh_adj["situation"]
        r["MADH_correction_pts"] = madh_adj["correction_effective"]

        # --- MFIL ---
        mfil_obs = row.get("MFIL_Taux", 0.0)
        mfil_adj = compute_cible_ajustee(
            cible_annuelle=obj["MFIL_TAUX_CIBLE"],
            taux_observe=mfil_obs,
            cible_min=bornes.get("MFIL_TAUX_MIN"),
            cible_max=bornes.get("MFIL_TAUX_MAX"),
        )
        r["MFIL_cible_annuelle"] = obj["MFIL_TAUX_CIBLE"]
        r["MFIL_taux_observe"] = mfil_obs
        r["MFIL_cible_ajustee"] = mfil_adj["cible_ajustee"]
        r["MFIL_situation"] = mfil_adj["situation"]
        r["MFIL_correction_pts"] = mfil_adj["correction_effective"]

        # --- Alerte ---
        ecart_max = max(
            abs(mlni_adj["ecart_vs_annuel"]),
            abs(madh_adj["ecart_vs_annuel"]),
            abs(mfil_adj["ecart_vs_annuel"]),
        )
        r["ecart_alerte"] = ecart_max > alerte_pts

        results.append(r)

    df_out = pd.DataFrame(results)

    # Log
    n_alerte = df_out["ecart_alerte"].sum() if not df_out.empty else 0
    if n_alerte > 0:
        logger.warning(
            "%s : %d/%d bases en ALERTE (ecart > %.1f pts sur au moins une marge).",
            enseigne, n_alerte, len(df_out), alerte_pts,
        )

    for marge in ("MLNI", "MADH", "MFIL"):
        col_sit = f"{marge}_situation"
        if col_sit in df_out.columns:
            n_retard = (df_out[col_sit] == "en_retard").sum()
            n_avance = (df_out[col_sit] == "en_avance").sum()
            n_cible = (df_out[col_sit] == "sur_cible").sum()
            logger.info(
                "%s %s : %d sur_cible, %d en_avance, %d en_retard.",
                enseigne, marge, n_cible, n_avance, n_retard,
            )

    return df_out


# ---------------------------------------------------------------------------
# 4. Integration : injecter les cibles ajustees dans le batch MC
# ---------------------------------------------------------------------------

def inject_cibles_ajustees(
    df_products: pd.DataFrame,
    df_objectifs_base: pd.DataFrame,
) -> pd.DataFrame:
    """Enrichit les produits avec les cibles ajustees de leur base.

    Ajoute MLNI_Cible_Ajustee, MADH_Cible_Ajustee, MFIL_Cible_Ajustee
    pour chaque produit en fonction de sa base.

    Parameters
    ----------
    df_products : pd.DataFrame
        Produits a optimiser (avec CODBAS).
    df_objectifs_base : pd.DataFrame
        Output de compute_objectifs_par_base.

    Returns
    -------
    pd.DataFrame
        Produits enrichis.
    """
    cols_join = [
        "CODBAS",
        "MLNI_cible_ajustee",
        "MADH_cible_ajustee",
        "MFIL_cible_ajustee",
        "MLNI_situation",
        "MADH_situation",
        "MFIL_situation",
    ]
    existing = [c for c in cols_join if c in df_objectifs_base.columns]

    df = df_products.merge(
        df_objectifs_base[existing],
        on="CODBAS",
        how="left",
    )

    # Fallback si pas de cible ajustee (base non trouvee)
    for marge in ("MLNI", "MADH", "MFIL"):
        col = f"{marge}_cible_ajustee"
        if col not in df.columns:
            df[col] = np.nan

    return df


# ---------------------------------------------------------------------------
# 5. Synthese pour le reporting
# ---------------------------------------------------------------------------

def build_synthese_objectifs(
    df_objectifs_base: pd.DataFrame,
    enseigne: str,
) -> Dict[str, Any]:
    """Construit un resume des objectifs pour le reporting.

    Returns
    -------
    dict
        Resume : nb bases par situation, ecarts moyens, alertes.
    """
    if df_objectifs_base.empty:
        return {"enseigne": enseigne, "nb_bases": 0}

    synthese = {
        "enseigne": enseigne,
        "nb_bases": len(df_objectifs_base),
        "nb_alertes": int(df_objectifs_base["ecart_alerte"].sum()),
    }

    for marge in ("MLNI", "MADH", "MFIL"):
        col_obs = f"{marge}_taux_observe"
        col_cible = f"{marge}_cible_annuelle"
        col_ajustee = f"{marge}_cible_ajustee"
        col_sit = f"{marge}_situation"

        if col_obs in df_objectifs_base.columns:
            synthese[f"{marge}_observe_moyen"] = round(
                df_objectifs_base[col_obs].mean(), 2,
            )
            synthese[f"{marge}_cible_annuelle"] = round(
                df_objectifs_base[col_cible].mean(), 2,
            )
            synthese[f"{marge}_cible_ajustee_moyen"] = round(
                df_objectifs_base[col_ajustee].mean(), 2,
            )
            synthese[f"{marge}_nb_en_retard"] = int(
                (df_objectifs_base[col_sit] == "en_retard").sum(),
            )
            synthese[f"{marge}_nb_en_avance"] = int(
                (df_objectifs_base[col_sit] == "en_avance").sum(),
            )
            synthese[f"{marge}_nb_sur_cible"] = int(
                (df_objectifs_base[col_sit] == "sur_cible").sum(),
            )

    return synthese
