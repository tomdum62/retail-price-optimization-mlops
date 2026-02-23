"""
Validation des contraintes de marge.

Verifie la faisabilite des prix recommandes vis-a-vis :
  - MADH >= 0 (adherent ne perd pas)
  - MFIL >= 0 (filiere ne perd pas)
  - MLNI >= cible (marge globale enseigne)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.config import OBJECTIFS, TVA_DEFAULT

logger = logging.getLogger(__name__)


def validate_margins(
    pvc: float,
    pc: float,
    pa: float,
    tva: float = TVA_DEFAULT,
    enseigne: str = "INTERMARCHE",
) -> Dict[str, any]:
    """Valide les 3 marges pour un triplet (PVC, PC, PA).

    Returns
    -------
    dict
        MADH_Val, MFIL_Val, MLNI_Val, MADH_Taux, MFIL_Taux, MLNI_Taux,
        MADH_ok, MFIL_ok, MLNI_ok, all_ok.
    """
    obj = OBJECTIFS.get(enseigne, OBJECTIFS.get("INTERMARCHE"))
    coef_tva = 1.0 + tva / 100.0

    ca_ttc = pvc
    ca_ht = pvc / coef_tva

    madh_val = ca_ht - pc
    mfil_val = pc - pa
    mlni_val = ca_ht - pa

    madh_taux = madh_val / ca_ttc * 100 if ca_ttc > 0 else 0
    mfil_taux = mfil_val / pc * 100 if pc > 0 else 0
    mlni_taux = mlni_val / ca_ttc * 100 if ca_ttc > 0 else 0

    madh_ok = madh_taux >= obj["MADH_TAUX_MIN"]
    mfil_ok = mfil_taux >= obj["MFIL_TAUX_MIN"]
    mlni_ok = mlni_taux >= obj["MLNI_TAUX"] * 0.5  # Seuil produit = 50% de l'objectif global

    return {
        "MADH_Val": round(madh_val, 4),
        "MFIL_Val": round(mfil_val, 4),
        "MLNI_Val": round(mlni_val, 4),
        "MADH_Taux": round(madh_taux, 2),
        "MFIL_Taux": round(mfil_taux, 2),
        "MLNI_Taux": round(mlni_taux, 2),
        "MADH_ok": madh_ok,
        "MFIL_ok": mfil_ok,
        "MLNI_ok": mlni_ok,
        "all_ok": madh_ok and mfil_ok and mlni_ok,
    }


def compute_prix_bornes(
    pa: float,
    prix_concurrent: float,
    indice_cible: float,
    mlni_cible: float,
    tva: float = TVA_DEFAULT,
) -> Dict[str, float]:
    """Calcule les bornes de prix faisables pour un produit.

    Bornes :
      - PVC_min : plancher marge MLNI → PVC_min = PA / (1/coef_tva - mlni_cible/100)
      - PVC_max : plafond indice → PVC_max = prix_concurrent * indice_cible / 100
      - PC_min = PA (MFIL >= 0)
      - PC_max = CA_HT (MADH >= 0)

    Returns
    -------
    dict
        PVC_min, PVC_max, PC_min, PC_max, feasible, tension.
    """
    coef_tva = 1.0 + tva / 100.0

    # PVC plancher marge
    denom = 1.0 / coef_tva - mlni_cible / 100.0
    if denom <= 0:
        pvc_min = np.inf
    else:
        pvc_min = pa / denom

    # PVC plafond indice
    if prix_concurrent > 0:
        pvc_max = prix_concurrent * indice_cible / 100.0
    else:
        pvc_max = np.inf

    # PC bornes
    pc_min = pa
    pc_max = pvc_max / coef_tva if pvc_max < np.inf else np.inf

    feasible = pvc_min <= pvc_max
    tension = not feasible  # plancher marge > plafond indice

    return {
        "PVC_min": round(pvc_min, 4) if pvc_min < np.inf else np.inf,
        "PVC_max": round(pvc_max, 4) if pvc_max < np.inf else np.inf,
        "PC_min": round(pc_min, 4),
        "PC_max": round(pc_max, 4) if pc_max < np.inf else np.inf,
        "feasible": feasible,
        "tension": tension,
    }


def flag_produits_tensions(
    df_products: pd.DataFrame,
    enseigne: str,
) -> pd.DataFrame:
    """Identifie les produits en tension (marge plancher > indice plafond).

    Returns
    -------
    pd.DataFrame
        Avec colonnes additionnelles : PVC_min, PVC_max, feasible, tension.
    """
    obj = OBJECTIFS.get(enseigne)
    if obj is None:
        raise ValueError(f"Enseigne inconnue : {enseigne}")

    col_conc = obj["colonne_prix_concurrent"]
    indice_cible = obj["INDICE_CIBLE"]
    mlni_cible = obj["MLNI_TAUX"]

    results = []
    for _, row in df_products.iterrows():
        prix_conc = row.get(col_conc, 0) or 0
        pa = row.get("PA", 0) or 0

        bornes = compute_prix_bornes(
            pa=pa,
            prix_concurrent=prix_conc,
            indice_cible=indice_cible,
            mlni_cible=mlni_cible,
        )
        results.append(bornes)

    df_bornes = pd.DataFrame(results)
    df_out = pd.concat([df_products.reset_index(drop=True), df_bornes], axis=1)

    n_tension = df_out["tension"].sum()
    if n_tension > 0:
        logger.warning(
            "%s : %d/%d produits en tension (marge plancher > indice plafond).",
            enseigne, n_tension, len(df_out),
        )

    return df_out
