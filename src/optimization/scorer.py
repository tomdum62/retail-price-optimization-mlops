"""
Scoring multi-criteres des scenarios Monte Carlo.

Score = w_marge * S_marge + w_indice * S_indice + w_lissage * S_lissage

Poids depuis config :
  - marge   : 0.50  (maximisation MLNI)
  - indice  : 0.35  (proximite a l'indice cible)
  - lissage : 0.15  (penalite saut prix J→J+1)
"""

import logging

import numpy as np

from src.config import (
    MC_CONFIG,
    MC_SCORE_WEIGHTS,
    TVA_DEFAULT,
)

logger = logging.getLogger(__name__)


class PriceScorer:
    """Scoring vectorise de scenarios (PVC, PC)."""

    def __init__(self):
        self.weights = MC_SCORE_WEIGHTS
        scoring_cfg = MC_CONFIG["scoring"]
        self.marge_scale_max = scoring_cfg["marge_scale_max_pct"]
        self.indice_zone_opt = scoring_cfg["indice_zone_optimale_pts"]
        self.indice_pen_above = scoring_cfg["indice_penalite_au_dessus"]
        self.indice_pen_below = scoring_cfg["indice_penalite_en_dessous"]
        self.lissage_seuil = scoring_cfg["lissage_seuil_pct"]

    def score_batch(
        self,
        scenarios: np.ndarray,
        pa: float,
        prix_concurrent: float,
        indice_cible: float,
        pvc_precedent: float,
        tva: float = TVA_DEFAULT,
    ) -> np.ndarray:
        """Score N scenarios pour un produit x base.

        Parameters
        ----------
        scenarios : np.ndarray shape (N, 2)
            Colonnes [PVC, PC].
        pa : float
            Prix d'achat.
        prix_concurrent : float
            Prix concurrent de reference.
        indice_cible : float
            Indice cible (ex: 102).
        pvc_precedent : float
            PVC du run precedent (pour lissage).
        tva : float
            TVA applicable.

        Returns
        -------
        np.ndarray shape (N,)
            Score combine [0, 1] pour chaque scenario.
        """
        pvc = scenarios[:, 0]
        pc = scenarios[:, 1]
        n = len(pvc)

        coef_tva = 1.0 + tva / 100.0

        # --- Score marge ---
        ca_ht = pvc / coef_tva
        mlni_val = ca_ht - pa
        mlni_taux = np.where(pvc > 0, mlni_val / pvc * 100, 0)
        s_marge = np.clip(mlni_taux / self.marge_scale_max, 0, 1)

        # --- Score indice ---
        if prix_concurrent > 0:
            indice = pvc / prix_concurrent * 100
            ecart = indice - indice_cible

            s_indice = np.ones(n)
            # Au-dessus de la zone optimale : penalite forte
            mask_above = ecart > self.indice_zone_opt
            s_indice[mask_above] = np.clip(
                1 - (ecart[mask_above] - self.indice_zone_opt) / self.indice_pen_above,
                0, 1,
            )
            # En dessous de la zone optimale : penalite legere
            mask_below = ecart < -self.indice_zone_opt
            s_indice[mask_below] = np.clip(
                1 - (-ecart[mask_below] - self.indice_zone_opt) / self.indice_pen_below,
                0, 1,
            )
        else:
            s_indice = np.ones(n) * 0.5  # pas d'info concurrent

        # --- Score lissage ---
        if pvc_precedent > 0:
            variation_pct = np.abs(pvc - pvc_precedent) / pvc_precedent * 100
            s_lissage = np.where(
                variation_pct <= self.lissage_seuil,
                1.0,
                np.clip(1 - (variation_pct - self.lissage_seuil) / 20, 0, 1),
            )
        else:
            s_lissage = np.ones(n)

        # --- Score combine ---
        score = (
            self.weights["marge"] * s_marge
            + self.weights["indice"] * s_indice
            + self.weights["lissage"] * s_lissage
        )

        return score

    def select_best(
        self,
        scenarios: np.ndarray,
        scores: np.ndarray,
        pa: float,
        tva: float = TVA_DEFAULT,
    ) -> dict:
        """Selectionne le meilleur scenario et enrichit les metriques.

        Returns
        -------
        dict
            PVC_Optimal, PC_Optimal, Score, MLNI_Prevu, MADH_Prevu,
            MFIL_Prevu, Is_Feasible, N_Feasible.
        """
        coef_tva = 1.0 + tva / 100.0
        pvc = scenarios[:, 0]
        pc = scenarios[:, 1]

        # Masque de faisabilite (MADH >= 0, MFIL >= 0)
        ca_ht = pvc / coef_tva
        madh = ca_ht - pc
        mfil = pc - pa
        feasible = (madh >= 0) & (mfil >= 0)
        n_feasible = int(feasible.sum())

        if n_feasible == 0:
            # Aucun scenario faisable : prendre le meilleur quand meme
            best_idx = np.argmax(scores)
            is_feasible = False
        else:
            # Parmi les faisables, prendre le meilleur score
            scores_f = np.where(feasible, scores, -np.inf)
            best_idx = np.argmax(scores_f)
            is_feasible = True

        best_pvc = pvc[best_idx]
        best_pc = pc[best_idx]
        best_ca_ht = best_pvc / coef_tva

        return {
            "PVC_Optimal": round(best_pvc, 2),
            "PC_Optimal": round(best_pc, 4),
            "Score": round(scores[best_idx], 4),
            "MLNI_Prevu": round((best_ca_ht - pa) / best_pvc * 100, 2) if best_pvc > 0 else 0,
            "MADH_Prevu": round((best_ca_ht - best_pc) / best_pvc * 100, 2) if best_pvc > 0 else 0,
            "MFIL_Prevu": round((best_pc - pa) / best_pc * 100, 2) if best_pc > 0 else 0,
            "Is_Feasible": is_feasible,
            "N_Feasible": n_feasible,
        }
