"""
Simulateur Monte Carlo de scenarios de prix.

Genere N scenarios (PVC, PC) pour chaque produit x base, en respectant
les bornes de faisabilite :
  - PVC_min : plancher de marge MLNI (issu de la perequation)
  - PVC_max : plafond d'indice vs concurrent
  - PC_min = PA  (MFIL >= 0)
  - PC_max = CA_HT = PVC / (1 + TVA/100)  (MADH >= 0)

PVC arrondis sur paliers FL (X.X9, pas de 0.10).
"""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    MC_N_SCENARIOS,
    MC_PVC_VARIATION_PCT,
    MC_SEED,
    FL_PVC_STEP,
    FL_PVC_SUFFIX,
    TVA_DEFAULT,
)

logger = logging.getLogger(__name__)


class PriceSimulator:
    """Generateur Monte Carlo de scenarios de prix (PVC, PC)."""

    def __init__(
        self,
        n_scenarios: int = MC_N_SCENARIOS,
        variation_pct: float = MC_PVC_VARIATION_PCT,
        seed: int = MC_SEED,
    ):
        self.n_scenarios = n_scenarios
        self.variation_pct = variation_pct
        self.rng = np.random.default_rng(seed)

    def generate_scenarios(
        self,
        pa: float,
        pvc_current: float,
        prix_concurrent: Optional[float],
        indice_cible: float,
        mlni_cible: float,
        tva: float = TVA_DEFAULT,
    ) -> np.ndarray:
        """Genere N scenarios (PVC, PC) pour un produit x base.

        La cible MLNI vient de la perequation (calculee en amont).

        Returns
        -------
        np.ndarray
            Matrice (N, 2) avec colonnes [PVC, PC].
            Si infaisable, retourne shape (0, 2).
        """
        if pa <= 0 or pvc_current <= 0:
            return np.empty((0, 2), dtype=np.float64)

        coef_tva = 1.0 + tva / 100.0

        # --- Borne 1 : plancher de marge MLNI ---
        denom = 1.0 / coef_tva - mlni_cible / 100.0
        if denom <= 0:
            pvc_min_marge = pvc_current * 1.5  # fallback
        else:
            pvc_min_marge = pa / denom

        # --- Borne 2 : plafond d'indice ---
        if prix_concurrent is not None and prix_concurrent > 0:
            pvc_max_indice = prix_concurrent * indice_cible / 100.0
        else:
            pvc_max_indice = pvc_current * (1.0 + self.variation_pct / 100.0)

        # --- Borne 3 : variation autour du PVC actuel ---
        pvc_var_min = pvc_current * (1.0 - self.variation_pct / 100.0)
        pvc_var_max = pvc_current * (1.0 + self.variation_pct / 100.0)

        # --- Intersection ---
        pvc_lower = max(pvc_min_marge, pvc_var_min)
        pvc_upper = min(pvc_max_indice, pvc_var_max)

        # Relaxation si infaisable
        if pvc_lower > pvc_upper:
            midpoint = (pvc_min_marge + pvc_max_indice) / 2.0
            pvc_lower = midpoint * 0.95
            pvc_upper = midpoint * 1.05

            if pvc_lower >= pvc_upper:
                pvc_lower = pvc_current * 0.97
                pvc_upper = pvc_current * 1.03

        if pvc_lower >= pvc_upper:
            pvc_lower = pvc_current * 0.99
            pvc_upper = pvc_current * 1.01

        pvc_lower = max(pvc_lower, 0.01)
        pvc_upper = max(pvc_upper, pvc_lower + 0.01)

        # --- Tirage PVC sur paliers FL (X.X9) ---
        paliers = self._generate_fl_grid(pvc_lower, pvc_upper)

        if len(paliers) == 0:
            paliers = np.array([self._round_pvc_fl((pvc_lower + pvc_upper) / 2)])

        pvc_scenarios = self.rng.choice(paliers, size=self.n_scenarios)

        # --- Tirage PC : uniforme dans [PA, CA_HT] ---
        ca_ht = pvc_scenarios / coef_tva
        pc_min = np.full(self.n_scenarios, pa)
        pc_max = ca_ht

        mask_infeasible = pc_min > pc_max
        if mask_infeasible.any():
            pc_max[mask_infeasible] = pc_min[mask_infeasible]

        u = self.rng.random(self.n_scenarios)
        pc_scenarios = pc_min + u * (pc_max - pc_min)

        return np.column_stack([pvc_scenarios, pc_scenarios])

    @staticmethod
    def _round_pvc_fl(pvc: float) -> float:
        """Arrondi au palier FL X.X9 le plus proche."""
        if pvc <= 0:
            return 0.09
        rounded = round((pvc + 0.01) * 10) / 10 - 0.01
        return max(round(rounded, 2), 0.09)

    @staticmethod
    def _generate_fl_grid(pvc_min: float, pvc_max: float) -> np.ndarray:
        """Genere tous les paliers FL (X.X9) dans [pvc_min, pvc_max].

        Utilise un compteur entier pour eviter la derive flottante.
        """
        start_tenth = math.ceil((pvc_min + 0.01) * 10) / 10 - 0.01
        start_tenth = round(start_tenth, 2)
        if start_tenth < pvc_min - 0.001:
            start_tenth = round(start_tenth + FL_PVC_STEP, 2)

        paliers = []
        i = 0
        while True:
            p = round(start_tenth + i * FL_PVC_STEP, 2)
            if p > pvc_max + 0.001:
                break
            if p >= pvc_min - 0.001:
                paliers.append(p)
            i += 1
        return np.array(paliers, dtype=np.float64)

    def generate_batch(
        self,
        df_products: pd.DataFrame,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """Genere des scenarios pour tous les produits d'un DataFrame.

        Parameters
        ----------
        df_products : pd.DataFrame
            Colonnes requises : CODE_PRODUIT, CODBAS, PA, PVC_Current,
            Prix_Concurrent, Indice_Cible, MLNI_Cible.

        Returns
        -------
        dict
            Cle = (CODE_PRODUIT, CODBAS), Valeur = np.ndarray (N, 2).
        """
        required = {
            "CODE_PRODUIT", "CODBAS", "PA", "PVC_Current",
            "Prix_Concurrent", "Indice_Cible", "MLNI_Cible",
        }
        missing = required - set(df_products.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes : {missing}")

        results = {}
        n_generated = 0
        n_skipped = 0

        for _, row in df_products.iterrows():
            key = (str(row["CODE_PRODUIT"]), str(row["CODBAS"]))

            prix_conc = row["Prix_Concurrent"]
            if pd.isna(prix_conc):
                prix_conc = None

            scenarios = self.generate_scenarios(
                pa=row["PA"],
                pvc_current=row["PVC_Current"],
                prix_concurrent=prix_conc,
                indice_cible=row["Indice_Cible"],
                mlni_cible=row["MLNI_Cible"],
            )

            results[key] = scenarios
            if scenarios.shape[0] > 0:
                n_generated += 1
            else:
                n_skipped += 1

        logger.info(
            "Simulation batch : %d/%d produits avec scenarios (%d ignores).",
            n_generated, n_generated + n_skipped, n_skipped,
        )
        return results
