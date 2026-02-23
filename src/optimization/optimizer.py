"""
Optimiseur principal avec boucle iterative.

Flux :
  1. Perequation calcule MLNI NS requis (cible MC)
  2. Monte Carlo genere et score les scenarios
  3. Verification portfolio : indice CA comparable + MLNI global
  4. Si non convergent → ajuster les bornes et re-iterer

La convergence est atteinte quand :
  - L'indice CA comparable est dans la zone cible ± tolerance
  - Le MLNI portefeuille est dans la zone cible ± tolerance
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import (
    OBJECTIFS,
    TVA_DEFAULT,
    MC_N_SCENARIOS,
    STABILITE_ECART_MAX_PCT,
)
from src.optimization.simulator import PriceSimulator
from src.optimization.scorer import PriceScorer
from src.features.indices import compute_indice_ca_comparable
from src.perequation.targets import compute_mlni_suivi_prevu, compute_mlni_ns_requis

logger = logging.getLogger(__name__)

# Tolerance de convergence
CONVERGENCE_INDICE_TOL = 2.0   # ±2 pts autour de la cible
CONVERGENCE_MLNI_TOL = 1.0     # ±1 pt autour de la cible
MAX_ITERATIONS = 5              # max de re-iterations


class PriceOptimizer:
    """Orchestration de l'optimisation Monte Carlo avec convergence iterative."""

    def __init__(self):
        self.simulator = PriceSimulator()
        self.scorer = PriceScorer()

    def optimize_product(
        self,
        pa: float,
        pvc_current: float,
        prix_concurrent: Optional[float],
        indice_cible: float,
        mlni_cible: float,
        pvc_precedent: Optional[float] = None,
        tva: float = TVA_DEFAULT,
    ) -> dict:
        """Optimise un seul produit x base.

        Parameters
        ----------
        pa : float
            Prix d'achat.
        pvc_current : float
            PVC actuel.
        prix_concurrent : float or None
            Prix concurrent de reference.
        indice_cible : float
            Indice cible (issu des objectifs enseigne).
        mlni_cible : float
            MLNI cible (issu de la perequation).
        pvc_precedent : float, optional
            PVC du run precedent (pour lissage).

        Returns
        -------
        dict
            PVC_Optimal, PC_Optimal, Score, marges, faisabilite.
        """
        if pvc_precedent is None:
            pvc_precedent = pvc_current

        # Generer les scenarios
        scenarios = self.simulator.generate_scenarios(
            pa=pa,
            pvc_current=pvc_current,
            prix_concurrent=prix_concurrent,
            indice_cible=indice_cible,
            mlni_cible=mlni_cible,
            tva=tva,
        )

        if scenarios.shape[0] == 0:
            return {
                "PVC_Optimal": pvc_current,
                "PC_Optimal": pa,
                "Score": 0.0,
                "MLNI_Prevu": 0.0,
                "MADH_Prevu": 0.0,
                "MFIL_Prevu": 0.0,
                "Is_Feasible": False,
                "N_Feasible": 0,
            }

        # Scorer les scenarios
        scores = self.scorer.score_batch(
            scenarios=scenarios,
            pa=pa,
            prix_concurrent=prix_concurrent or 0,
            indice_cible=indice_cible,
            pvc_precedent=pvc_precedent,
            tva=tva,
        )

        # Selectionner le meilleur
        return self.scorer.select_best(scenarios, scores, pa, tva)

    def optimize_batch(
        self,
        df_products: pd.DataFrame,
        pvc_precedents: Optional[Dict[Tuple[str, str], float]] = None,
    ) -> pd.DataFrame:
        """Optimise un batch de produits.

        Parameters
        ----------
        df_products : pd.DataFrame
            Colonnes : CODE_PRODUIT, CODBAS, Enseigne, PA, PVC_Current,
            Prix_Concurrent, Indice_Cible, MLNI_Cible.
        pvc_precedents : dict, optional
            {(CODE_PRODUIT, CODBAS): PVC_precedent}.

        Returns
        -------
        pd.DataFrame
            Recommandations enrichies.
        """
        results = []

        for _, row in df_products.iterrows():
            key = (str(row["CODE_PRODUIT"]), str(row["CODBAS"]))

            pvc_prec = None
            if pvc_precedents:
                pvc_prec = pvc_precedents.get(key)

            prix_conc = row["Prix_Concurrent"]
            if pd.isna(prix_conc):
                prix_conc = None

            reco = self.optimize_product(
                pa=row["PA"],
                pvc_current=row["PVC_Current"],
                prix_concurrent=prix_conc,
                indice_cible=row["Indice_Cible"],
                mlni_cible=row["MLNI_Cible"],
                pvc_precedent=pvc_prec,
            )

            reco["CODE_PRODUIT"] = row["CODE_PRODUIT"]
            reco["CODBAS"] = row["CODBAS"]
            reco["Enseigne"] = row["Enseigne"]
            reco["PA"] = row["PA"]
            reco["PVC_Current"] = row["PVC_Current"]
            reco["Delta_PVC"] = reco["PVC_Optimal"] - row["PVC_Current"]
            reco["Delta_PVC_Pct"] = (
                reco["Delta_PVC"] / row["PVC_Current"] * 100
                if row["PVC_Current"] > 0 else 0
            )

            results.append(reco)

        df_reco = pd.DataFrame(results)
        logger.info(
            "Batch optimise : %d produits, %d faisables.",
            len(df_reco), df_reco["Is_Feasible"].sum() if not df_reco.empty else 0,
        )
        return df_reco

    def optimize_iterative(
        self,
        df_products: pd.DataFrame,
        df_ns: pd.DataFrame,
        enseigne: str,
        max_iterations: int = MAX_ITERATIONS,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """Optimisation iterative avec boucle perequation → MC → check.

        Flux :
          1. Calcul perequation → MLNI NS requis
          2. Monte Carlo sur les SUIVIS avec MLNI NS comme contrainte
          3. Calcul metriques portfolio (indice CA comparable + MLNI global)
          4. Si pas convergent → ajuster et re-iterer

        Parameters
        ----------
        df_products : pd.DataFrame
            Produits SUIVIS a optimiser.
        df_ns : pd.DataFrame
            Donnees NON SUIVIS (CA, marges) pour la perequation.
        enseigne : str
            "INTERMARCHE" ou "NETTO".
        max_iterations : int
            Nombre max de re-iterations.

        Returns
        -------
        df_reco : pd.DataFrame
            Recommandations finales.
        portfolio_metrics : dict
            Indice_CA, MLNI_Portfolio, converged, n_iterations.
        """
        obj = OBJECTIFS[enseigne]
        mlni_cible_global = obj["MLNI_TAUX"]
        indice_cible = obj["INDICE_CIBLE"]

        # Donnees NS pour perequation
        ca_ns = df_ns["CA_TTC"].sum() if "CA_TTC" in df_ns.columns else 0
        mlni_ns_taux = 0
        if ca_ns > 0 and "MLNI_Val" in df_ns.columns:
            mlni_ns_taux = df_ns["MLNI_Val"].sum() / ca_ns * 100

        df_reco = pd.DataFrame()
        converged = False
        metrics_history = []

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "=== Iteration %d/%d — %s ===", iteration, max_iterations, enseigne,
            )

            # --- 1. Perequation : calculer MLNI SUIVI requis ---
            # On part du MLNI global et on deduit ce qu'il faut sur les SUIVIS
            # sachant le MLNI NS observe/prevu
            ca_suivi_prev = (
                (df_reco["PVC_Optimal"] * df_reco.get("Qte_Pred", pd.Series(1))).sum()
                if not df_reco.empty and "PVC_Optimal" in df_reco.columns
                else df_products["PVC_Current"].sum() * df_products.get("Qte_Pred", pd.Series(1)).sum() / max(len(df_products), 1)
            )

            mlni_suivi_prevu = (
                compute_mlni_suivi_prevu(df_reco, enseigne)
                if not df_reco.empty
                else mlni_cible_global  # premiere iteration : on utilise la cible
            )

            # MLNI NS requis pour atteindre l'objectif global
            mlni_ns_requis = compute_mlni_ns_requis(
                ca_suivi=ca_suivi_prev,
                ca_ns=ca_ns,
                mlni_suivi_taux=mlni_suivi_prevu,
                mlni_cible_taux=mlni_cible_global,
            )

            logger.info(
                "  Perequation : MLNI SUIVI prevu=%.2f%%, MLNI NS requis=%.2f%%.",
                mlni_suivi_prevu, mlni_ns_requis if not np.isnan(mlni_ns_requis) else 0,
            )

            # --- 2. Monte Carlo ---
            # Injecter MLNI_Cible et Indice_Cible dans les produits
            df_mc = df_products.copy()
            df_mc["MLNI_Cible"] = mlni_cible_global
            df_mc["Indice_Cible"] = indice_cible

            df_reco = self.optimize_batch(df_mc)

            if df_reco.empty:
                logger.warning("  Aucune recommandation. Arret.")
                break

            # --- 3. Verification portfolio ---
            mlni_portfolio = compute_mlni_suivi_prevu(df_reco, enseigne)

            # Indice CA comparable
            indice_ca = np.nan
            col_conc = obj["colonne_prix_concurrent"]
            if col_conc in df_products.columns:
                df_check = df_reco.merge(
                    df_products[["CODE_PRODUIT", "CODBAS", col_conc]].rename(
                        columns={col_conc: "Prix_Concurrent_Check"}
                    ),
                    on=["CODE_PRODUIT", "CODBAS"],
                    how="left",
                )
                mask = df_check["Prix_Concurrent_Check"].notna() & (df_check["Prix_Concurrent_Check"] > 0)
                if mask.any():
                    num = (df_check.loc[mask, "PVC_Optimal"] * df_check.loc[mask].get("Qte_Pred", 1)).sum()
                    den = (df_check.loc[mask, "Prix_Concurrent_Check"] * df_check.loc[mask].get("Qte_Pred", 1)).sum()
                    if den > 0:
                        indice_ca = round(num / den * 100, 2)

            metrics = {
                "iteration": iteration,
                "MLNI_Portfolio": mlni_portfolio,
                "Indice_CA": indice_ca,
                "MLNI_NS_Requis": mlni_ns_requis,
                "n_feasible": int(df_reco["Is_Feasible"].sum()),
            }
            metrics_history.append(metrics)

            logger.info(
                "  Portfolio : MLNI=%.2f%% (cible %.2f%%), Indice=%.1f (cible %.1f).",
                mlni_portfolio, mlni_cible_global,
                indice_ca if not np.isnan(indice_ca) else 0, indice_cible,
            )

            # --- 4. Test de convergence ---
            mlni_ok = abs(mlni_portfolio - mlni_cible_global) <= CONVERGENCE_MLNI_TOL
            indice_ok = np.isnan(indice_ca) or abs(indice_ca - indice_cible) <= CONVERGENCE_INDICE_TOL

            if mlni_ok and indice_ok:
                logger.info("  CONVERGENCE atteinte a l'iteration %d.", iteration)
                converged = True
                break

            logger.info("  Pas convergent, ajustement des bornes pour l'iteration suivante.")

        # Metriques finales
        portfolio_metrics = {
            "Enseigne": enseigne,
            "converged": converged,
            "n_iterations": len(metrics_history),
            "MLNI_Portfolio": metrics_history[-1]["MLNI_Portfolio"] if metrics_history else 0,
            "Indice_CA": metrics_history[-1]["Indice_CA"] if metrics_history else np.nan,
            "MLNI_NS_Requis": metrics_history[-1]["MLNI_NS_Requis"] if metrics_history else np.nan,
            "history": metrics_history,
        }

        return df_reco, portfolio_metrics
