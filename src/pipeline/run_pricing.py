"""
Pipeline d'orchestration pricing bi-hebdomadaire.

Flux complet :
  1. FORECAST : LightGBM V2 → predictions Qte/CA par produit x base
  2. SEPARATION : SUIVI / NON SUIVI / POLCO
  3. PEREQUATION : calcul MLNI NS requis (cible pour le MC)
  4. MONTE CARLO ITERATIF :
     a. Generer scenarios (PVC, PC) avec contraintes (indice + MLNI)
     b. Scorer : marge 50% + indice 35% + lissage 15%
     c. Verification portfolio (indice CA comparable + MLNI global)
     d. Si non convergent → re-iterer (max 5 iterations)
  5. SORTIE : ecriture en tables Databricks (append, partitionne par run_id)

Deux modes de scheduling :
  - FRIDAY  : indices frais (mis a jour jeudi), forecast J+7
  - TUESDAY : reajustement avec actuals du weekend/lundi, forecast J+4

Regles :
  - SUIVI uniquement optimise
  - NON SUIVI : JAMAIS touche, levier de perequation
  - POLCO : PVC impose (INTERMARCHE), reporte a part
  - Seuls les produits avec indices concurrents sont optimises
  - PVC arrondi sur paliers FL (X.X9)
"""

import logging
import time
from datetime import date, datetime
from typing import Dict, Optional

import pandas as pd

from src.config import (
    BASES,
    OBJECTIFS,
    OBJECTIFS_ANNUELS_ACTIF,
    PRICING_SCHEDULE,
    MODES,
    TVA_DEFAULT,
)
from src.output.databricks_writer import DatabricksWriter, generate_run_id

logger = logging.getLogger(__name__)


class PricingPipeline:
    """Orchestre le pipeline complet : forecast → perequation → MC iteratif → sortie."""

    def __init__(self):
        self.model = None

    def load_model(self):
        """Charge le modele de forecast."""
        from src.forecast.train import load_model
        self.model = load_model()
        logger.info("Modele charge.")

    def run(
        self,
        mode: str = "friday",
        execution_mode: str = "complet",
        run_date: Optional[date] = None,
        save: bool = True,
    ) -> Dict:
        """Point d'entree principal du pipeline.

        Parameters
        ----------
        mode : str
            'friday' ou 'tuesday' (scheduling).
        execution_mode : str
            'complet', 'simulation', ou 'suivis_only'.
        run_date : date, optional
            Date du run.
        save : bool
            Ecrire les resultats en tables.

        Returns
        -------
        dict
            recommendations, polco, perequation, portfolio_metrics, output_tables.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

        if run_date is None:
            run_date = date.today()

        schedule = PRICING_SCHEDULE.get(mode)
        if schedule is None:
            raise ValueError(f"Mode scheduling inconnu : {mode}")

        exec_cfg = MODES.get(execution_mode)
        if exec_cfg is None:
            raise ValueError(f"Mode execution inconnu : {execution_mode}")

        horizon = schedule["forecast_horizon_jours"]
        run_id = generate_run_id(mode)

        print("=" * 70)
        print(f"PRICING PIPELINE — {mode.upper()} ({execution_mode})")
        print(f"  Run ID     : {run_id}")
        print(f"  Date       : {run_date}")
        print(f"  Horizon    : J+{horizon}")
        print(f"  Description: {schedule['description']}")
        print("=" * 70)

        t0 = time.time()

        # ================================================================
        # STEP 1 — FORECAST
        # ================================================================
        print("\n--- STEP 1 : Forecast LightGBM ---")
        df_preds = self._run_forecast(horizon)
        print(f"  Predictions : {len(df_preds):,} lignes")

        if df_preds.empty:
            print("  Aucune prediction. Arret.")
            return {"recommendations": pd.DataFrame(), "polco": pd.DataFrame()}

        # ================================================================
        # STEP 2 — SEPARATION POPULATIONS
        # ================================================================
        print("\n--- STEP 2 : Separation SUIVI / NS / POLCO ---")
        df_suivi, df_ns, df_polco = self._separate_populations(df_preds, exec_cfg)
        print(f"  SUIVI      : {len(df_suivi):,}")
        print(f"  NON SUIVI  : {len(df_ns):,}")
        print(f"  POLCO      : {len(df_polco):,}")

        # ================================================================
        # STEP 3 — PEREQUATION (AVANT MC)
        # ================================================================
        print("\n--- STEP 3 : Perequation ---")
        df_pereq = {}
        for enseigne in OBJECTIFS:
            if exec_cfg.get("perequation", True):
                pereq = self._compute_perequation(df_suivi, df_ns, enseigne)
                df_pereq[enseigne] = pereq
                if pereq is not None and not pereq.empty:
                    mlni_ns_req = pereq["MLNI_NS_Requis"].mean()
                    print(f"  {enseigne} : MLNI NS requis = {mlni_ns_req:.2f}%")

        # ================================================================
        # STEP 3b — OBJECTIFS ANNUELS PAR BASE (si actif)
        # ================================================================
        all_objectifs_base = {}
        all_synthese_obj = {}

        if OBJECTIFS_ANNUELS_ACTIF:
            print("\n--- STEP 3b : Objectifs annuels MFIL/MADH par base ---")
            for enseigne in OBJECTIFS:
                df_obj_base = self._compute_objectifs_annuels(
                    df_suivi, df_ns, enseigne,
                )
                if df_obj_base is not None and not df_obj_base.empty:
                    all_objectifs_base[enseigne] = df_obj_base
                    from src.perequation.objectifs_annuels import build_synthese_objectifs
                    synth = build_synthese_objectifs(df_obj_base, enseigne)
                    all_synthese_obj[enseigne] = synth
                    obj_e = OBJECTIFS[enseigne]
                    print(f"  {enseigne} : {len(df_obj_base)} bases")
                    print(f"    MLNI annuel={obj_e['MLNI_TAUX']}%, "
                          f"MADH annuel={obj_e['MADH_TAUX_CIBLE']}%, "
                          f"MFIL annuel={obj_e['MFIL_TAUX_CIBLE']}%")
                    print(f"    Alertes : {synth.get('nb_alertes', 0)} bases")

        # ================================================================
        # STEP 4 — MONTE CARLO ITERATIF (par enseigne)
        # ================================================================
        print("\n--- STEP 4 : Optimisation Monte Carlo iterative ---")
        all_recos = []
        all_portfolio = {}

        for enseigne in OBJECTIFS:
            df_ens = df_suivi[df_suivi["Enseigne"] == enseigne] if "Enseigne" in df_suivi.columns else df_suivi

            if df_ens.empty:
                print(f"  {enseigne} : aucun produit SUIVI.")
                continue

            # Injecter les cibles ajustees par base si actif
            if OBJECTIFS_ANNUELS_ACTIF and enseigne in all_objectifs_base:
                from src.perequation.objectifs_annuels import inject_cibles_ajustees
                df_ens = inject_cibles_ajustees(df_ens, all_objectifs_base[enseigne])
                print(f"  {enseigne} : cibles ajustees injectees dans {len(df_ens):,} produits.")

            df_ns_ens = df_ns[df_ns["Enseigne"] == enseigne] if "Enseigne" in df_ns.columns else df_ns

            print(f"\n  --- {enseigne} ({len(df_ens):,} produits) ---")
            df_reco, portfolio = self._run_optimization_iterative(
                df_ens, df_ns_ens, enseigne,
            )

            # Valider la decomposition MFIL + MADH = MLNI
            if OBJECTIFS_ANNUELS_ACTIF and not df_reco.empty:
                from src.perequation.objectifs_annuels import validate_decomposition
                required_cols = {"MADH_Val", "MFIL_Val", "MLNI_Val"}
                if required_cols.issubset(set(df_reco.columns)):
                    df_reco = validate_decomposition(df_reco)
                    n_ok = df_reco["decomposition_ok"].sum()
                    print(f"    Decomposition MFIL+MADH=MLNI : {n_ok}/{len(df_reco)} OK")

            if not df_reco.empty:
                all_recos.append(df_reco)
                all_portfolio[enseigne] = portfolio
                print(f"    Converge : {'OUI' if portfolio['converged'] else 'NON'}")
                print(f"    Iterations : {portfolio['n_iterations']}")
                print(f"    MLNI Portfolio : {portfolio['MLNI_Portfolio']:.2f}%")
                indice_ca = portfolio.get('Indice_CA')
                if indice_ca and not pd.isna(indice_ca):
                    print(f"    Indice CA : {indice_ca:.1f}")

        df_reco_all = pd.concat(all_recos, ignore_index=True) if all_recos else pd.DataFrame()

        if not df_reco_all.empty:
            n_feasible = df_reco_all["Is_Feasible"].sum()
            print(f"\n  Total : {len(df_reco_all):,} produits, {n_feasible:,} faisables")

        # ================================================================
        # STEP 5 — SORTIE
        # ================================================================
        output_tables = {}
        if save:
            print("\n--- STEP 5 : Ecriture tables ---")

            schema = exec_cfg.get("schema_sortie")
            writer = DatabricksWriter(run_id=run_id, mode=execution_mode, schema_override=schema)

            tables_to_write = {}

            if not df_reco_all.empty:
                tables_to_write["recommandations"] = df_reco_all

            if not df_polco.empty:
                tables_to_write["polco"] = df_polco

            for enseigne, pereq in df_pereq.items():
                if pereq is not None and not pereq.empty:
                    if "perequation" not in tables_to_write:
                        tables_to_write["perequation"] = pereq
                    else:
                        tables_to_write["perequation"] = pd.concat(
                            [tables_to_write["perequation"], pereq], ignore_index=True,
                        )

            if all_portfolio:
                df_portfolio = pd.DataFrame([
                    {k: v for k, v in m.items() if k != "history"}
                    for m in all_portfolio.values()
                ])
                tables_to_write["portfolio_metrics"] = df_portfolio

            # Objectifs annuels par base
            if all_objectifs_base:
                df_obj_all = pd.concat(
                    list(all_objectifs_base.values()), ignore_index=True,
                )
                tables_to_write["objectifs_annuels_base"] = df_obj_all

            output_tables = writer.write_all(tables_to_write)

            for name, table in output_tables.items():
                print(f"  {name:<25s} : {table}")

        elapsed = time.time() - t0
        print(f"\n  Total : {elapsed:.1f}s")

        return {
            "recommendations": df_reco_all,
            "polco": df_polco,
            "perequation": df_pereq,
            "portfolio_metrics": all_portfolio,
            "output_tables": output_tables,
            "stats": {
                "run_id": run_id,
                "mode": mode,
                "execution_mode": execution_mode,
                "date": run_date,
                "n_predictions": len(df_preds),
                "n_suivi": len(df_suivi),
                "n_ns": len(df_ns),
                "n_polco": len(df_polco),
                "n_optimized": len(df_reco_all),
                "elapsed_s": elapsed,
            },
        }

    def _run_forecast(self, horizon_days: int) -> pd.DataFrame:
        """Execute le forecast."""
        if self.model is None:
            self.load_model()

        from src.forecast.predict import predict_horizon
        from src.features.historique import load_historique

        # Charger l'historique recent
        # En production, source = table Databricks
        # Pour le dev, on cherche un fichier local
        from src.config import PROJECT_ROOT
        local_path = PROJECT_ROOT / "data" / "models" / "df_product_v2.parquet"
        if local_path.exists():
            df_history = pd.read_parquet(local_path)
            df_history["Date"] = pd.to_datetime(df_history["Date"])
            df_history["CODBAS"] = df_history["CODBAS"].astype(str)
            max_date = df_history["Date"].max()
            cutoff = max_date - pd.Timedelta(days=60)
            df_history = df_history[df_history["Date"] >= cutoff]
        else:
            logger.warning("Dataset historique non trouve.")
            return pd.DataFrame()

        return predict_horizon(self.model, df_history, horizon_days)

    def _separate_populations(
        self,
        df_preds: pd.DataFrame,
        exec_cfg: dict,
    ):
        """Separe en SUIVI / NON SUIVI / POLCO."""
        from src.config import PRODUITS_CONFIG

        col_suivi = PRODUITS_CONFIG["suivis"]["filtre_colonne"]
        val_suivi = PRODUITS_CONFIG["suivis"]["valeur_filtre"]
        val_ns = PRODUITS_CONFIG["non_suivis"]["valeur_filtre"]

        if col_suivi in df_preds.columns:
            df_suivi = df_preds[df_preds[col_suivi] == val_suivi].copy()
            df_ns = df_preds[df_preds[col_suivi] == val_ns].copy()
        else:
            df_suivi = df_preds.copy()
            df_ns = pd.DataFrame()

        # POLCO : flag is_polco
        if "is_polco" in df_preds.columns:
            df_polco = df_preds[df_preds["is_polco"] == 1].copy()
            # Retirer les POLCO des SUIVIS
            df_suivi = df_suivi[df_suivi.get("is_polco", 0) != 1].copy()
        else:
            df_polco = pd.DataFrame()

        if not exec_cfg.get("inclure_polco", True):
            df_polco = pd.DataFrame()

        return df_suivi, df_ns, df_polco

    def _compute_perequation(
        self,
        df_suivi: pd.DataFrame,
        df_ns: pd.DataFrame,
        enseigne: str,
    ):
        """Calcule la perequation pour une enseigne."""
        from src.perequation.matrix import build_perequation_matrix
        from src.perequation.targets import compute_perequation_targets

        # Combiner SUIVI et NS pour la matrice
        required_cols = {"CA_TTC", "MLNI_Val", "Maille_Suivi", "Semaine_FFL", "CODBAS", "Enseigne"}

        for name, df in [("SUIVI", df_suivi), ("NS", df_ns)]:
            missing = required_cols - set(df.columns)
            if missing:
                logger.warning("Perequation %s : colonnes manquantes %s.", name, missing)
                return pd.DataFrame()

        df_combined = pd.concat([df_suivi, df_ns], ignore_index=True)

        matrix = build_perequation_matrix(df_combined)
        if matrix.empty:
            return pd.DataFrame()

        return compute_perequation_targets(matrix, enseigne)

    def _compute_objectifs_annuels(
        self,
        df_suivi: pd.DataFrame,
        df_ns: pd.DataFrame,
        enseigne: str,
    ):
        """Calcule les objectifs annuels MFIL/MADH ajustes par base."""
        from src.perequation.objectifs_annuels import compute_objectifs_par_base

        # Combiner SUIVI + NS pour avoir les marges totales par base
        df_all = pd.concat([df_suivi, df_ns], ignore_index=True)
        if "Enseigne" in df_all.columns:
            df_all = df_all[df_all["Enseigne"] == enseigne]

        if df_all.empty:
            return pd.DataFrame()

        coef_tva = 1.0 + TVA_DEFAULT / 100.0

        # Calculer les marges si pas deja presentes
        if "MLNI_Val" not in df_all.columns:
            if "CA_HT" in df_all.columns and "Val_Achat" in df_all.columns:
                df_all["MLNI_Val"] = df_all["CA_HT"] - df_all["Val_Achat"]
            elif "CA_TTC" in df_all.columns and "Val_Achat" in df_all.columns:
                df_all["MLNI_Val"] = df_all["CA_TTC"] / coef_tva - df_all["Val_Achat"]
        if "MADH_Val" not in df_all.columns:
            if "CA_HT" in df_all.columns and "Val_Cession" in df_all.columns:
                df_all["MADH_Val"] = df_all["CA_HT"] - df_all["Val_Cession"]
        if "MFIL_Val" not in df_all.columns:
            if "Val_Cession" in df_all.columns and "Val_Achat" in df_all.columns:
                df_all["MFIL_Val"] = df_all["Val_Cession"] - df_all["Val_Achat"]

        # Agreger par base
        agg_cols = {}
        for col in ["CA_TTC", "CA_HT", "Val_Cession", "Val_Achat", "MLNI_Val", "MADH_Val", "MFIL_Val"]:
            if col in df_all.columns:
                agg_cols[col] = "sum"

        if not agg_cols:
            return pd.DataFrame()

        df_base = df_all.groupby("CODBAS").agg(agg_cols).reset_index()

        # Calculer les taux par base
        if "CA_TTC" in df_base.columns and "MLNI_Val" in df_base.columns:
            df_base["MLNI_Taux"] = (
                df_base["MLNI_Val"] / df_base["CA_TTC"] * 100
            ).where(df_base["CA_TTC"] > 0, 0)
        if "CA_TTC" in df_base.columns and "MADH_Val" in df_base.columns:
            df_base["MADH_Taux"] = (
                df_base["MADH_Val"] / df_base["CA_TTC"] * 100
            ).where(df_base["CA_TTC"] > 0, 0)
        if "Val_Cession" in df_base.columns and "MFIL_Val" in df_base.columns:
            df_base["MFIL_Taux"] = (
                df_base["MFIL_Val"] / df_base["Val_Cession"] * 100
            ).where(df_base["Val_Cession"] > 0, 0)

        return compute_objectifs_par_base(df_base, enseigne)

    def _run_optimization_iterative(
        self,
        df_suivi: pd.DataFrame,
        df_ns: pd.DataFrame,
        enseigne: str,
    ):
        """Lance l'optimisation iterative pour une enseigne."""
        from src.optimization.optimizer import PriceOptimizer

        optimizer = PriceOptimizer()

        # Preparer les colonnes requises pour le MC
        df_mc = df_suivi.copy()

        # Renommer pour matcher l'interface du MC
        rename_map = {}
        if "CODE_PRODUIT_STANDARD" in df_mc.columns:
            rename_map["CODE_PRODUIT_STANDARD"] = "CODE_PRODUIT"
        if "Val_Achat" in df_mc.columns:
            rename_map["Val_Achat"] = "PA"
        if "PVC" in df_mc.columns and "PVC_Current" not in df_mc.columns:
            rename_map["PVC"] = "PVC_Current"

        if rename_map:
            df_mc = df_mc.rename(columns=rename_map)

        # Colonnes requises
        obj = OBJECTIFS[enseigne]
        col_conc = obj["colonne_prix_concurrent"]
        if col_conc in df_mc.columns and "Prix_Concurrent" not in df_mc.columns:
            df_mc["Prix_Concurrent"] = df_mc[col_conc]

        return optimizer.optimize_iterative(
            df_products=df_mc,
            df_ns=df_ns,
            enseigne=enseigne,
        )


def run_pricing(
    mode: str = "friday",
    execution_mode: str = "complet",
    run_date: Optional[date] = None,
    save: bool = True,
) -> Dict:
    """Point d'entree fonctionnel."""
    pipeline = PricingPipeline()
    return pipeline.run(
        mode=mode,
        execution_mode=execution_mode,
        run_date=run_date,
        save=save,
    )


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "friday"
    exec_mode = sys.argv[2] if len(sys.argv) > 2 else "complet"
    run_pricing(mode=mode, execution_mode=exec_mode)
