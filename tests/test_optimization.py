"""Tests pour le module d'optimisation Monte Carlo."""

import numpy as np
import pytest


class TestSimulator:
    """Tests pour le simulateur de scenarios."""

    def test_generate_scenarios_basic(self):
        """Genere des scenarios avec des parametres normaux."""
        from src.optimization.simulator import PriceSimulator

        sim = PriceSimulator(n_scenarios=100, seed=42)

        scenarios = sim.generate_scenarios(
            pa=1.0,
            pvc_current=2.49,
            prix_concurrent=2.50,
            indice_cible=102.0,
            mlni_cible=34.75,
        )

        assert scenarios.shape == (100, 2)
        assert scenarios[:, 0].min() > 0  # PVC > 0
        assert scenarios[:, 1].min() >= 0.99  # PC >= PA (≈1.0)

    def test_generate_scenarios_invalid_pa(self):
        """PA <= 0 → aucun scenario."""
        from src.optimization.simulator import PriceSimulator

        sim = PriceSimulator(n_scenarios=100)
        scenarios = sim.generate_scenarios(
            pa=0, pvc_current=2.49, prix_concurrent=2.50,
            indice_cible=102, mlni_cible=34.75,
        )
        assert scenarios.shape == (0, 2)

    def test_fl_grid(self):
        """Les paliers FL sont bien en X.X9."""
        from src.optimization.simulator import PriceSimulator

        paliers = PriceSimulator._generate_fl_grid(1.50, 3.00)

        assert len(paliers) > 0
        for p in paliers:
            cents = round(p * 100) % 10
            assert cents == 9, f"Palier {p} ne finit pas en 9"

    def test_fl_grid_known_values(self):
        """Paliers FL connus dans une plage."""
        from src.optimization.simulator import PriceSimulator

        paliers = PriceSimulator._generate_fl_grid(1.00, 2.00)
        expected = [1.09, 1.19, 1.29, 1.39, 1.49, 1.59, 1.69, 1.79, 1.89, 1.99]
        np.testing.assert_array_almost_equal(paliers, expected, decimal=2)

    def test_round_pvc_fl(self):
        """Arrondi au palier FL le plus proche."""
        from src.optimization.simulator import PriceSimulator

        assert PriceSimulator._round_pvc_fl(1.45) == 1.49
        assert PriceSimulator._round_pvc_fl(2.00) == 1.99
        assert PriceSimulator._round_pvc_fl(0.03) == 0.09

    def test_generate_batch(self, sample_products_for_mc):
        """Batch de produits genere des scenarios pour chaque produit."""
        from src.optimization.simulator import PriceSimulator

        sim = PriceSimulator(n_scenarios=50, seed=42)
        results = sim.generate_batch(sample_products_for_mc)

        assert len(results) == 5
        for key, scenarios in results.items():
            assert scenarios.shape[1] == 2


class TestScorer:
    """Tests pour le scoring multi-criteres."""

    def test_score_batch(self):
        """Score entre 0 et 1."""
        from src.optimization.scorer import PriceScorer

        scorer = PriceScorer()
        scenarios = np.array([
            [2.49, 1.50],
            [2.99, 1.80],
            [1.99, 1.20],
        ])

        scores = scorer.score_batch(
            scenarios=scenarios,
            pa=1.0,
            prix_concurrent=2.50,
            indice_cible=102.0,
            pvc_precedent=2.49,
        )

        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_select_best(self):
        """Selection du meilleur scenario."""
        from src.optimization.scorer import PriceScorer

        scorer = PriceScorer()
        scenarios = np.array([
            [2.49, 1.50],
            [2.99, 1.80],
            [1.99, 1.20],
        ])
        scores = np.array([0.3, 0.8, 0.5])

        result = scorer.select_best(scenarios, scores, pa=1.0)

        assert result["PVC_Optimal"] == 2.99
        assert result["Score"] == pytest.approx(0.8, abs=0.01)


class TestOptimizer:
    """Tests pour l'optimiseur."""

    def test_optimize_product(self):
        """Optimisation d'un seul produit."""
        from src.optimization.optimizer import PriceOptimizer

        opt = PriceOptimizer()
        result = opt.optimize_product(
            pa=1.0,
            pvc_current=2.49,
            prix_concurrent=2.50,
            indice_cible=102.0,
            mlni_cible=34.75,
        )

        assert "PVC_Optimal" in result
        assert "Score" in result
        assert result["PVC_Optimal"] > 0

    def test_optimize_batch(self, sample_products_for_mc):
        """Optimisation batch."""
        from src.optimization.optimizer import PriceOptimizer

        opt = PriceOptimizer()
        df_reco = opt.optimize_batch(sample_products_for_mc)

        assert len(df_reco) == 5
        assert "PVC_Optimal" in df_reco.columns
        assert "Delta_PVC" in df_reco.columns
        assert "Is_Feasible" in df_reco.columns
