"""Tests pour le module perequation."""

import numpy as np
import pytest


class TestPerequationTargets:
    """Tests pour le calcul des cibles de perequation."""

    def test_mlni_ns_requis_basic(self):
        """Cas nominal : calcul MLNI NS requis."""
        from src.perequation.targets import compute_mlni_ns_requis

        # CA_suivi=70%, CA_ns=30%, MLNI_suivi=30%, cible=34.75%
        result = compute_mlni_ns_requis(
            ca_suivi=70000,
            ca_ns=30000,
            mlni_suivi_taux=30.0,
            mlni_cible_taux=34.75,
        )

        # MLNI_ns = (34.75*100000 - 30*70000) / 30000 * 100
        #         = (3475000 - 2100000) / 30000 * 100
        #         = 1375000 / 30000 * 100 = 4583.33... → ~45.83%
        # Mais avec nos units : (34.75/100*100000 - 30/100*70000) / 30000 *100
        #  = (34750 - 21000) / 30000 * 100 = 13750/30000*100 = 45.83%
        assert abs(result - 45.8333) < 0.01

    def test_mlni_ns_requis_ca_ns_zero(self):
        """CA NS = 0 → NaN."""
        from src.perequation.targets import compute_mlni_ns_requis

        result = compute_mlni_ns_requis(
            ca_suivi=100000,
            ca_ns=0,
            mlni_suivi_taux=30.0,
            mlni_cible_taux=34.75,
        )
        assert np.isnan(result)

    def test_mlni_ns_requis_already_met(self):
        """Objectif deja atteint par les SUIVIS → NS requis < cible."""
        from src.perequation.targets import compute_mlni_ns_requis

        result = compute_mlni_ns_requis(
            ca_suivi=90000,
            ca_ns=10000,
            mlni_suivi_taux=36.0,  # au-dessus de la cible
            mlni_cible_taux=34.75,
        )
        # Le NS requis sera plus faible car les SUIVIS compensent
        assert result < 34.75

    def test_prix_bornes_from_mlni(self):
        """Derivation des bornes de prix depuis un objectif MLNI."""
        from src.perequation.targets import compute_prix_bornes_from_mlni

        result = compute_prix_bornes_from_mlni(
            pa=1.0,
            mlni_cible_taux=34.75,
        )

        assert result["feasible"]
        assert result["PVC_min"] > 1.0  # PVC_min > PA
        assert result["PC_min"] == 1.0  # PC_min = PA


class TestPerequationMatrix:
    """Tests pour la matrice de perequation."""

    def test_build_matrix(self, sample_perequation_data):
        """Construction de la matrice pivotee."""
        from src.perequation.matrix import build_perequation_matrix

        matrix = build_perequation_matrix(sample_perequation_data)

        assert not matrix.empty
        assert "CA_TTC_SUIVI" in matrix.columns
        assert "CA_TTC_NS" in matrix.columns
        assert "CA_TTC_TOTAL" in matrix.columns

        # Total = SUIVI + NS
        row = matrix.iloc[0]
        assert abs(row["CA_TTC_TOTAL"] - row["CA_TTC_SUIVI"] - row["CA_TTC_NS"]) < 0.01


class TestConstraints:
    """Tests pour la validation des contraintes de marge."""

    def test_validate_margins_ok(self):
        """Cas faisable : toutes les marges OK."""
        from src.perequation.constraints import validate_margins

        result = validate_margins(
            pvc=2.99,
            pc=1.80,
            pa=1.20,
            enseigne="INTERMARCHE",
        )

        assert result["MADH_ok"]
        assert result["MFIL_ok"]
        assert result["all_ok"]

    def test_validate_margins_mfil_negative(self):
        """PC < PA → MFIL negatif."""
        from src.perequation.constraints import validate_margins

        result = validate_margins(
            pvc=2.99,
            pc=1.00,  # < PA
            pa=1.20,
        )

        assert not result["MFIL_ok"]
        assert not result["all_ok"]

    def test_compute_prix_bornes(self):
        """Bornes de prix coherentes."""
        from src.perequation.constraints import compute_prix_bornes

        result = compute_prix_bornes(
            pa=1.0,
            prix_concurrent=2.50,
            indice_cible=102.0,
            mlni_cible=34.75,
        )

        assert result["PVC_min"] > 0
        assert result["PVC_max"] == pytest.approx(2.55, abs=0.01)  # 2.50 * 102/100
        assert result["PC_min"] == 1.0  # = PA
        # La faisabilite depend des valeurs concretes
        assert isinstance(result["feasible"], (bool, np.bool_))
