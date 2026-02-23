"""Tests pour le module objectifs_annuels."""

import numpy as np
import pandas as pd
import pytest


class TestValidateDecomposition:
    """Tests pour la validation MFIL_Val + MADH_Val = MLNI_Val."""

    def test_decomposition_ok(self):
        """Cas nominal : decomposition coherente."""
        from src.perequation.objectifs_annuels import validate_decomposition

        df = pd.DataFrame({
            "MADH_Val": [100.0, 200.0, 50.0],
            "MFIL_Val": [80.0, 150.0, 30.0],
            "MLNI_Val": [180.0, 350.0, 80.0],
        })
        result = validate_decomposition(df)
        assert result["decomposition_ok"].all()
        assert (result["decomposition_ecart"].abs() < 0.01).all()

    def test_decomposition_ko(self):
        """Decomposition incoherente : MFIL + MADH != MLNI."""
        from src.perequation.objectifs_annuels import validate_decomposition

        df = pd.DataFrame({
            "MADH_Val": [100.0],
            "MFIL_Val": [80.0],
            "MLNI_Val": [200.0],  # 100+80=180 != 200
        })
        result = validate_decomposition(df)
        assert not result["decomposition_ok"].iloc[0]
        assert abs(result["decomposition_ecart"].iloc[0] - (-20.0)) < 0.01

    def test_decomposition_missing_columns(self):
        """Colonnes manquantes : flag KO sans erreur."""
        from src.perequation.objectifs_annuels import validate_decomposition

        df = pd.DataFrame({"MADH_Val": [100.0]})  # manque MFIL_Val, MLNI_Val
        result = validate_decomposition(df)
        assert not result["decomposition_ok"].iloc[0]


class TestCibleAjustee:
    """Tests pour le calcul de la cible ajustee avec amortissement."""

    def test_sur_cible(self):
        """Ecart dans la tolerance : pas de correction."""
        from src.perequation.objectifs_annuels import compute_cible_ajustee

        result = compute_cible_ajustee(
            cible_annuelle=22.0,
            taux_observe=22.5,  # +0.5pt, dans tolerance ±1pt
        )
        assert result["situation"] == "sur_cible"
        assert result["cible_ajustee"] == 22.0
        assert result["correction_effective"] == 0.0

    def test_en_avance(self):
        """Base en avance : on relache la cible."""
        from src.perequation.objectifs_annuels import compute_cible_ajustee

        result = compute_cible_ajustee(
            cible_annuelle=22.0,
            taux_observe=25.0,  # +3pts en avance
            coef_amortissement=0.40,
            ecart_max_pts=3.0,
        )
        assert result["situation"] == "en_avance"
        assert result["ecart_vs_annuel"] == 3.0
        # correction_brute = -3.0, amortie = -3.0*0.40 = -1.2
        assert abs(result["correction_effective"] - (-1.2)) < 0.01
        # cible_ajustee = 22.0 + (-1.2) = 20.8
        assert abs(result["cible_ajustee"] - 20.8) < 0.01

    def test_en_retard(self):
        """Base en retard : on durcit la cible."""
        from src.perequation.objectifs_annuels import compute_cible_ajustee

        result = compute_cible_ajustee(
            cible_annuelle=22.0,
            taux_observe=18.0,  # -4pts en retard
            coef_amortissement=0.40,
            ecart_max_pts=3.0,
        )
        assert result["situation"] == "en_retard"
        assert result["ecart_vs_annuel"] == -4.0
        # correction_brute = +4.0, amortie = 4.0*0.40 = 1.6
        assert abs(result["correction_effective"] - 1.6) < 0.01
        # cible_ajustee = 22.0 + 1.6 = 23.6
        assert abs(result["cible_ajustee"] - 23.6) < 0.01

    def test_anti_coup_de_volant_cap(self):
        """Ecart enorme : la correction est cappee a ecart_max."""
        from src.perequation.objectifs_annuels import compute_cible_ajustee

        result = compute_cible_ajustee(
            cible_annuelle=22.0,
            taux_observe=12.0,  # -10pts en retard
            coef_amortissement=0.40,
            ecart_max_pts=3.0,
        )
        assert result["situation"] == "en_retard"
        # correction_brute = +10.0, amortie = 10*0.40 = 4.0, cap = 3.0
        assert abs(result["correction_effective"] - 3.0) < 0.01
        # cible_ajustee = 22.0 + 3.0 = 25.0
        assert abs(result["cible_ajustee"] - 25.0) < 0.01

    def test_bornes_securite(self):
        """La cible ne depasse pas les bornes min/max."""
        from src.perequation.objectifs_annuels import compute_cible_ajustee

        result = compute_cible_ajustee(
            cible_annuelle=8.0,
            taux_observe=20.0,  # +12pts en avance
            coef_amortissement=0.40,
            ecart_max_pts=3.0,
            cible_min=6.0,
        )
        # cible_ajustee = 8.0 + (-3.0 cap) = 5.0, mais borne_min = 6.0
        assert result["cible_ajustee"] >= 6.0


class TestObjectifsParBase:
    """Tests pour compute_objectifs_par_base."""

    def test_basic(self):
        """Calcul des objectifs par base pour 2 bases."""
        from src.perequation.objectifs_annuels import compute_objectifs_par_base

        df = pd.DataFrame({
            "CODBAS": ["26", "55"],
            "MLNI_Taux": [36.0, 30.0],
            "MADH_Taux": [16.0, 10.0],
            "MFIL_Taux": [24.0, 16.0],
        })
        result = compute_objectifs_par_base(df, "INTERMARCHE")

        assert len(result) == 2
        assert "MLNI_cible_ajustee" in result.columns
        assert "MADH_cible_ajustee" in result.columns
        assert "MFIL_cible_ajustee" in result.columns
        assert "ecart_alerte" in result.columns

        # LYON (26) : MLNI 36% vs cible 34.75% => en avance
        lyon = result[result["CODBAS"] == "26"].iloc[0]
        assert lyon["MLNI_situation"] == "en_avance"

        # AMILLY (55) : MLNI 30% vs cible 34.75% => en retard
        amilly = result[result["CODBAS"] == "55"].iloc[0]
        assert amilly["MLNI_situation"] == "en_retard"


class TestInjectCiblesAjustees:
    """Tests pour inject_cibles_ajustees."""

    def test_injection(self):
        """Les cibles ajustees sont injectees dans les produits."""
        from src.perequation.objectifs_annuels import inject_cibles_ajustees

        df_products = pd.DataFrame({
            "CODE_PRODUIT": ["P1", "P2", "P3"],
            "CODBAS": ["26", "26", "55"],
        })
        df_obj = pd.DataFrame({
            "CODBAS": ["26", "55"],
            "MLNI_cible_ajustee": [33.5, 35.5],
            "MADH_cible_ajustee": [14.5, 16.0],
            "MFIL_cible_ajustee": [21.0, 23.0],
            "MLNI_situation": ["en_avance", "en_retard"],
            "MADH_situation": ["sur_cible", "en_retard"],
            "MFIL_situation": ["sur_cible", "en_retard"],
        })

        result = inject_cibles_ajustees(df_products, df_obj)

        assert len(result) == 3
        assert "MLNI_cible_ajustee" in result.columns

        # P1 et P2 sur base 26 => cible 33.5
        assert result.loc[result["CODE_PRODUIT"] == "P1", "MLNI_cible_ajustee"].iloc[0] == 33.5
        # P3 sur base 55 => cible 35.5
        assert result.loc[result["CODE_PRODUIT"] == "P3", "MLNI_cible_ajustee"].iloc[0] == 35.5


class TestSyntheseObjectifs:
    """Tests pour build_synthese_objectifs."""

    def test_synthese(self):
        """La synthese resume correctement les situations."""
        from src.perequation.objectifs_annuels import (
            compute_objectifs_par_base,
            build_synthese_objectifs,
        )

        df = pd.DataFrame({
            "CODBAS": ["26", "55", "20"],
            "MLNI_Taux": [36.0, 30.0, 34.75],
            "MADH_Taux": [16.0, 10.0, 15.0],
            "MFIL_Taux": [24.0, 16.0, 22.0],
        })
        df_obj = compute_objectifs_par_base(df, "INTERMARCHE")
        synth = build_synthese_objectifs(df_obj, "INTERMARCHE")

        assert synth["enseigne"] == "INTERMARCHE"
        assert synth["nb_bases"] == 3
        assert "MLNI_nb_en_avance" in synth
        assert "MFIL_nb_en_retard" in synth
