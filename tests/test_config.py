"""Tests pour le loader config."""

import pytest


def test_config_loads():
    """Le YAML se charge sans erreur."""
    from src.config import CFG, OBJECTIFS, BASES
    assert isinstance(CFG, dict)
    assert len(OBJECTIFS) >= 2
    assert len(BASES) >= 19


def test_objectifs_enseignes():
    """Les objectifs par enseigne sont coherents."""
    from src.config import OBJECTIFS

    assert "INTERMARCHE" in OBJECTIFS
    assert "NETTO" in OBJECTIFS

    itm = OBJECTIFS["INTERMARCHE"]
    assert itm["MLNI_TAUX"] == 34.75
    assert itm["INDICE_CIBLE"] == 102.0
    assert itm["concurrent_reference"] == "E.Leclerc"

    netto = OBJECTIFS["NETTO"]
    assert netto["MLNI_TAUX"] == 29.25
    assert netto["INDICE_CIBLE"] == 100.0
    assert netto["concurrent_reference"] == "Lidl"


def test_monte_carlo_params():
    """Les parametres Monte Carlo sont valides."""
    from src.config import MC_N_SCENARIOS, MC_PVC_VARIATION_PCT, MC_SCORE_WEIGHTS

    assert MC_N_SCENARIOS == 10_000
    assert MC_PVC_VARIATION_PCT == 15.0
    assert abs(sum(MC_SCORE_WEIGHTS.values()) - 1.0) < 0.01


def test_fl_arrondi():
    """Les parametres d'arrondi FL sont corrects."""
    from src.config import FL_PVC_STEP, FL_PVC_SUFFIX

    assert FL_PVC_STEP == 0.10
    assert FL_PVC_SUFFIX == 9


def test_tva_default():
    """TVA F&L = 5.5%."""
    from src.config import TVA_DEFAULT
    assert TVA_DEFAULT == 5.5


def test_validation_passes():
    """La validation de la config passe sans erreur."""
    from src.config import validate_config
    # Ne doit pas lever d'exception
    validate_config()


def test_produits_taxonomy():
    """Les 3 populations produits sont definies."""
    from src.config import PRODUITS_CONFIG

    assert "suivis" in PRODUITS_CONFIG
    assert "non_suivis" in PRODUITS_CONFIG
    assert "polco" in PRODUITS_CONFIG

    assert PRODUITS_CONFIG["suivis"]["politique_prix"] == "optimisation"
    assert PRODUITS_CONFIG["non_suivis"]["politique_prix"] == "passthrough"
    assert PRODUITS_CONFIG["polco"]["politique_prix"] == "impose"


def test_tables_databricks():
    """Les tables de sortie sont definies."""
    from src.config import TABLES

    expected = {"recommandations", "polco", "perequation", "synthese",
                "decomposition_marge", "portfolio_metrics"}
    assert expected.issubset(set(TABLES.keys()))

    for key, cfg in TABLES.items():
        assert "nom" in cfg, f"Table {key} sans nom"
        assert "mode_ecriture" in cfg, f"Table {key} sans mode_ecriture"
        assert cfg["mode_ecriture"] == "append", f"Table {key} pas en append"
