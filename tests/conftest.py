"""
Fixtures pytest partagees.

Donnees synthetiques pour les tests unitaires.
Pas besoin de donnees reelles — on genere des DataFrames coherents
avec la structure attendue par les modules.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_historique():
    """DataFrame historique financier synthetique (10 produits x 2 bases x 30 jours)."""
    np.random.seed(42)

    produits = [f"PROD_{i:04d}" for i in range(10)]
    bases = ["26", "55"]  # LYON, AMILLY
    enseignes = ["INTERMARCHE", "NETTO"]
    dates = pd.date_range("2026-01-01", periods=30, freq="D")

    rows = []
    for produit in produits:
        for base_idx, base in enumerate(bases):
            enseigne = enseignes[base_idx % 2]
            pa = round(np.random.uniform(0.5, 3.0), 2)  # prix d'achat

            for dt in dates:
                pvc = round(pa * np.random.uniform(1.3, 2.0), 2)
                qte = max(1, int(np.random.normal(50, 20)))
                ca_ttc = pvc * qte
                coef_tva = 1.055
                ca_ht = ca_ttc / coef_tva
                pc = round(pa * np.random.uniform(1.0, 1.5), 2)
                val_cession = pc * qte
                val_achat = pa * qte

                rows.append({
                    "CODE_PRODUIT_STANDARD": produit,
                    "CODBAS": base,
                    "Enseigne": enseigne,
                    "Date": dt,
                    "Maille_Suivi": "SUIVI" if np.random.random() > 0.3 else "NON SUIVI",
                    "PVC": pvc,
                    "PA": pa,
                    "Qte_Reelle": qte,
                    "CA_TTC": round(ca_ttc, 2),
                    "CA_HT": round(ca_ht, 2),
                    "Val_Cession": round(val_cession, 2),
                    "Val_Achat": round(val_achat, 2),
                })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_indices():
    """DataFrame indices prix concurrents synthetique."""
    np.random.seed(42)

    produits = [f"PROD_{i:04d}" for i in range(10)]
    bases = ["26", "55"]

    rows = []
    for produit in produits:
        for base in bases:
            rows.append({
                "CODE_PRODUIT_STANDARD": produit,
                "CODBAS": base,
                "Prix_Concurrent": round(np.random.uniform(1.0, 5.0), 2),
                "Prix_Moyen_LCL": round(np.random.uniform(1.0, 5.0), 2),
                "Prix_Moyen_LIDL": round(np.random.uniform(0.8, 4.0), 2),
                "Coef_Ponderation": round(np.random.uniform(0.5, 2.0), 2),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def sample_products_for_mc():
    """DataFrame produits prets pour le Monte Carlo."""
    np.random.seed(42)
    n = 5

    return pd.DataFrame({
        "CODE_PRODUIT": [f"PROD_{i:04d}" for i in range(n)],
        "CODBAS": ["26"] * n,
        "Enseigne": ["INTERMARCHE"] * n,
        "PA": np.random.uniform(0.5, 2.0, n).round(2),
        "PVC_Current": np.random.uniform(1.5, 4.0, n).round(2),
        "Prix_Concurrent": np.random.uniform(1.5, 5.0, n).round(2),
        "Indice_Cible": [102.0] * n,
        "MLNI_Cible": [34.75] * n,
        "Qte_Pred": np.random.randint(10, 200, n),
    })


@pytest.fixture
def sample_perequation_data():
    """DataFrame pour la matrice de perequation."""
    return pd.DataFrame({
        "Semaine_FFL": ["SEMAINE FFL S06-26"] * 4,
        "CODBAS": ["26", "26", "55", "55"],
        "Enseigne": ["INTERMARCHE", "INTERMARCHE", "NETTO", "NETTO"],
        "Maille_Suivi": ["SUIVI", "NON SUIVI", "SUIVI", "NON SUIVI"],
        "CA_TTC": [100000, 30000, 80000, 20000],
        "CA_HT": [94787, 28436, 75829, 18957],
        "Val_Cession": [70000, 22000, 58000, 15000],
        "Val_Achat": [60000, 18000, 52000, 12000],
        "MLNI_Val": [34787, 10436, 23829, 6957],
        "MADH_Val": [24787, 6436, 17829, 3957],
        "MFIL_Val": [10000, 4000, 6000, 3000],
        "Qte_Reelle": [5000, 1500, 4000, 1000],
    })
