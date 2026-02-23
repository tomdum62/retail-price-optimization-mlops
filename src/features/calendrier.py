"""
Features calendaires : semaine FFL, jours feries, saison.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Jours feries fixes (France)
JOURS_FERIES_FIXES = [
    (1, 1),   # Jour de l'an
    (5, 1),   # Fete du travail
    (5, 8),   # Victoire 1945
    (7, 14),  # Fete nationale
    (8, 15),  # Assomption
    (11, 1),  # Toussaint
    (11, 11), # Armistice
    (12, 25), # Noel
]


def add_calendar_features(df: pd.DataFrame, col_date: str = "Date") -> pd.DataFrame:
    """Ajoute les features calendaires au DataFrame.

    Features ajoutees :
        - jour_semaine (0=lundi...6=dimanche)
        - semaine_iso
        - mois
        - Semaine_FFL (identifiant SEMAINE FFL SXX-YY)
        - is_weekend
        - is_ferie
        - saison (0=hiver, 1=printemps, 2=ete, 3=automne)
    """
    dt = pd.to_datetime(df[col_date])

    df["jour_semaine"] = dt.dt.dayofweek
    df["semaine_iso"] = dt.dt.isocalendar().week.astype(int)
    df["mois"] = dt.dt.month
    df["annee"] = dt.dt.year

    # Semaine FFL
    iso = dt.dt.isocalendar()
    df["Semaine_FFL"] = (
        "SEMAINE FFL S"
        + iso.week.astype(str).str.zfill(2)
        + "-"
        + (iso.year % 100).astype(str)
    )

    # Weekend
    df["is_weekend"] = (df["jour_semaine"] >= 5).astype(int)

    # Jours feries
    mois_jour = list(zip(dt.dt.month, dt.dt.day))
    df["is_ferie"] = [1 if (m, d) in JOURS_FERIES_FIXES else 0 for m, d in mois_jour]

    # Saison meteorologique
    df["saison"] = np.where(
        dt.dt.month.isin([12, 1, 2]), 0,    # hiver
        np.where(
            dt.dt.month.isin([3, 4, 5]), 1,  # printemps
            np.where(
                dt.dt.month.isin([6, 7, 8]), 2,  # ete
                3,  # automne
            ),
        ),
    )

    return df
