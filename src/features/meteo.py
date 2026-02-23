"""
Features meteo par base logistique.

Utilise l'API Open-Meteo (gratuite) pour enrichir les donnees
de previsions meteorologiques localisees par base.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from src.config import BASES, METEO_CONFIG

logger = logging.getLogger(__name__)


def fetch_meteo_forecast(
    base_code: str,
    start_date: str,
    end_date: str,
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Recupere les previsions meteo pour une base logistique.

    Parameters
    ----------
    base_code : str
        Code de la base logistique (ex: "26" pour LYON).
    start_date, end_date : str
        Periode (format YYYY-MM-DD).
    variables : list, optional
        Variables meteo a recuperer. Defaut depuis config.

    Returns
    -------
    pd.DataFrame
        Colonnes : Date, CODBAS, + variables meteo.
    """
    if variables is None:
        variables = METEO_CONFIG["variables"]

    base = BASES.get(base_code)
    if base is None:
        logger.warning("Base %s inconnue, pas de meteo.", base_code)
        return pd.DataFrame()

    lat, lon = base["lat"], base["lon"]
    url = METEO_CONFIG["api_url"]

    try:
        import requests

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(variables),
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "Europe/Paris",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        df = pd.DataFrame(daily)
        if "time" in df.columns:
            df = df.rename(columns={"time": "Date"})
            df["Date"] = pd.to_datetime(df["Date"])

        df["CODBAS"] = base_code
        return df

    except Exception as e:
        logger.warning("Erreur meteo base %s : %s", base_code, e)
        return pd.DataFrame()


def fetch_meteo_all_bases(
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Recupere la meteo pour toutes les bases.

    Returns
    -------
    pd.DataFrame
        Donnees meteo concatenees pour toutes les bases.
    """
    frames = []
    for code in BASES:
        df = fetch_meteo_forecast(code, start_date, end_date)
        if not df.empty:
            frames.append(df)

    if not frames:
        logger.warning("Aucune donnee meteo recuperee.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)
    logger.info(
        "Meteo chargee : %d bases, %d lignes.",
        df_all["CODBAS"].nunique(), len(df_all),
    )
    return df_all
