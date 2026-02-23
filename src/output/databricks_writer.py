"""
Writer Databricks : ecriture des resultats en tables Delta.

Les tables sont en mode append avec partitionnement par run_id.
Pas de CSV — tout en tables en dur pour archivage et requetabilite BI.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from src.config import SORTIE_CONFIG, TABLES

logger = logging.getLogger(__name__)


def generate_run_id(mode: str) -> str:
    """Genere un identifiant unique de run.

    Format : PRICING_{mode}_{YYYYMMDD}_{HHmmss}
    """
    now = datetime.now()
    return f"PRICING_{mode.upper()}_{now.strftime('%Y%m%d_%H%M%S')}"


class DatabricksWriter:
    """Ecrit les resultats en tables Delta sur Databricks.

    En environnement local (pas de Spark), ecrit en parquet
    avec la meme structure de partitionnement.
    """

    def __init__(
        self,
        run_id: str,
        mode: str = "complet",
        schema_override: Optional[str] = None,
    ):
        self.run_id = run_id
        self.mode = mode
        self.schema = schema_override or SORTIE_CONFIG.get("tables", {}).get(
            "recommandations", {}
        ).get("schema", "pricing")
        self._spark = self._get_spark()

    @staticmethod
    def _get_spark():
        """Tente de recuperer une session Spark active."""
        try:
            from pyspark.sql import SparkSession
            return SparkSession.getActiveSession()
        except ImportError:
            return None

    def _add_run_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute les colonnes de metadata du run."""
        df = df.copy()
        df["run_id"] = self.run_id
        df["date_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["mode"] = self.mode
        return df

    def write_table(
        self,
        df: pd.DataFrame,
        table_key: str,
    ) -> str:
        """Ecrit un DataFrame dans une table Databricks.

        Parameters
        ----------
        df : pd.DataFrame
            Donnees a ecrire.
        table_key : str
            Cle de la table dans config (ex: "recommandations", "polco").

        Returns
        -------
        str
            Nom complet de la table ecrite.
        """
        table_config = TABLES.get(table_key)
        if table_config is None:
            raise ValueError(f"Table inconnue : {table_key}. Tables disponibles : {list(TABLES.keys())}")

        table_name = f"{self.schema}.{table_config['nom']}"
        mode_ecriture = table_config.get("mode_ecriture", "append")

        # Ajouter metadata
        df = self._add_run_metadata(df)

        if self._spark is not None:
            # Ecriture Databricks Delta
            self._write_delta(df, table_name, mode_ecriture, table_config)
        else:
            # Fallback local : ecriture parquet
            self._write_local(df, table_key, table_config)

        logger.info(
            "Table %s : %d lignes ecrites (mode=%s, run_id=%s).",
            table_name, len(df), mode_ecriture, self.run_id,
        )
        return table_name

    def _write_delta(
        self,
        df: pd.DataFrame,
        table_name: str,
        mode: str,
        config: dict,
    ):
        """Ecriture en table Delta via Spark."""
        sdf = self._spark.createDataFrame(df)
        partition_cols = config.get("colonnes_partition", [])

        writer = sdf.write.format("delta").mode(mode)
        if partition_cols:
            writer = writer.partitionBy(*partition_cols)

        writer.saveAsTable(table_name)

    def _write_local(
        self,
        df: pd.DataFrame,
        table_key: str,
        config: dict,
    ):
        """Fallback local : ecriture parquet partitionnee."""
        from pathlib import Path
        from src.config import PROJECT_ROOT

        output_dir = PROJECT_ROOT / "data" / "output" / table_key / f"run_id={self.run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / f"{config['nom']}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        logger.info("  Fallback local : %s", path)

    def write_all(
        self,
        results: Dict[str, pd.DataFrame],
    ) -> Dict[str, str]:
        """Ecrit tous les resultats en une fois.

        Parameters
        ----------
        results : dict
            {table_key: DataFrame} pour chaque table a ecrire.

        Returns
        -------
        dict
            {table_key: nom_table_ecrite}.
        """
        written = {}
        for key, df in results.items():
            if df is not None and not df.empty:
                table_name = self.write_table(df, key)
                written[key] = table_name
            else:
                logger.info("Table %s : vide, pas d'ecriture.", key)

        logger.info(
            "Run %s : %d tables ecrites sur %d.",
            self.run_id, len(written), len(results),
        )
        return written
