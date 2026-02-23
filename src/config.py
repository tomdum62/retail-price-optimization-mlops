"""
Loader central : charge config/business_rules.yaml et expose les constantes.

Toute la configuration metier est dans le YAML.
Ce module le charge, valide les champs critiques, et expose des objets
Python directement utilisables par les autres modules.

Usage :
    from src.config import CFG, OBJECTIFS, BASES, MC_CONFIG
"""

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "business_rules.yaml"


def load_yaml(path: Path | str | None = None) -> Dict[str, Any]:
    """Charge et retourne le dictionnaire YAML brut.

    Parameters
    ----------
    path : Path, optional
        Chemin vers le fichier YAML (defaut : config/business_rules.yaml).

    Returns
    -------
    dict
        Contenu brut du YAML.

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    """
    if path is None:
        path = CONFIG_PATH
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config introuvable : {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    logger.info("Config chargee : %s", path)
    return data


# ---------------------------------------------------------------------------
# Chargement au module-level
# ---------------------------------------------------------------------------
_RAW: Dict[str, Any] = load_yaml()

# Alias raccourcis
CFG = _RAW

# --- Enseignes & objectifs ---
ENSEIGNES: Dict[str, Dict[str, Any]] = _RAW["enseignes"]

OBJECTIFS: Dict[str, Dict[str, Any]] = {
    enseigne: {
        "MLNI_TAUX": params["MLNI_TAUX"],
        "MADH_TAUX_CIBLE": params.get("MADH_TAUX_CIBLE", 0.0),
        "MFIL_TAUX_CIBLE": params.get("MFIL_TAUX_CIBLE", 0.0),
        "MADH_TAUX_MIN": params["MADH_TAUX_MIN"],
        "MFIL_TAUX_MIN": params["MFIL_TAUX_MIN"],
        "INDICE_CIBLE": params["INDICE_CIBLE"],
        "concurrent_reference": params["concurrent_reference"],
        "colonne_prix_concurrent": params["colonne_prix_concurrent"],
    }
    for enseigne, params in ENSEIGNES.items()
}

# --- Objectifs annuels ---
OBJECTIFS_ANNUELS_CONFIG: Dict[str, Any] = _RAW.get("objectifs_annuels", {})
OBJECTIFS_ANNUELS_ACTIF: bool = OBJECTIFS_ANNUELS_CONFIG.get("actif", False)
AMORTISSEMENT_CONFIG: Dict[str, Any] = OBJECTIFS_ANNUELS_CONFIG.get("amortissement", {})
AMORTISSEMENT_COEF: float = AMORTISSEMENT_CONFIG.get("coef", 0.40)
AMORTISSEMENT_ECART_MAX: float = AMORTISSEMENT_CONFIG.get("ecart_max_correction_pts", 3.0)
TOLERANCE_CONFIG: Dict[str, Any] = OBJECTIFS_ANNUELS_CONFIG.get("tolerance", {})

# --- Formules financieres ---
TVA_DEFAULT: float = _RAW["formules"]["tva_default_pct"]
MARGE_ABERRANTE_MIN: float = _RAW["formules"]["marge_aberrante_min_pct"]
MARGE_ABERRANTE_MAX: float = _RAW["formules"]["marge_aberrante_max_pct"]
FILIERE_HORTICOLE_CODE: int = _RAW["formules"]["filiere_horticole_code"]

# --- Monte Carlo ---
MC_CONFIG: Dict[str, Any] = _RAW["monte_carlo"]
MC_N_SCENARIOS: int = MC_CONFIG["n_scenarios"]
MC_PVC_VARIATION_PCT: float = MC_CONFIG["pvc_variation_pct"]
MC_SEED: int = MC_CONFIG["seed"]
MC_SCORE_WEIGHTS: Dict[str, float] = MC_CONFIG["scoring"]["poids"]

# --- Contraintes prix ---
FL_PVC_STEP: float = _RAW["contraintes_prix"]["arrondis"]["pas"]
FL_PVC_SUFFIX: int = _RAW["contraintes_prix"]["arrondis"]["suffixe"]
STABILITE_ECART_MAX_PCT: float = _RAW["contraintes_prix"]["stabilite"]["ecart_max_pct"]

# --- Bases logistiques ---
BASES: Dict[str, Dict[str, Any]] = _RAW["bases"]

# --- Perequation ---
PEREQUATION_CONFIG: Dict[str, Any] = _RAW["perequation"]
PEREQUATION_CHUNKSIZE: int = PEREQUATION_CONFIG.get("chunksize", 500_000)

# --- Previsions ---
FORECAST_CONFIG: Dict[str, Any] = _RAW["previsions"]["forecast"]
LGBM_PARAMS: Dict[str, Any] = FORECAST_CONFIG["parametres_lgbm"]

# --- Scheduling ---
PRICING_SCHEDULE: Dict[str, Dict[str, Any]] = _RAW["scheduling"]

# --- Sortie ---
SORTIE_CONFIG: Dict[str, Any] = _RAW["sortie"]
TABLES: Dict[str, Dict[str, Any]] = SORTIE_CONFIG["tables"]

# --- Produits ---
PRODUITS_CONFIG: Dict[str, Any] = _RAW["produits"]
POLCO_ENSEIGNE: str = PRODUITS_CONFIG["polco"]["enseigne"]
POLCO_FORMAT: Dict[str, Any] = PRODUITS_CONFIG["polco"]["format"]

# --- Modes ---
MODES: Dict[str, Dict[str, Any]] = _RAW["modes"]

# --- Meteo ---
METEO_CONFIG: Dict[str, Any] = _RAW["meteo"]


# ---------------------------------------------------------------------------
# Validation minimale
# ---------------------------------------------------------------------------
def validate_config() -> None:
    """Verifie la coherence des parametres critiques.

    Raises
    ------
    ValueError
        Si un parametre est incoherent.
    """
    errors = []

    # TVA positive
    if TVA_DEFAULT <= 0:
        errors.append(f"TVA doit etre > 0, got {TVA_DEFAULT}")

    # Objectifs par enseigne
    for enseigne, obj in OBJECTIFS.items():
        if obj["MLNI_TAUX"] <= 0:
            errors.append(f"{enseigne}: MLNI_TAUX doit etre > 0")
        if obj["INDICE_CIBLE"] <= 0:
            errors.append(f"{enseigne}: INDICE_CIBLE doit etre > 0")
        # Validation decomposition MFIL + MADH (les taux ne s'additionnent pas
        # directement a cause des denominateurs differents, mais les cibles doivent
        # etre coherentes avec le MLNI global)
        if obj["MADH_TAUX_CIBLE"] > 0 and obj["MFIL_TAUX_CIBLE"] > 0:
            if obj["MADH_TAUX_CIBLE"] > obj["MLNI_TAUX"]:
                errors.append(
                    f"{enseigne}: MADH_TAUX_CIBLE ({obj['MADH_TAUX_CIBLE']}) "
                    f"> MLNI_TAUX ({obj['MLNI_TAUX']})"
                )

    # Monte Carlo
    if MC_N_SCENARIOS < 100:
        errors.append(f"MC_N_SCENARIOS trop faible : {MC_N_SCENARIOS}")

    poids_total = sum(MC_SCORE_WEIGHTS.values())
    if abs(poids_total - 1.0) > 0.01:
        errors.append(f"Poids scoring ne somment pas a 1.0 : {poids_total}")

    # FL arrondi
    if FL_PVC_SUFFIX not in (0, 5, 9):
        errors.append(f"Suffixe FL invalide : {FL_PVC_SUFFIX}")

    if errors:
        msg = "Erreurs de configuration :\n  " + "\n  ".join(errors)
        raise ValueError(msg)

    logger.info("Config validee (%d enseignes, %d bases).", len(OBJECTIFS), len(BASES))


# Valider au chargement
validate_config()
