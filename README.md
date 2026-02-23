# Retail Price Optimization — SCAFLF Fruits & Legumes

Optimisation des prix de vente consommateur (PVC) pour les Fruits & Legumes
de la SCAFLF (ITM / INTERMARCHE + NETTO), 19 bases logistiques.

## Architecture

```
retail-price-optimization-mlops/
├── config/
│   └── business_rules.yaml       # Regles metier (modifiable sans code)
├── src/
│   ├── config.py                  # Loader YAML → constantes Python
│   ├── features/                  # Chargement donnees, indices, POLCO, meteo
│   ├── forecast/                  # LightGBM : train, predict, evaluate
│   ├── perequation/               # Equilibrage marge SUIVI / NON SUIVI
│   ├── optimization/              # Monte Carlo : simulator, scorer, optimizer
│   ├── output/                    # Writer tables Databricks (Delta)
│   └── pipeline/                  # Orchestrateur principal
├── tests/                         # Tests unitaires (pytest)
├── notebooks/                     # Analyses exploratoires
└── pyproject.toml                 # Dependencies & config projet
```

## Pipeline Pricing

Flux complet bi-hebdomadaire (vendredi + mardi) :

```
1. FORECAST         LightGBM V2 → predictions Qte/CA par produit x base
                    |
2. SEPARATION       SUIVI (optimise) / NON SUIVI (passthrough) / POLCO (impose)
                    |
3. PEREQUATION      Calcul MLNI NS requis → cible pour le Monte Carlo
                    |                        (se calcule AVANT le MC)
4. MONTE CARLO      Boucle iterative par enseigne :
   ITERATIF           a. Generer 10K scenarios (PVC, PC) sur paliers FL (X.X9)
                      b. Scorer : marge 50% + indice 35% + lissage 15%
                      c. Verification portfolio (indice CA + MLNI global)
                      d. Si non convergent → re-iterer (max 5 iterations)
                    |
5. SORTIE           Tables Databricks en append (partitionne par run_id)
```

## Regles metier cles

| Regle | Valeur |
|-------|--------|
| INTERMARCHE MLNI cible | 34.75% |
| INTERMARCHE indice cible (vs E.Leclerc) | 102 |
| NETTO MLNI cible | 29.25% |
| NETTO indice cible (vs Lidl) | 100 |
| TVA F&L | 5.5% |
| Monte Carlo scenarios | 10 000 / produit x base |
| Paliers PVC | X.X9 (pas 0.10) |
| Scheduling | Vendredi (J+7) + Mardi (J+4) |

## Installation

```bash
pip install -e ".[dev]"
```

## Tests

```bash
pytest tests/ -v
```

## Execution

```bash
# Run vendredi (complet)
python -m src.pipeline.run_pricing friday complet

# Run mardi (reajustement)
python -m src.pipeline.run_pricing tuesday complet

# Simulation (dry run, pas d'ecriture prod)
python -m src.pipeline.run_pricing friday simulation
```

## Configuration

Toutes les regles metier sont dans `config/business_rules.yaml`.
Modifiable sans toucher au code. Les 13 sections couvrent :

1. **Taxonomie produits** : SUIVI / NON SUIVI / POLCO
2. **Enseignes & objectifs** : MLNI, indices par enseigne
3. **Formules financieres** : 3 marges SCAFLF (MADH, MFIL, MLNI)
4. **Perequation** : equilibrage SUIVI/NS, granularite DAX
5. **Monte Carlo** : scenarios, scoring, convergence
6. **Contraintes prix** : bornes, arrondis FL, stabilite
7. **Previsions** : parametres LightGBM, mix volume
8. **Bases logistiques** : 19 bases avec coordonnees GPS
9. **Sortie Databricks** : 6 tables Delta en append
10. **Scheduling** : vendredi/mardi
11. **Modes** : complet / simulation / suivis_only
12. **Meteo** : API Open-Meteo
13. **Parametres techniques** : logging, run_id
