"""
Optimisation Monte Carlo des prix.

Boucle iterative :
  1. Perequation → cible MLNI NS
  2. Monte Carlo → scenarios (PVC, PC) par produit
  3. Scoring multi-criteres (marge + indice + lissage)
  4. Verification portfolio (indice CA comparable + MLNI global)
  5. Si non convergent → re-iterer avec bornes ajustees
"""
