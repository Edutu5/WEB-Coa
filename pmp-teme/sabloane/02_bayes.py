"""Cursul 1 - probabilitati conditionate si formula lui Bayes. Rulabil ca atare."""

import numpy as np
import pandas as pd

rng = np.random.default_rng()

# --- Bayes numeric, cu formula probabilitatii totale -------------------------
# P(H | E) = P(E|H) P(H) / [ P(E|H) P(H) + P(E|H^c) P(H^c) ]
def bayes(p_h, p_e_dat_h, p_e_dat_nonh):
    numarator = p_e_dat_h * p_h
    numitor = numarator + p_e_dat_nonh * (1 - p_h)
    return numarator / numitor

print("Test medical: P(boala | pozitiv) =", round(bayes(0.0001, 0.99, 0.01), 4))
print("Spam, un cuvant:   P(spam | oferta) =", round(bayes(0.3, 0.8, 0.15), 4))
# doua indicii independente conditionat: se inmultesc verosimilitatile in fiecare clasa
print("Spam, doua cuvinte:P(spam | ambele) =",
      round(bayes(0.3, 0.8 * 0.6, 0.15 * 0.10), 4))

# --- estimarea probabilitatilor conditionate prin simulare -------------------
N = 100000
X = rng.integers(1, 7, N)
Y = rng.integers(1, 7, N)

A = X + Y >= 10                    # evenimentele se scriu ca masti booleene
B = X % 2 == 0

print()
print("P(A | B) =", round(A[B].mean(), 4))     # frecventa lui A in interiorul conditiei
print("P(A si B) =", round((A & B).mean(), 4))
print("P(A)*P(B) =", round(A.mean() * B.mean(), 4), " -> egale <=> independente")

# --- paradoxul lui Simpson cu pandas -----------------------------------------
date = pd.DataFrame({
    "grup":      ["A", "A", "B", "B"],
    "strat":     ["s1", "s2", "s1", "s2"],
    "succese":   [18, 48, 55, 15],
    "total":     [20, 80, 70, 30],
})
date["rata"] = date["succese"] / date["total"]
print()
print(date)

# agregam peste straturi si RECALCULAM raportul (nu media ratelor)
total = date.groupby("grup")[["succese", "total"]].sum()
total["rata"] = total["succese"] / total["total"]
print()
print(total)
print("Inversarea apare cand ponderile straturilor difera mult intre grupuri.")
