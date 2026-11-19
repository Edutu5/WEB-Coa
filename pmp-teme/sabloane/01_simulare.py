"""Cursul 1 - simularea variabilelor aleatoare. Rulabil ca atare."""

import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

rng = np.random.default_rng()        # pune un numar in paranteza daca vrei reproductibilitate

# --- esantion fara repetitie dintr-un .csv -----------------------------------
# with open("studenti.csv", encoding="utf-8") as f:
#     lista = [rand[0] for rand in csv.reader(f) if rand]
# alesi = random.sample(lista, 3)              # fara repetitie
# alesi = rng.choice(lista, 3, replace=False)  # varianta numpy

# --- distributiile uzuale ----------------------------------------------------
N = 10000
bernoulli = rng.random(N) < 0.3              # 1 cu probabilitate 0.3
binomial = rng.binomial(n=10, p=0.3, size=N)  # nr. de succese in 10 incercari
geometric = rng.geometric(p=0.3, size=N)      # nr. de incercari pana la primul succes
poisson = rng.poisson(lam=20, size=N)         # nr. de evenimente intr-un interval
exponential = rng.exponential(scale=1 / 3, size=N)   # ATENTIE: scale = 1/lambda
normal = rng.normal(loc=0, scale=1, size=N)
uniform = rng.uniform(0, 1, size=N)
zar = rng.integers(1, 7, size=N)              # capatul din dreapta e exclus
alegere = rng.choice([0, 1, 2], size=N, p=[0.2, 0.5, 0.3])

print("Binomial(10, 0.3): media", round(binomial.mean(), 3), "(teoretic 3.0)")
print("Exp(lambda=3):     media", round(exponential.mean(), 3), "(teoretic", round(1 / 3, 3), ")")

# --- joc care se repeta pana la un eveniment ---------------------------------
def joaca(p):
    """Numara pasii pana la primul succes si aduna un castig."""
    n = 0
    s = 0.0
    while True:
        n += 1
        if rng.random() < p:
            s += rng.integers(1, 7) - 3       # ce se intampla la succes
            return n, s
        s -= 0.5                               # ce se intampla la esec

rezultate = [joaca(0.5) for _ in range(10000)]
S = np.array([s for n, s in rezultate])
print("Media lui S:", round(S.mean(), 3))

# --- amestec de distributii --------------------------------------------------
# Alegem intai componenta, apoi generam din ea.
lam = np.array([3, 6, 4])
p = lam / lam.sum()                            # ponderi proportionale cu ratele
componenta = rng.choice(3, size=N, p=p)
X = rng.exponential(scale=1 / lam[componenta])
print("Amestec: media", round(X.mean(), 4), " deviatia", round(X.std(), 4))

# --- estimarea unei probabilitati prin simulare ------------------------------
print("P(X > 0.5) =", round((X > 0.5).mean(), 4))

# --- histograma (variabile discrete) -----------------------------------------
plt.hist(S, bins=30, density=True)
plt.title("Distributia lui S")
plt.savefig("hist.png", dpi=110, bbox_inches="tight")   # SALVEAZA inainte de show
plt.show()

# --- densitate aproximativa (variabile continue) -----------------------------
grila, densitate, _ = az.kde(X)                # pe arviz < 1.0: az.plot_kde(X)
plt.plot(grila, densitate)
plt.title("Densitatea lui X")
plt.savefig("kde.png", dpi=110, bbox_inches="tight")
plt.show()
