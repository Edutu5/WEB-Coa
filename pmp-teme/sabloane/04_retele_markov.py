"""Cursul 3 - retele Markov (MRF) cu pyAgrum, plus ICM. Rulabil ca atare."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pyagrum as gum
from pyagrum.markov_random_field import ShaferShenoyMRFInference

noduri = ["S1", "S2", "S3"]
muchii = [("S1", "S2"), ("S2", "S3")]
stari = ["Soare", "Ploaie", "Nori"]

# 1. graful neorientat
graf = nx.Graph()
graf.add_nodes_from(noduri)
graf.add_edges_from(muchii)
print("Vecinii lui S2:", sorted(graf.neighbors("S2")))
print("Arbore:", nx.is_tree(graf))

# 2. modelul MRF
p = {"Soare": 0.5, "Ploaie": 0.3, "Nori": 0.2}
Psi = {"Soare":  {"Soare": 0.7, "Ploaie": 0.2, "Nori": 0.1},
       "Ploaie": {"Soare": 0.2, "Ploaie": 0.6, "Nori": 0.2},
       "Nori":   {"Soare": 0.1, "Ploaie": 0.3, "Nori": 0.6}}

mrf = gum.MarkovRandomField()
for s in noduri:
    mrf.add(gum.LabelizedVariable(s, "", stari))

for s in noduri:                      # factor unar
    mrf.addFactor([s])
    for v in stari:
        mrf.factor({s})[{s: v}] = p[v]

for i, j in muchii:                   # factor pereche
    mrf.addFactor([i, j])             # completeaza celula cu celula, cu dictionar:
    for vi in stari:                  # fillWith poate transpune matricea
        for vj in stari:
            mrf.factor({i, j})[{i: vi, j: vj}] = Psi[vi][vj]

print(mrf)

# 3. inferenta marginala
ie = ShaferShenoyMRFInference(mrf)
ie.setEvidence({"S1": "Ploaie"})
ie.makeInference()
post = ie.posterior("S3")
print(post)
best = post.argmax()
print("MAP marginal la S3:", stari[best[0][0]["S3"]], round(best[1], 4))

# 4. configuratia MAP comuna (diferita, in general, de maximele marginale)
tinte = ["S2", "S3"]
ie2 = ShaferShenoyMRFInference(mrf)
ie2.setEvidence({"S1": "Ploaie"})
ie2.addJointTarget(set(tinte))
ie2.makeInference()
comun = ie2.jointPosterior(set(tinte))
amax = comun.argmax()
print("MAP comun:", {s: stari[amax[0][0][s]] for s in tinte}, round(amax[1], 4))

# --- energie si ICM pe o grila (restaurare de imagini) -----------------------
# Gibbs: P(x) = (1/Z) exp(-E(x)), deci argmax P = argmin E. Z nu trebuie calculat.
y = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]])     # imaginea observata
grila = nx.grid_2d_graph(*y.shape)                   # vecinatate de ordinul intai

def energie(x, lam):
    return lam * np.sum((x - y) ** 2) + sum((x[a] - x[b]) ** 2 for a, b in grila.edges())

def energie_locala(x, i, j, v, lam):
    """Doar termenii care contin pixelul (i,j) - restul se simplifica la comparatie."""
    e = lam * (v - y[i, j]) ** 2
    for vecin in grila.neighbors((i, j)):
        e += (v - x[vecin]) ** 2
    return e

def icm(lam):
    """Optimizeaza un pixel o data; gaseste un minim LOCAL."""
    x = y.copy()
    schimbat = True
    while schimbat:
        schimbat = False
        for nod in grila.nodes():
            i, j = nod
            bun = int(np.argmin([energie_locala(x, i, j, v, lam) for v in (0, 1)]))
            if bun != x[i, j]:
                x[i, j] = bun
                schimbat = True
    return x

print()
for lam in [0.3, 1.0, 3.0]:
    x = icm(lam)
    print("lambda =", lam, " E =", round(energie(x, lam), 2), " x* =", x.tolist())

plt.imshow(icm(1.0), cmap="gray", vmin=0, vmax=1)
plt.savefig("mrf.png", dpi=110, bbox_inches="tight")
plt.show()
