"""Cursul 2 - retele Bayesiene cu pgmpy. Rulabil ca atare."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. structura: lista de muchii (parinte, copil)
model = DiscreteBayesianNetwork([("B", "T"), ("B", "X"), ("B", "D"), ("T", "D")])

# 2. tabelele conditionate
# fara parinti: o singura coloana, cate un rand pe stare
cpd_b = TabularCPD("B", 2, [[0.9], [0.1]])

# un parinte: coloanele in ordinea starilor parintelui
cpd_t = TabularCPD("T", 2, [[0.7, 0.2],      # P(T=0 | B=0), P(T=0 | B=1)
                            [0.3, 0.8]],     # P(T=1 | B=0), P(T=1 | B=1)
                   evidence=["B"], evidence_card=[2])

cpd_x = TabularCPD("X", 2, [[0.9, 0.1],
                            [0.1, 0.9]], evidence=["B"], evidence_card=[2])

# doi parinti: ULTIMA variabila din evidence variaza cel mai repede.
# Pentru evidence=["B","T"], coloanele sunt (B,T) = (0,0), (0,1), (1,0), (1,1).
cpd_d = TabularCPD("D", 2, [[0.9, 0.5, 0.6, 0.1],
                            [0.1, 0.5, 0.4, 0.9]],
                   evidence=["B", "T"], evidence_card=[2, 2])

model.add_cpds(cpd_b, cpd_t, cpd_x, cpd_d)
print("Model valid:", model.check_model())    # verifica si ca fiecare coloana sumeaza la 1

# 3. independentele citite din graf
print("Independente locale pentru X:", model.local_independencies("X"))

# 4. inferenta
infer = VariableElimination(model)
print()
print("marginala  P(B):")
print(infer.query(["B"]))
print("conditionata  P(B | T=1, D=1):")
print(infer.query(["B"], evidence={"T": 1, "D": 1}))
print("comuna  P(B, T | D=1):")
print(infer.query(["B", "T"], evidence={"D": 1}))

# valoarea numerica a unei stari, daca ai nevoie de ea intr-un calcul
rez = infer.query(["B"], evidence={"T": 1, "D": 1})
print("P(B=1 | T=1, D=1) =", round(rez.values[1], 4))

# 5. desenul grafului
nx.draw(model, pos=nx.circular_layout(model), with_labels=True, node_size=2000, alpha=0.5)
plt.savefig("retea.png", dpi=110, bbox_inches="tight")
plt.show()

# --- CPD construit cu o formula, nu scris de mana ----------------------------
# Util cand o variabila are multe stari (de exemplu, binomiala).
# from scipy.stats import binom
# coloane = [binom.pmf(np.arange(13), 2 * n, p) for p in (4/7, 0.5) for n in range(1, 7)]
# cpd = TabularCPD("M", 13, np.array(coloane).T, evidence=["S", "N"], evidence_card=[2, 6])
