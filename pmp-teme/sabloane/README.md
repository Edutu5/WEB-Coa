# Şabloane şi formulare — PMP

Cod rulabil şi formule, organizate pe cursuri. Materia de examen: **primele 5 cursuri**. Cursul 6 (PyMC) **nu intră**; la inferenţă Bayesiană intră doar partea teoretică, cu `scipy.stats` şi `arviz`.

| Fişier | Curs | Conţinut |
|---|---|---|
| `01_simulare.py` | 1 | distribuţii, eşantionare, jocuri repetate, amestecuri, histogramă, KDE |
| `02_bayes.py` | 1 | Bayes numeric, probabilităţi condiţionate prin simulare, Simpson cu pandas |
| `03_retele_bayesiene.py` | 2 | `pgmpy`: DAG, CPD-uri, `check_model`, interogări |
| `04_retele_markov.py` | 3 | `pyAgrum`: MRF, factori, inferenţă; energie Gibbs şi ICM |
| `05_hmm.py` | 4 | Forward şi Viterbi implementate de mână |
| `06_inferenta_bayesiana.py` | 5 | conjugare Beta–Binomial şi Gamma–Poisson, HDI, predicţie |

Fiecare se rulează direct: `python3 01_simulare.py`.

---

# Importurile — ce, unde şi pentru ce

```python
import csv                   # citirea unei liste dintr-un .csv
import random                # random.sample(lista, k) - esantion fara repetitie
import numpy as np           # vectori, generator aleator, matrice
import matplotlib.pyplot as plt
import pandas as pd          # tabele: groupby, pivot
import networkx as nx        # grafuri neorientate, grile, desen
import arviz as az           # az.kde, az.hdi
from scipy import stats      # beta, gamma, binom, expon, norm
from scipy.special import betaln

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import pyagrum as gum
from pyagrum.markov_random_field import ShaferShenoyMRFInference

from hmmlearn import hmm     # doar daca vrei varianta gata facuta pentru HMM
```

| Import | Folosit la | Pentru ce anume |
|---|---|---|
| `csv`, `random` | Tema 1, ex. 1 | citire listă + `random.sample(lista, k)` |
| `numpy as np` | peste tot | `rng = np.random.default_rng()`, operaţii vectorizate, măşti booleene |
| `matplotlib.pyplot as plt` | oriunde e grafic | `hist`, `plot`, `imshow`, `savefig` |
| `pandas as pd` | Tema 2, ex. 3 | `DataFrame`, `groupby`, `pivot` — paradoxul Simpson |
| `networkx as nx` | Temele 3, 4 | `Graph`, `grid_2d_graph`, `is_tree`, `neighbors`, `draw` |
| `arviz as az` | Tema 1 ex. 3, Cursul 5 | `az.kde` (densitate), `az.hdi` (interval) |
| `scipy.stats` | Tema 3 ex. 3, Tema 6 | `binom.pmf`, `beta`, `gamma`, `expon` |
| `scipy.special.betaln` | Tema 6, ex. 1 | predicţia Beta–Binomial, în logaritmi |
| `pgmpy.*` | Tema 3 | reţele Bayesiene |
| `pyagrum as gum` | Tema 4 | reţele Markov (MRF) |
| `hmmlearn` | Laborator 5 | `CategoricalHMM` — alternativă la Viterbi scris de mână |

**`pymc` nu intră în materie.**

---

# Cursul 1 — Incertitudine, Bayes, variabile aleatoare

**Axiome:** $P(\Omega) = 1$, $P(A) \ge 0$, iar pentru evenimente incompatibile $P(\bigcup A_i) = \sum P(A_i)$.

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Probabilitate condiţionată şi independenţă:**

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

$$A \perp B \iff P(A \cap B) = P(A)P(B) \iff P(A \mid B) = P(A)$$

**Regula produsului şi Bayes:**

$$P(A \cap B) = P(A \mid B)P(B) = P(B \mid A)P(A)$$

$$P(A \mid B) = \frac{P(B \mid A)\,P(A)}{P(B)}$$

**Formula probabilităţii totale** (numitorul lui Bayes):

$$P(B) = \sum_i P(B \mid A_i) P(A_i)$$

**Medie şi dispersie:**

$$\mathbb{E}[X] = \sum_i x_i P(X = x_i), \qquad \text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]$$

## Distribuţiile de reţinut

| Distribuţie | Model | Medie | Dispersie | NumPy |
|---|---|---|---|---|
| Bernoulli(p) | un singur succes/eşec | $p$ | $p(1-p)$ | `rng.random(N) < p` |
| Binomial(n, p) | succese în $n$ încercări | $np$ | $np(1-p)$ | `rng.binomial(n, p, N)` |
| Geometric(p) | încercări până la primul succes | $1/p$ | $(1-p)/p^2$ | `rng.geometric(p, N)` |
| Poisson(λ) | evenimente într-un interval | $\lambda$ | $\lambda$ | `rng.poisson(lam, N)` |
| Exponential(λ) | timp până la un eveniment | $1/\lambda$ | $1/\lambda^2$ | `rng.exponential(1/lam, N)` |
| Normal(μ, σ) | zgomot aditiv | $\mu$ | $\sigma^2$ | `rng.normal(mu, sigma, N)` |

**Rezultatul contraintuitiv de reţinut:** cu prevalenţă 0.0001 şi un test de 99% acurateţe, $P(\text{boală} \mid +) \approx 0.0098$. Un test foarte bun produce multe alarme false când evenimentul căutat e rar.

---

# Cursul 2 — Reţele Bayesiene

**Definiţie:** un DAG plus, pentru fiecare variabilă, o distribuţie condiţionată de părinţii ei.

$$P(X_1, \dots, X_n) = \prod_{i=1}^{n} P(X_i \mid \text{Pa}(X_i))$$

**Proprietatea Markov locală:** fiecare variabilă este independentă de nedescendenţii săi, condiţionat de părinţii săi.

## Cele trei structuri şi d-separarea

| Structură | Fără condiţionare | Condiţionăm pe B |
|---|---|---|
| lanţ $A \to B \to C$ | cale deschisă | **blocată** |
| furcă $A \leftarrow B \to C$ | cale deschisă | **blocată** |
| collider $A \to B \leftarrow C$ | **blocată** | deschisă |

**Colliderul funcţionează invers faţă de celelalte două.** Observarea unui *descendent* al colliderului deschide şi ea calea.

**Regula d-separării.** O cale e blocată de mulţimea observată $Z$ dacă are (i) un nod din $Z$ care nu e collider, sau (ii) un collider care nu e în $Z$ şi nu are niciun descendent în $Z$. Dacă toate căile dintre $X$ şi $Y$ sunt blocate, atunci $X \perp Y \mid Z$.

**Explaining away:** dacă alarma a pornit şi aflăm că a fost o defecţiune, probabilitatea unui incendiu scade — două cauze independente devin dependente după observarea efectului comun.

**Eliminarea variabilelor:** fixează evidenţa → înmulţeşte factorii → sumează peste variabilele ascunse → normalizează.

$$P(D, I \mid A{=}1) \propto P(D)P(I) \sum_{G,L,S} P(G \mid D,I) P(L \mid G) P(S \mid I) P(A{=}1 \mid L,S)$$

**Economia de parametri:** o distribuţie comună arbitrară pe $n$ variabile binare are $2^n - 1$ parametri; o variabilă binară cu $k$ părinţi binari are $2^k$.

**Atenţie:** observaţia $P(Y \mid X{=}x)$ nu este intervenţia $P(Y \mid do(X{=}x))$. Un DAG nu devine model cauzal doar fiindcă desenăm săgeţi.

---

# Cursul 3 — Reţele Markov (MRF)

**Proprietatea Markov locală:** $X_i \perp X_{V \setminus (\{i\} \cup N(i))} \mid X_{N(i)}$ — vecinii sunt tot ce contează.

**Separarea:** dacă toate drumurile de la $A$ la $C$ trec prin $B$, atunci $X_A \perp X_C \mid X_B$.

**Clică:** mulţime de noduri în care oricare două sunt conectate. Într-un lanţ sau o grilă, clicile maximale sunt **muchiile**.

**Hammersley–Clifford** (pentru distribuţii strict pozitive):

$$P(x) = \frac{1}{Z} \prod_{C \in \mathcal{C}} \psi_C(x_C), \qquad \psi_C > 0$$

**Potenţialele sunt ponderi, nu probabilităţi.** Nu sumează la 1; normalizarea se face o singură dată, global, prin $Z$.

**Distribuţia Gibbs:**

$$P(x) = \frac{1}{Z} e^{-E(x)}, \qquad Z = \sum_x e^{-E(x)} \qquad \Longrightarrow \qquad \arg\max_x P(x) = \arg\min_x E(x)$$

$Z$ nu depinde de $x$, deci nu trebuie calculat când compari configuraţii.

**Modelul Ising** ($x_i \in \{-1, +1\}$):

$$E(x) = -J \sum_{\{i,j\} \in E} x_i x_j - h \sum_i x_i$$

$J > 0$ favorizează vecinii egali; $h$ e un câmp extern.

**Energia de netezire a imaginilor:**

$$E(x; y) = \lambda \sum_i (x_i - y_i)^2 + \sum_{\{i,j\} \in E} (x_i - x_j)^2$$

Primul termen e **fidelitatea** faţă de observaţie, al doilea **regularitatea spaţială**; $1/\lambda$ este forţa netezirii. Pentru un pixel cu $k$ vecini, din care $d$ îl contrazic, inversarea se produce dacă $\lambda < 2d - k$.

**Energia Potts pentru etichetare:**

$$E(\ell) = \sum_i D_i(\ell_i) + \lambda \sum_{\{i,j\} \in E} \mathbf{1}_{\{\ell_i \ne \ell_j\}}$$

**Inferenţă marginală vs MAP:** „care e probabilitatea aici?" şi „care e configuraţia globală cea mai probabilă?" sunt întrebări diferite, cu răspunsuri în general diferite.

| Metodă | Idee | Limitare |
|---|---|---|
| ICM | minimizare coordonată | minime locale |
| Metropolis | eşantionare stochastică | timp de amestecare |
| Simulated annealing | temperatură descrescătoare | alegerea programului de răcire |

## MRF versus reţea Bayesiană

| Reţea Bayesiană | Reţea Markov |
|---|---|
| graf orientat aciclic | graf neorientat |
| factori condiţionaţi | potenţiale de clică |
| independenţe prin d-separare | independenţe prin separare |
| normalizare locală a CPD-urilor | normalizare globală prin $Z$ |

---

# Cursul 4 — Lanţuri Markov şi HMM

**Proprietatea Markov:** $P(S_{t+1} \mid S_t, S_{t-1}, \dots) = P(S_{t+1} \mid S_t)$.

**Lanţ Markov:** $\pi_i = P(S_1 = q_i)$ şi $a_{ij} = P(S_{t+1} = q_j \mid S_t = q_i)$. **Fiecare linie a lui $A$ sumează la 1.**

- traiectorie: $P(S_1{=}i, S_2{=}j, S_3{=}k) = \pi_i a_{ij} a_{jk}$
- evoluţie: $p_t = \pi A^{t-1}$; $A^k$ dă tranziţiile în $k$ paşi
- distribuţie staţionară: $p^* A = p^*$
- durata într-o stare e geometrică: $P(D = d) = a_{ii}^{d-1}(1 - a_{ii})$, $\mathbb{E}[D] = \frac{1}{1 - a_{ii}}$
- estimare din date: $\hat{a}_{ij} = N_{ij} / \sum_k N_{ik}$

**HMM — trei obiecte, nu două:** $\pi$, $A$ şi matricea de emisie $b_j(o) = P(O_t = o \mid X_t = j)$.

$$P(s_{1:T}, o_{1:T}) = \pi_{s_1} b_{s_1}(o_1) \prod_{t=2}^{T} a_{s_{t-1}s_t}\, b_{s_t}(o_t)$$

## Cele trei întrebări

| Întrebare | Algoritm | Recurenţa |
|---|---|---|
| cât de probabilă e secvenţa observată? | **Forward** | $\alpha_t(j) = b_j(o_t) \sum_i \alpha_{t-1}(i) a_{ij}$ |
| care e cel mai probabil traseu? | **Viterbi** | $\delta_t(j) = b_j(o_t) \max_i \{\delta_{t-1}(i) a_{ij}\}$ |
| cum învăţăm parametrii? | **Baum–Welch** (EM) | maxim local |

Iniţializare pentru amândouă: $\alpha_1(j) = \delta_1(j) = \pi_j b_j(o_1)$.

$$P(o_{1:T}) = \sum_j \alpha_T(j)$$

**Viterbi are nevoie şi de $\psi_t(j) = \arg\max_i \{\delta_{t-1}(i) a_{ij}\}$** pentru backtracking: $\delta$ spune *cât de bun* e traseul, $\psi$ spune *care* e.

**Backward şi netezire:** $\beta_T(i) = 1$, $\beta_t(i) = \sum_j a_{ij} b_j(o_{t+1}) \beta_{t+1}(j)$, iar

$$P(X_t = q_i \mid o_{1:T}) = \frac{\alpha_t(i)\beta_t(i)}{\sum_k \alpha_t(k)\beta_t(k)}$$

**Complexitate:** $O(TN^2)$ faţă de $N^T$ prin enumerare.

**De nu confundat:** $P(o_{1:T})$ este o sumă peste toate traseele; $\max P(x, o)$ este probabilitatea comună a **unui singur** traseu şi a observaţiilor. Nu se compară direct. Probabilitatea condiţionată a traseului este $P(x^* \mid o) = P(x^*, o) / P(o)$.

**Stare marginală optimă ≠ secvenţă optimă:** alegerea celei mai probabile stări separat la fiecare moment nu dă, în general, secvenţa global cea mai probabilă.

---

# Cursul 5 — Inferenţă Bayesiană

$$p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{p(y)} \ \propto\ p(y \mid \theta)\,p(\theta)$$

| Termen | Nume | Ce e |
|---|---|---|
| $p(\theta)$ | prior | ce credeam înainte de date |
| $p(y \mid \theta)$ | verosimilitate | ce spun datele |
| $p(\theta \mid y)$ | posterior | ce credem după actualizare |
| $p(y) = \int p(y \mid \theta)p(\theta)d\theta$ | evidenţă | constantă de normalizare |

**Factorii care nu conţin $\theta$ se pot arunca** — se simplifică exact, apărând şi la numărător, şi la numitor. Dacă expresia rămasă are forma nucleului unei distribuţii cunoscute, posteriorul **este** acea distribuţie: constanta se reconstituie din condiţia de normalizare.

## Conjugare Beta–Binomial

$$\theta \sim \text{Beta}(\alpha, \beta), \quad Y \mid \theta \sim \text{Binomial}(n, \theta) \quad \Longrightarrow \quad \theta \mid y \sim \text{Beta}(\alpha + y,\ \beta + n - y)$$

## Conjugare Gamma–Poisson

$$\lambda \sim \text{Gamma}(\alpha, \beta), \quad K_i \mid \lambda \sim \text{Poisson}(\lambda) \quad \Longrightarrow \quad \lambda \mid k \sim \text{Gamma}\Big(\alpha + \textstyle\sum k_i,\ \beta + n\Big)$$

Se citeşte: $\alpha$ numără evenimente, $\beta$ numără unităţi de timp. Priorul `Gamma(2, 0.1)` înseamnă „am văzut 2 evenimente în 0.1 unităţi de timp".

## Estimări punctuale

| Estimare | Ce maximizează | Beta$(a,b)$ | Gamma$(\alpha,\beta)$ |
|---|---|---|---|
| **MLE** | verosimilitatea | $y/n$ | $\sum k_i / n$ |
| **MAP** (modul) | posteriorul | $\dfrac{a-1}{a+b-2}$ | $\dfrac{\alpha-1}{\beta}$ |
| **Media** | — | $\dfrac{a}{a+b}$ | $\dfrac{\alpha}{\beta}$ |

Estimarea optimă depinde de funcţia de pierdere: pierdere pătratică → **media**, pierdere absolută → **mediana**.

## Intervale

- **Interval credibil central** de 95%: `dist.ppf([0.025, 0.975])` — 2.5% în fiecare coadă.
- **HDI**: cel mai *scurt* interval cu masa cerută — `az.hdi(esantion, prob=0.94)`. Pentru distribuţii unimodale, capetele au densităţi egale.

Un HDI de 94% înseamnă că 94% din masa posterioară a parametrului se află acolo. **Nu** înseamnă că parametrul variază de la un experiment la altul.

## Predicţia posterioară

$$p(\tilde{y} \mid y) = \int p(\tilde{y} \mid \theta)\, p(\theta \mid y)\, d\theta$$

Integrează incertitudinea asupra parametrului, nu foloseşte o singură estimare punctuală. Pentru Beta–Binomial, probabilitatea ca din $m$ noi observaţii niciuna să nu fie succes:

$$P(X = 0 \mid y) = \frac{B(a,\ b + m)}{B(a,\ b)}$$

---

# Capcane care costă timp la examen

**`scipy.stats.gamma` foloseşte scara, nu rata.** Dacă enunţul dă $\beta$ ca rată, scrie `stats.gamma(alfa, scale=1/beta)`. Fără inversare primeşti un rezultat complet greşit şi nicio eroare.

**`rng.integers(1, 7)` exclude capătul din dreapta** — dă 1…6, corect pentru un zar.

**`rng.exponential(scale=...)`** cere media, adică $1/\lambda$.

**Ordinea coloanelor în `TabularCPD`:** cu `evidence=["B","T"]`, ultima variabilă variază cel mai repede: $(0,0), (0,1), (1,0), (1,1)$.

**`fillWith` în pyAgrum poate transpune matricea.** Completează celulă cu celulă, cu dicţionar: `factor({i,j})[{i: vi, j: vj}] = valoare`.

**`BayesianNetwork` e depreciat** în pgmpy 1.x — numele curent e `DiscreteBayesianNetwork`.

**ArviZ ≥ 1.0 a schimbat API-ul:**

| arviz < 1.0 | arviz ≥ 1.0 |
|---|---|
| `az.plot_kde(x)` | `grila, dens, _ = az.kde(x)` apoi `plt.plot(...)` |
| `az.plot_posterior(idata)` | `az.plot_dist(idata, ci_kind="hdi", ci_prob=0.94)` |
| `az.hdi(x, hdi_prob=0.94)` | `az.hdi(x, prob=0.94)` |
| `az.summary(idata, hdi_prob=0.95)` | `az.summary(idata, ci_prob=0.95, ci_kind="hdi")` |

**`plt.savefig` ÎNAINTE de `plt.show()`** — altfel figura se închide şi rămâi fără fişier. Examenul cere graficele salvate.

```python
plt.savefig("ex1.png", dpi=110, bbox_inches="tight")
plt.show()
```

**Nu confunda probabilitatea comună cu cea condiţionată.** La Viterbi, scorul final e $P(x^*, o)$, nu $P(x^* \mid o)$.

**Egalităţi la `argmax`:** când două trasee au acelaşi scor, `np.argmax` alege primul indice. O bibliotecă poate alege altul — ambele răspunsuri sunt corecte.
