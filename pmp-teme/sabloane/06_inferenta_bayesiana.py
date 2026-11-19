"""Cursul 5 - inferenta Bayesiana prin conjugare. Rulabil ca atare. FARA PyMC."""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats
from scipy.special import betaln

# ============ Beta - Binomial (proportii, rate de defectare) =================
# prior  theta ~ Beta(a, b)
# date   Y | theta ~ Binomial(n, theta), observam y succese
# post   theta | y ~ Beta(a + y, b + n - y)
a0, b0, n, y = 2, 18, 50, 8
a, b = a0 + y, b0 + n - y
post = stats.beta(a, b)
prior = stats.beta(a0, b0)

print("Beta-Binomial: posterior Beta(", a, ",", b, ")")
print("  MLE  =", round(y / n, 4))                       # y / n
print("  MAP  =", round((a - 1) / (a + b - 2), 4))       # modul: (a-1)/(a+b-2)
print("  medie=", round(a / (a + b), 4))                 # a / (a+b)

lo, hi = post.ppf([0.025, 0.975])          # interval credibil CENTRAL de 95%
print("  interval central 95% = [", round(lo, 4), ",", round(hi, 4), "]")
print("  P(theta < 0.10 | y) =", round(post.cdf(0.10), 4))

# predictie: P(X = 0 din m noi) = B(a, b+m) / B(a, b), calculat prin logaritmi
m = 10
print("  P(0 defecte din", m, ") =", round(np.exp(betaln(a, b + m) - betaln(a, b)), 4))

# ============ Gamma - Poisson (rate de evenimente pe unitatea de timp) ======
# prior  lambda ~ Gamma(alfa, beta)        ATENTIE: beta este RATA
# date   K_i | lambda ~ Poisson(lambda), n observatii, suma k_i evenimente
# post   lambda | k ~ Gamma(alfa + suma k_i, beta + n)
# Citire: alfa numara evenimente, beta numara unitati de timp.
alfa0, beta0, n_ore, total = 18, 1, 10, 180
alfa, beta = alfa0 + total, beta0 + n_ore
post_l = stats.gamma(alfa, scale=1 / beta)   # scipy foloseste SCARA = 1 / rata

print()
print("Gamma-Poisson: posterior Gamma(", alfa, ",", beta, ")")
print("  MAP  =", round((alfa - 1) / beta, 4))           # modul: (alfa-1)/beta
print("  medie=", round(alfa / beta, 4))                 # alfa / beta

# HDI: cel mai SCURT interval cu masa ceruta (diferit de intervalul central)
esantion = post_l.rvs(100000, random_state=1)
print("  HDI 94% =", np.round(az.hdi(esantion, prob=0.94), 4))   # arviz<1: hdi_prob=
print("  central 94% =", np.round(post_l.ppf([0.03, 0.97]), 4))

# ============ grafic prior vs posterior =====================================
t = np.linspace(0, 1, 500)
plt.plot(t, prior.pdf(t), label="prior")
plt.plot(t, post.pdf(t), label="posterior")
plt.xlabel("theta")
plt.legend()
plt.savefig("prior_posterior.png", dpi=110, bbox_inches="tight")
plt.show()

# ============ echivalente de API in ArviZ ===================================
# arviz < 1.0                       arviz >= 1.0
# az.plot_kde(x)                    grila, dens, _ = az.kde(x); plt.plot(grila, dens)
# az.plot_posterior(idata)          az.plot_dist(idata, ci_kind="hdi", ci_prob=0.94)
# az.hdi(x, hdi_prob=0.94)          az.hdi(x, prob=0.94)
# az.summary(idata, hdi_prob=0.95)  az.summary(idata, ci_prob=0.95, ci_kind="hdi")
