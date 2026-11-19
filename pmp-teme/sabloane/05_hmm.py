"""Cursul 4 - modele Markov ascunse: Forward si Viterbi. Rulabil ca atare."""

import numpy as np

stari = ["D", "M", "U"]
simboluri = ["FB", "B", "S", "NS"]

# cele trei obiecte ale unui HMM
pi = np.array([1 / 3, 1 / 3, 1 / 3])          # distributia initiala
A = np.array([[0.00, 0.50, 0.50],             # A[i][j] = P(X_{t+1}=j | X_t=i)
              [0.50, 0.25, 0.25],
              [0.50, 0.25, 0.25]])
B = np.array([[0.10, 0.20, 0.40, 0.30],       # B[j][o] = P(O_t=o | X_t=j)
              [0.15, 0.25, 0.50, 0.10],
              [0.20, 0.30, 0.40, 0.10]])

print("verificare: randuri A", A.sum(1), " randuri B", B.sum(1))

secventa = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B", "S"]
o = [simboluri.index(x) for x in secventa]
T, N = len(o), len(stari)


def forward(pi, A, B, o):
    """alpha_t(j) = P(o_1..o_t, X_t=j). Intoarce tabelul si P(o_1..o_T)."""
    T, N = len(o), len(pi)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, o[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, o[t]] * np.sum(alpha[t - 1] * A[:, j])   # SUMA
    return alpha, alpha[-1].sum()


def viterbi(pi, A, B, o):
    """delta_t(j) = max P(x_1..x_t, o_1..o_t) cu x_t=j. Intoarce traseul si scorul."""
    T, N = len(o), len(pi)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = pi * B[:, o[0]]
    for t in range(1, T):
        for j in range(N):
            candidati = delta[t - 1] * A[:, j]
            psi[t, j] = np.argmax(candidati)                             # MAXIM
            delta[t, j] = candidati[psi[t, j]] * B[j, o[t]]
    traseu = [int(np.argmax(delta[-1]))]
    for t in range(T - 1, 0, -1):               # backtracking, in ordine inversa
        traseu.append(psi[t, traseu[-1]])
    return traseu[::-1], delta[-1].max(), delta


alpha, p_obs = forward(pi, A, B, o)
traseu, scor, delta = viterbi(pi, A, B, o)

print()
print("alpha_1 =", np.round(alpha[0], 6))
print("P(o_1..o_T) =", p_obs, "  <- suma peste TOATE traseele")
print()
print("delta_1 =", np.round(delta[0], 6))
print("delta_2 =", np.round(delta[1], 6))
print("traseu optim:", [stari[s] for s in traseu])
print("P(x*, o) =", scor, "  <- un SINGUR traseu; nu se compara cu P(o)")
print("P(x* | o) =", scor / p_obs)

# --- varianta cu biblioteca ---------------------------------------------------
# from hmmlearn import hmm
# model = hmm.CategoricalHMM(n_components=3)
# model.startprob_, model.transmat_, model.emissionprob_ = pi, A, B
# logprob, path = model.decode(np.array(o).reshape(-1, 1), algorithm="viterbi")
# print([stari[s] for s in path], np.exp(logprob))
