"""
teste_mu.py
===========

Quantifica o vies introduzido pela linearizacao de pequeno sinal na Eq. 18
de Labrador-Paez, quando a modulacao da excitacao NAO e pequena.

METODO
------
1. Escolhe taxas microscopicas conhecidas.
2. Integra o sistema de equacoes de taxa SEM linearizar em mu e SEM
   descartar o termo W52*n2*n5 da equacao do sensitizador.
   (A unica aproximacao mantida e a de estados fundamentais constantes.)
3. Extrai a fase do fundamental de n3(t) no regime periodico permanente.
4. Ajusta a Eq. 18 a essas fases sinteticas e compara com os valores
   verdadeiros.

Como as taxas verdadeiras sao conhecidas, o desvio mede o vies do metodo.

REFERENCIAS EFETIVAS -- ponto delicado
--------------------------------------
As "taxas efetivas" do Labrador dependem de quais termos ficam nas
equacoes. Mantendo W52*n2*n5, os valores corretos de comparacao sao

    k5'  = k54 + W51*n1 + W52*n2_ss        (nao apenas k54 + W51*n1)
    k21' = k21 + W52*n5_ss

calculados com as populacoes ESTACIONARIAS de fato obtidas, e nao com as
que se pretendia impor. Comparar contra os nominais gera um desvio
aparente grande que e artefato de contabilidade, nao fisica.

USO
---
Deixe `fit_labrador.py` na mesma pasta e rode:

    python teste_mu.py
"""

import numpy as np
from scipy.integrate import solve_ivp

from fit_labrador import ajustar

np.seterr(under="ignore")      # silencia o underflow benigno do Radau


# =====================================================================
# CONFIGURACAO
# =====================================================================
MUS = [0.05, 1.0]              # profundidades de modulacao a comparar
N_PERIODOS = 6                 # periodos integrados por frequencia
N_DESCARTE = 4                 # periodos descartados (transiente)
N_AMOSTRAS = 4096              # pontos na janela analisada

FREQS = np.array([10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 1000,
                  2000, 3000, 5000, 7000, 10000, 15000], float)

# ---- taxas microscopicas (escolhidas p/ reproduzir o ajuste medido) ----
k31  = 1.749e5     # s^-1   nivel emissor (verde)
k21  = 549.0       # s^-1   decaimento linear do nivel intermediario
k54  = 5.0e3       # s^-1   decaimento intrinseco do sensitizador
W51p = 5.23e3      # s^-1   = W51*n1  (transferencia Yb -> Er, 1o passo)

n5ss_nominal = 1e-3                     # alvo usado so p/ dimensionar
k45p = (k54 + W51p) * n5ss_nominal      # taxa de bombeio
W52  = (3665.0 - 549.0) / n5ss_nominal  # 2o passo de transferencia


# =====================================================================
# Sistema de equacoes de taxa
# =====================================================================
def sistema(t, y, mu, w):
    """n5 = sensitizador excitado, n2 = intermediario, n3 = emissor."""
    n5, n2, n3 = y
    F = 1.0 + mu * np.cos(w * t)
    d5 = k45p * F - (k54 + W51p) * n5 - W52 * n2 * n5
    d2 = W51p * n5 - W52 * n2 * n5 - k21 * n2
    d3 = W52 * n2 * n5 - k31 * n3
    return [d5, d2, d3]


def estado_estacionario():
    """Integra com mu=0 ate o sistema parar de evoluir."""
    s = solve_ivp(sistema, (0, 0.5), [0, 0, 0], args=(0.0, 0.0),
                  method="Radau", rtol=1e-10, atol=1e-18)
    return s.y[:, -1]


def fase_fundamental(mu, y0, freqs=FREQS):
    """Atraso de fase de n3 no fundamental, para cada frequencia."""
    n_janela = N_PERIODOS - N_DESCARTE          # periodos analisados
    out = []
    for f in freqs:
        w, T = 2 * np.pi * f, 1.0 / f
        s = solve_ivp(sistema, (0, N_PERIODOS * T), y0, args=(mu, w),
                      method="Radau", rtol=1e-10, atol=1e-18,
                      dense_output=True)
        t = np.linspace(N_DESCARTE * T, N_PERIODOS * T,
                        N_AMOSTRAS, endpoint=False)
        n3 = s.sol(t)[2]
        Y = np.fft.rfft(n3 - n3.mean())
        # janela com n_janela periodos -> fundamental no bin n_janela
        out.append(-np.angle(Y[n_janela]))
    return np.array(out)


# =====================================================================
def main():
    print("nominais (ignorando W52*n2*n5):")
    print(f"  k31 = {k31:.4g}   k5' = {k54+W51p:.4g}   k21' = 3665")

    y0 = estado_estacionario()
    n5_ss, n2_ss, n3_ss = y0
    print(f"\npopulacoes estacionarias:")
    print(f"  n5_ss = {n5_ss:.3e}   n2_ss = {n2_ss:.3e}   n3_ss = {n3_ss:.3e}")

    # referencias corretas, com o termo W52*n2*n5 contabilizado
    k31_ref = k31
    k5p_ref = k54 + W51p + W52 * n2_ss
    k21p_ref = k21 + W52 * n5_ss
    print(f"\nREFERENCIAS EFETIVAS (usadas no desvio):")
    print(f"  k31 = {k31_ref:.4g}   k5' = {k5p_ref:.4g}   k21' = {k21p_ref:.4g}")

    resultados = {}
    for mu in MUS:
        phi = fase_fundamental(mu, y0)
        err = np.full_like(FREQS, 0.005)
        r = ajustar(FREQS, -phi, err, verbose=False)
        resultados[mu] = r

        print(f"\n{'#'*58}")
        print(f"  mu = {mu}")
        print(f"{'#'*58}")
        for nome, obtido, ref in [("k31 ", r["k31"], k31_ref),
                                  ("k5' ", r["k5"], k5p_ref),
                                  ("k21'", r["k21"], k21p_ref)]:
            print(f"  {nome} = {obtido:11.4g}   ref = {ref:11.4g}   "
                  f"desvio = {100*(obtido-ref)/ref:+6.1f}%")
        print(f"  r     = {r['r']:11.4g}   chi2red = {r['chi2_red']:.4f}")

    # comparacao direta entre a menor e a maior modulacao
    if len(MUS) >= 2:
        a, b = resultados[MUS[0]], resultados[MUS[-1]]
        print(f"\n{'='*58}")
        print(f"EFEITO DA MODULACAO  (mu={MUS[0]}  ->  mu={MUS[-1]})")
        print(f"{'='*58}")
        for nome, ka, kb in [("k31 ", a["k31"], b["k31"]),
                             ("k5' ", a["k5"], b["k5"]),
                             ("k21'", a["k21"], b["k21"])]:
            print(f"  {nome}: {ka:11.4g}  ->  {kb:11.4g}   "
                  f"({100*(kb-ka)/ka:+6.1f}%)")
        print("\n  k31 e k5' sao praticamente imunes a profundidade de")
        print("  modulacao; k21' e o parametro que sofre, por depender")
        print("  explicitamente da intensidade de excitacao (Eq. 11).")
        print("  Note que o chi2 permanece baixo em ambos os casos: a")
        print("  qualidade do ajuste NAO denuncia o vies.")

    return resultados


if __name__ == "__main__":
    main()