"""
fit_labrador.py
===============

Ajusta a Eq. 18 de Labrador-Paez et al. (J. Phys. Chem. Lett. 2023, 14, 3436)
aos dados de diferenca de fase medidos no dominio da frequencia.

    Phi(w) = arctan(a31) + arctan(a5) + arctan( r*a21 / (1 + r + a21^2) )

    a31 = w/k31    k31  : taxa do nivel emissor (verde, 4S3/2)
    a5  = w/k5     k5'  : taxa efetiva do sensitizador (Yb 2F5/2)
    a21 = w/k21    k21' : taxa efetiva do nivel intermediario (4I11/2)
    r   = k21/k21' (0 < r <= 1)

USO
---
Coloque este arquivo na mesma pasta de `treament_tcspc_data.py` e rode:

    python fit_labrador.py

ou, dentro do Jupyter:

    %run fit_labrador.py

ou ainda, para usar funcao a funcao:

    from fit_labrador import carregar, ajustar, plotar
    freqs, pd, err = carregar(DATA_DIR, particle_id=1)
    res = ajustar(freqs, pd, err)
    plotar(freqs, pd, err, res)

OBSERVACOES IMPORTANTES
-----------------------
1. k31 e k5' entram de forma SIMETRICA nos dois primeiros arctan. O ajuste
   nao consegue distinguir qual e qual -- ele devolve o par. A atribuicao
   depende de conhecimento externo (p.ex. medir o decaimento do Yb a 1030 nm,
   ou o decaimento direto do verde no dominio do tempo).
   Por convencao, este script reporta o MAIOR como k31 e o MENOR como k5',
   que e o caso do Labrador (k31=18048, k5'=6000 s^-1).

2. r costuma ser mal determinado se o termo dele for pequeno. Use
   `r_fixo=1.0` (App II) se o ajuste nao convergir bem.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.fft import fft, fftfreq

from treament_tcspc_data import (
    fetch_data_filenames,
    extract_data_info_from_path,
    get_mean_amp_and_phase,
)

# =====================================================================
# CONFIGURACAO -- edite apenas esta secao
# =====================================================================
DATA_DIR    = "./30particles/"
PARTICLE_ID = 1          # p{PARTICLE_ID}_lum
F_MAX       = 1.5e4      # Hz -- descarta pontos acima disso (SNR ruim)
F_MIN       = 0.0        # Hz
SNR_MIN_DB  = 3.0        # descarta pontos com SNR abaixo disso (None desliga)
R_FIXO      = None       # None -> ajusta r ; 1.0 -> App II do Labrador
CORRIGIR_DT = True       # usa sample_interval = (max-min)/(len-1)
SIGMA_SIST  = 0.02        # rad -- piso de erro sistematico (~1 grau)


# =====================================================================
# FFT com o passo temporal corrigido
# =====================================================================
def _get_fft(time, curve):
    """FFT com sample_interval correto (time em microssegundos)."""
    if CORRIGIR_DT:
        dt = (np.max(time) - np.min(time)) / (len(time) - 1) / 1e6
    else:
        dt = np.max(time) / len(curve) / 1e6
    return fftfreq(len(curve), dt), fft(curve)


def _fasor(time, curve, f0):
    """Fasor complexo no bin mais proximo de f0."""
    xf, yf = _get_fft(time, curve)
    m = xf > 0
    xf, yf = xf[m], yf[m]
    i = np.argmin(np.abs(xf - f0))
    return yf[i]


def _snr_db(time, curve, f0):
    """SNR em dB: potencia no fundamental contra a media do resto (sem DC/2w)."""
    xf, yf = _get_fft(time, curve)
    m = xf >= 0
    xf, yf = xf[m], yf[m]
    p = np.abs(yf) ** 2
    i_dc = np.argmin(np.abs(xf))
    i_f = np.argmin(np.abs(xf - f0))
    i_2f = np.argmin(np.abs(xf - 2 * f0))
    p_sig = p[i_f]
    p_res = p.copy()
    for i in (i_dc, i_f, i_2f):
        p_res[i] = 0.0
    n_eff = max(len(p) - 3, 1)
    p_noise = p_res.sum() / n_eff
    if p_noise <= 1e-15:
        return np.inf
    return 10 * np.log10(p_sig / p_noise)


# =====================================================================
# Carregamento
# =====================================================================
def _steps(filenames, tag):
    """Extrai e ordena por frequencia os arquivos .npy que contem `tag`."""
    steps = [
        extract_data_info_from_path(p)
        for p in filenames
        if (tag in p) and ("step" in p) and (".npy" in p)
    ]
    steps = [s for s in steps if s is not None]
    return sorted(steps, key=lambda s: s["freq"])


def carregar(data_dir=DATA_DIR, particle_id=PARTICLE_ID, verbose=True):
    """Devolve (freqs, phase_diff, erro, snr) ja alinhados por frequencia."""
    filenames = fetch_data_filenames(data_dir)
    exc_steps = _steps(filenames, "exc_laser")
    lum_steps = _steps(filenames, f"p{particle_id}_lum")

    exc_por_freq = {s["freq"]: s for s in exc_steps}
    freqs_comuns = sorted(f for f in (s["freq"] for s in lum_steps) if f in exc_por_freq)

    if verbose:
        print(f"excitacao : {len(exc_steps)} passos")
        print(f"particula : {len(lum_steps)} passos")
        print(f"em comum  : {len(freqs_comuns)} frequencias")
        faltando = sorted({s['freq'] for s in lum_steps} - set(freqs_comuns))
        if faltando:
            print(f"  AVISO -- sem referencia de excitacao: {faltando}")

    freqs, pd_raw, pd_err, snr = [], [], [], []
    for f0 in freqs_comuns:
        lum = next(s for s in lum_steps if s["freq"] == f0)
        exc = exc_por_freq[f0]

        fas_l = [_fasor(m[:, 0], m[:, 1], f0) for m in lum["data"]]
        fas_e = [_fasor(m[:, 0], m[:, 1], f0) for m in exc["data"]]
        if not fas_l or not fas_e:
            continue

        _, _, ph_l, er_l = get_mean_amp_and_phase(fas_l)
        _, _, ph_e, er_e = get_mean_amp_and_phase(fas_e)

        freqs.append(f0)
        pd_raw.append(ph_l - ph_e)
        pd_err.append(np.hypot(er_l, er_e))
        snr.append(np.mean([_snr_db(m[:, 0], m[:, 1], f0) for m in lum["data"]]))

    freqs = np.array(freqs, dtype=float)
    pd_raw = np.unwrap(np.array(pd_raw))
    pd_err = np.array(pd_err)
    snr = np.array(snr)

    # erro nulo (uma unica repeticao) quebra o peso do ajuste
    pd_err[pd_err <= 0] = np.nanmedian(pd_err[pd_err > 0]) if np.any(pd_err > 0) else 0.01

    return freqs, pd_raw, pd_err, snr


def filtrar(freqs, pd, err, snr, f_min=F_MIN, f_max=F_MAX, snr_min=SNR_MIN_DB,
            verbose=True):
    """Aplica os cortes de frequencia e SNR."""
    m = (freqs >= f_min) & (freqs <= f_max)
    if snr_min is not None:
        m &= (snr >= snr_min)
    if verbose:
        print(f"\nmantidos {m.sum()} de {len(freqs)} pontos")
        for f0, s, ok in zip(freqs, snr, m):
            if not ok:
                print(f"  descartado {f0:>9.1f} Hz  (SNR {s:5.1f} dB)")
    return freqs[m], pd[m], err[m], snr[m]


# =====================================================================
# Modelo e ajuste
# =====================================================================
def phi_labrador(f, k31, k5, k21, r):
    """Eq. 18 do Labrador. Devolve o atraso de fase (positivo), em rad."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    a31, a5, a21 = w / k31, w / k5, w / k21
    return (np.arctan(a31) + np.arctan(a5)
            + np.arctan(r * a21 / (1.0 + r + a21 ** 2)))


def ajustar(freqs, pd, err, r_fixo=R_FIXO, p0=None, verbose=True):
    """Ajusta a Eq. 18. `pd` e a diferenca de fase medida (negativa)."""
    phi_obs = -np.asarray(pd)          # converte para atraso positivo
    w_max = 2 * np.pi * freqs.max()

    if p0 is None:
        p0 = [w_max, w_max / 5.0, w_max / 20.0]   # k31, k5, k21

    ajusta_r = r_fixo is None

    def residuo(p):
        k31, k5, k21 = np.exp(p[:3])               # log garante positividade
        r = 1.0 / (1.0 + np.exp(-p[3])) if ajusta_r else r_fixo
        return (phi_labrador(freqs, k31, k5, k21, r) - phi_obs) / err

    x0 = list(np.log(p0)) + ([0.0] if ajusta_r else [])
    sol = least_squares(residuo, x0, method="lm", max_nfev=20000)

    k = np.exp(sol.x[:3])
    r = 1.0 / (1.0 + np.exp(-sol.x[3])) if ajusta_r else r_fixo

    # incertezas: covariancia = (J^T J)^-1 * s^2
    dof = max(len(freqs) - len(sol.x), 1)
    s2 = 2 * sol.cost / dof
    try:
        cov = np.linalg.inv(sol.jac.T @ sol.jac) * s2
        sig_log = np.sqrt(np.abs(np.diag(cov))[:3])
        sig_k = k * sig_log                       # propaga do log
    except np.linalg.LinAlgError:
        sig_k = np.full(3, np.nan)

    # convencao: maior das duas primeiras taxas = k31
    ordem = np.argsort([k[0], k[1]])[::-1]
    k31, k5 = k[ordem]
    s31, s5 = sig_k[ordem]
    k21, s21 = k[2], sig_k[2]

    res = {
        "k31": k31, "k31_err": s31,
        "k5": k5,   "k5_err": s5,
        "k21": k21, "k21_err": s21,
        "r": r, "r_ajustado": ajusta_r,
        "chi2_red": s2, "sol": sol,
    }

    if verbose:
        print("\n" + "=" * 58)
        print("AJUSTE -- Eq. 18 de Labrador-Paez")
        print("=" * 58)
        print(f"  k31  = {k31:11.4g} +/- {s31:8.3g} s^-1    "
              f"(tau = {1e6/k31:8.2f} us)")
        print(f"  k5'  = {k5:11.4g} +/- {s5:8.3g} s^-1    "
              f"(tau = {1e6/k5:8.2f} us)")
        print(f"  k21' = {k21:11.4g} +/- {s21:8.3g} s^-1    "
              f"(tau = {1e6/k21:8.2f} us)")
        print(f"  r    = {r:11.4g}" + ("  (ajustado)" if ajusta_r else "  (fixo)"))
        if ajusta_r:
            print(f"  k21  = r*k21' = {r*k21:.4g} s^-1")
        print(f"\n  chi2 reduzido = {s2:.3f}   ({len(freqs)} pontos)")
        print("\n  Lembre: k31 e k5' sao intercambiaveis no modelo.")
        print("  A atribuicao acima e por convencao (maior = k31);")
        print("  confirme com uma medida independente.")
        print("=" * 58)

    return res


# =====================================================================
# Grafico
# =====================================================================
def plotar(freqs, pd, err, res, freqs_todas=None, pd_todas=None,
           err_todas=None, salvar=None):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)

    if freqs_todas is not None:
        desc = ~np.isin(freqs_todas, freqs)
        if desc.any():
            ax1.errorbar(freqs_todas[desc], -pd_todas[desc], yerr=err_todas[desc],
                         fmt="o", ms=4, color="0.75", capsize=3,
                         label="descartados", zorder=1)

    ax1.errorbar(freqs, -pd, yerr=err, fmt="o", ms=5, capsize=4,
                 color="tab:blue", label="dados", zorder=3)

    ff = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 500)
    ax1.plot(ff, phi_labrador(ff, res["k31"], res["k5"], res["k21"], res["r"]),
             "-", color="tab:red", lw=2, label="Eq. 18 (ajuste)", zorder=2)

    for y, lab in [(np.pi / 2, "90"), (np.pi, "180")]:
        ax1.axhline(y, ls=":", color="0.6", lw=1)
        ax1.text(freqs.min(), y + 0.03, f"{lab}$^\\circ$", color="0.5", fontsize=9)

    ax1.set_xscale("log")
    ax1.set_ylabel("Atraso de fase (rad)")
    ax1.legend(frameon=False)
    ax1.set_title(f"Ajuste -- k31={res['k31']:.3g}, "
                  f"k5'={res['k5']:.3g}, k21'={res['k21']:.3g} s$^{{-1}}$")

    resid = (-pd) - phi_labrador(freqs, res["k31"], res["k5"],
                                 res["k21"], res["r"])
    ax2.errorbar(freqs, resid, yerr=err, fmt="o", ms=4, capsize=3,
                 color="tab:blue")
    ax2.axhline(0, color="tab:red", lw=1.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequencia (Hz)")
    ax2.set_ylabel("Residuo (rad)")

    if salvar:
        fig.savefig(salvar, dpi=150)
    plt.show()
    return fig


# =====================================================================
def main():
    freqs, pd, err, snr = carregar()
    fs, ps, es, _ = filtrar(freqs, pd, err, snr)

    if len(fs) < 4:
        print("\nPoucos pontos apos o filtro. Afrouxe F_MAX ou SNR_MIN_DB.")
        return

    es = np.hypot(es, SIGMA_SIST)      # piso sistemático de ~1 grau

    res = ajustar(fs, ps, es)
    plotar(fs, ps, es, res, freqs, pd, err)
    return res


if __name__ == "__main__":
    main()
