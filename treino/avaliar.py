"""
avaliar.py: compara os métodos no LapsBM (locutores que a U-Net nunca viu).

    python treino/avaliar.py --modelo modelos/unet_v1/best.pt --mlflow

Métodos: ruidoso, wiener, wiener+pm_sel, unet, unet+pm_sel.
Métricas: SNR, LSD, STOI (média entre falas) + diferença pareada com IC 95% (bootstrap).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pystoi import stoi
from scipy.signal import stft as sstft

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "treino"))
import edp_lab as L  # noqa: E402
from modelo import carregar, realcar  # noqa: E402
from registro import Registro  # noqa: E402

SR = 16000
PARES = ["unet", "unet+pm_sel", "wiener+pm_sel", "wiener"]


def snr_db(ref, y):
    return 10 * np.log10(np.sum(ref ** 2) / (np.sum((ref - y) ** 2) + 1e-12) + 1e-12)


def lsd_db(ref, y):
    _, _, A = sstft(ref, SR, nperseg=512, noverlap=384)
    _, _, B = sstft(y, SR, nperseg=512, noverlap=384)
    da, db = 20 * np.log10(np.abs(A) + 1e-5), 20 * np.log10(np.abs(B) + 1e-5)
    return float(np.mean(np.sqrt(np.mean((da - db) ** 2, axis=0))))


def metricas(ref, y):
    n = min(len(ref), len(y))
    ref, y = ref[:n].astype(np.float64), y[:n].astype(np.float64)
    return dict(SNR=snr_db(ref, y), LSD=lsd_db(ref, y), STOI=float(stoi(ref, y, SR, extended=False)))


def pm_sel(y, pre=None, n_iter=10):
    return L.aplicar_edp(y, "perona_malik", pre=pre, seletivo=True, margem=8.0, kappa=10.0, dt=0.2,
                         n_iter=n_iter)["y_out"]


def aplicar_todos(yr, modelo, dev):
    out = {"ruidoso": yr}
    try:
        out["wiener"] = L.aplicar_edp(yr, "perona_malik", pre="wiener", n_iter=0)["y_out"]
    except Exception as e:                      # n_iter=0 pode não ser aceito
        print("Aviso: 'wiener' sozinho indisponível:", repr(e))
    out["wiener+pm_sel"] = pm_sel(yr, pre="wiener")
    out["unet"] = realcar(modelo, yr, dev)
    out["unet+pm_sel"] = pm_sel(out["unet"])
    return out


def boot_ic(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="modelos/unet_v1/best.pt")
    ap.add_argument("--manifesto", default="data/manifests/lapsbm.csv")
    ap.add_argument("--ruidos", nargs="+", default=["branco", "babble"])
    ap.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 20])
    ap.add_argument("--duracao", type=float, default=4.0)
    ap.add_argument("--n", type=int, default=35, help="1 fala por locutor, ate n locutores")
    ap.add_argument("--saida", default="resultados_unet")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--experimento", default="avaliacao")
    ap.add_argument("--nome", default="avaliacao_unet_v1")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    modelo, ck = carregar(a.modelo, dev)
    print(f"Dispositivo: {dev} | modelo da época {ck['epoca']} (val_loss {ck['val_loss']:.4f})")

    df = pd.read_csv(a.manifesto).sort_values("id").groupby("speaker").head(1).head(a.n)
    audios = []
    for r in df.itertuples():
        y, _ = sf.read(r.wav, dtype="float32")
        y = y[: int(a.duracao * SR)]
        y = y - y.mean()          # o LapsBM tem DC grande; o treino também usa sinal sem DC
        audios.append((r.speaker, y * (0.9 / (np.max(np.abs(y)) + 1e-9))))
    print(f"{len(audios)} falas (1 por locutor) | ruídos {a.ruidos} | SNRs {a.snrs}")

    linhas = []
    for tipo in a.ruidos:
        for snr in a.snrs:
            for i, (sp, y) in enumerate(audios):
                pool = [p for s2, p in audios if s2 != sp]
                yr = L.adicionar_ruido_tipo(y, snr, tipo, seed=1000 + i, pool=pool)
                for nome, yo in aplicar_todos(yr, modelo, dev).items():
                    linhas.append(dict(ruido=tipo, snr_entrada=snr, fala=i, metodo=nome, **metricas(y, yo)))
            print(f"  {tipo} {snr:g} dB feito", flush=True)

    d = pd.DataFrame(linhas)
    pasta = Path(a.saida)
    pasta.mkdir(parents=True, exist_ok=True)
    d.to_csv(pasta / "por_fala.csv", index=False)
    res = d.groupby(["ruido", "snr_entrada", "metodo"])[["SNR", "LSD", "STOI"]].mean().round(3)
    res.to_csv(pasta / "resumo.csv")
    print("\n=== MÉDIA entre falas ===")
    print(res.to_string())

    # diferenças pareadas
    print("\n=== Diferença pareada (IC 95%) ===")
    dif = []
    for (tipo, snr), g in d.groupby(["ruido", "snr_entrada"]):
        p = g.pivot(index="fala", columns="metodo", values=["SNR", "STOI"])
        for a_, b_ in [("unet", "wiener+pm_sel"), ("unet+pm_sel", "unet"), ("unet", "ruidoso")]:
            if a_ in p["SNR"] and b_ in p["SNR"]:
                for m in ("SNR", "STOI"):
                    x = (p[m][a_] - p[m][b_]).to_numpy()
                    lo, hi = boot_ic(x)
                    dif.append(dict(ruido=tipo, snr_entrada=snr, comparacao=f"{a_} - {b_}", metrica=m,
                                    media=round(x.mean(), 3), ic_inf=round(lo, 3), ic_sup=round(hi, 3),
                                    veredito="MELHORA" if lo > 0 else "PIORA" if hi < 0 else "sem diferença clara"))
    dd = pd.DataFrame(dif)
    dd.to_csv(pasta / "diferencas.csv", index=False)
    print(dd.to_string(index=False))

    # gráfico
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ms = ["SNR", "LSD", "STOI"]
    fig, ax = plt.subplots(len(a.ruidos), 3, figsize=(13, 3.6 * len(a.ruidos)), squeeze=False)
    for i, tipo in enumerate(a.ruidos):
        for j, m in enumerate(ms):
            for nome, g in d[d.ruido == tipo].groupby("metodo"):
                s = g.groupby("snr_entrada")[m].mean()
                ax[i, j].plot(s.index, s.values, marker="o", label=nome, ls="--" if nome == "ruidoso" else "-")
            ax[i, j].set_title(f"{m} ({tipo})")
            ax[i, j].set_xlabel("SNR de entrada (dB)")
            ax[i, j].grid(alpha=.3)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pasta / "comparacao.png", dpi=130)

    reg = Registro(a.mlflow, a.experimento, a.nome, dict(modelo=a.modelo, epoca=ck["epoca"], ruidos=a.ruidos,
                                                         snrs=a.snrs, n_falas=len(audios), dispositivo=dev))
    for (tipo, snr, met), r in res.iterrows():
        for m in ms:
            reg.metricas({f"{tipo}_snr{snr:g}_{met.replace('+', '_')}_{m}": r[m]})
    for f in ("resumo.csv", "diferencas.csv", "por_fala.csv", "comparacao.png"):
        reg.arquivo(pasta / f)
    reg.fim()
    print(f"\nSalvo em {pasta.resolve()}")


if __name__ == "__main__":
    main()
