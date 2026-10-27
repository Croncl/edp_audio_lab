"""
benchmark_asr.py — O realce de áudio melhora a transcrição do faster-whisper?

Para cada fala do manifesto (por padrão, LapsBM/test) e para cada condição de ruído
(limpo, e ruído branco a SNR 20/10/5/0 dB), aplica cada "frente" de realce e transcreve.
Mede o WER (%) com normalização de texto e compara cada frente contra "nenhum" com
intervalo de confiança de 95% por bootstrap (pareado: mesmas falas, mesmo ruído).

    python benchmark_asr.py                              # tiny, 3 falas por locutor (~105 falas)
    python benchmark_asr.py --modelo base --por-locutor 20
    python benchmark_asr.py --frentes nenhum wiener --snrs 10 0

Atenção: o LapsBM foi gravado em ambiente não controlado, então o "limpo" já tem ruído real.
Saídas em resultados_asr/: falas_<modelo>.csv (uma linha por fala×condição×frente),
resumo_<modelo>.csv e wer_<modelo>.png.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import edp_lab as L
from textutil import normalizar_texto

FRENTES = ["nenhum", "perona_malik", "wiener", "wiener+pm", "pm_seletivo", "wiener+pm_sel"]
KAPPA, DT, N_ITER, N_ITER_SEL = 10.0, 0.2, 3, 10
CFG = {"tipo": "branco", "pool": []}   # pool: (locutor, audio) p/ babble


# ----------------------------------------------------------------------
def aplicar_frente(y, frente):
    """Aplica uma técnica de realce e devolve o áudio com o MESMO volume (RMS) da entrada."""
    if frente == "nenhum":
        return y
    if frente == "perona_malik":
        r = L.aplicar_edp(y, "perona_malik", kappa=KAPPA, dt=DT, n_iter=N_ITER)
    elif frente == "wiener":
        r = L.aplicar_edp(y, None, pre="wiener")
    elif frente == "wiener+pm":
        r = L.aplicar_edp(y, "perona_malik", pre="wiener", kappa=KAPPA, dt=DT, n_iter=N_ITER)
    elif frente == "pm_seletivo":
        r = L.aplicar_edp(y, "perona_malik", seletivo=True, kappa=KAPPA, dt=DT, n_iter=N_ITER_SEL)
    elif frente == "wiener+pm_sel":
        r = L.aplicar_edp(y, "perona_malik", pre="wiener", seletivo=True, kappa=KAPPA, dt=DT, n_iter=N_ITER_SEL)
    else:
        raise ValueError(f"frente desconhecida: {frente}")
    return L.igualar_nivel(y, r["y_out"])


def dist_edicao(a, b):
    """Distância de Levenshtein entre duas listas (palavras)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, yv in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != yv)))
        prev = cur
    return prev[-1]


def escolher_falas(manifesto, split, por_locutor, seed=0):
    df = pd.read_csv(manifesto)
    df = df[df.split == split].copy()
    df = df[df.texto_norm.fillna("").str.strip() != ""]
    rng = np.random.default_rng(seed)
    partes = []
    for _, g in df.groupby("speaker"):
        idx = rng.permutation(len(g))[:por_locutor]
        partes.append(g.iloc[np.sort(idx)])
    return pd.concat(partes).reset_index(drop=True)


def rodar(asr, falas, frentes, snrs, log=print):
    """asr: função(audio float32 16 kHz) -> texto. Retorna DataFrame (falas × condição × frente)."""
    condicoes = ["limpo"] + [f"snr{s}" for s in snrs]
    linhas, t_asr = [], 0.0
    t0 = time.perf_counter()
    for ci, cond in enumerate(condicoes):
        for i, f in falas.iterrows():
            y, sr = sf.read(f.wav, dtype="float32")
            y = y - y.mean()   # remove DC
            assert sr == L.SR, f"{f.wav}: esperado {L.SR} Hz, veio {sr}"
            if cond != "limpo":
                y = L.adicionar_ruido_tipo(y, float(cond[3:]), CFG["tipo"], seed=1000 + i, pool=[p for sp, p in CFG["pool"] if sp != f.speaker])   # mesmo ruído p/ todas as frentes
            for fr in frentes:
                ye = aplicar_frente(y, fr)
                t1 = time.perf_counter()
                hip = normalizar_texto(asr(ye))
                t_asr += time.perf_counter() - t1
                ref = f.texto_norm.split()
                linhas.append(dict(id=f.id, speaker=f.speaker, condicao=cond, frente=fr, ref=f.texto_norm, hip=hip,
                                   erros=dist_edicao(ref, hip.split()), palavras=len(ref), dur_s=f.dur_s))
            if (i + 1) % 20 == 0 or i + 1 == len(falas):
                feito = ci * len(falas) + i + 1
                total = len(condicoes) * len(falas)
                dec = time.perf_counter() - t0
                log(f"  [{cond}] fala {i + 1}/{len(falas)} | progresso {100 * feito / total:.0f}% | "
                    f"decorrido {dec / 60:.1f} min | restante ~{dec * (total - feito) / feito / 60:.1f} min")
    df = pd.DataFrame(linhas)
    df.attrs["tempo_asr_s"] = t_asr
    return df


def resumir(df, frentes, boot=2000, seed=0):
    """WER por condição×frente e Δ vs 'nenhum' com IC95% bootstrap (pareado)."""
    rng = np.random.default_rng(seed)
    condicoes = list(dict.fromkeys(df.condicao))
    linhas = []
    for cond in condicoes:
        d = df[df.condicao == cond]
        base = d[d.frente == "nenhum"].sort_values("id") if "nenhum" in frentes else None
        n = d[d.frente == frentes[0]].shape[0]
        idx = rng.integers(0, n, size=(boot, n))
        for fr in frentes:
            g = d[d.frente == fr].sort_values("id")
            e, p = g.erros.to_numpy(), g.palavras.to_numpy()
            wer = 100 * e.sum() / p.sum()
            lo, hi = np.percentile(100 * e[idx].sum(1) / p[idx].sum(1), [2.5, 97.5])
            linha = dict(condicao=cond, frente=fr, falas=len(g), WER=wer, WER_ic_lo=lo, WER_ic_hi=hi)
            if base is not None and fr != "nenhum":
                eb = base.erros.to_numpy()
                delta = 100 * (e - eb).sum() / p.sum()
                dl, dh = np.percentile(100 * (e[idx] - eb[idx]).sum(1) / p[idx].sum(1), [2.5, 97.5])
                linha.update(delta_vs_nenhum=delta, delta_lo=dl, delta_hi=dh,
                             efeito="MELHORA" if dh < 0 else ("PIORA" if dl > 0 else "sem diferença clara"))
            linhas.append(linha)
    return pd.DataFrame(linhas)


def grafico(resumo, saida, modelo):
    import matplotlib.pyplot as plt
    cond = list(dict.fromkeys(resumo.condicao))
    rot = ["limpo" if c == "limpo" else f"{c[3:]} dB" for c in cond]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fr in dict.fromkeys(resumo.frente):
        d = resumo[resumo.frente == fr].set_index("condicao").loc[cond]
        cor = L.CORES_VAR.get("ruidoso" if fr == "nenhum" else fr, "#52514e")
        ax.errorbar(range(len(cond)), d.WER, yerr=[d.WER - d.WER_ic_lo, d.WER_ic_hi - d.WER], color=cor, lw=2,
                    marker="o", ms=5, capsize=3, label=L.ROTULOS_VAR.get(fr, "Sem realce" if fr == "nenhum" else fr))
    ax.set_xticks(range(len(cond)), rot)
    ax.set_xlabel("Condição (SNR do ruído branco adicionado)")
    ax.set_ylabel("WER (%)  ↓")
    ax.set_title(f"Whisper {modelo}: WER por condição e técnica de realce (barras = IC 95%)")
    ax.legend()
    fig.savefig(saida, bbox_inches="tight")


# ----------------------------------------------------------------------
def carregar_modelo(nome, device, threads):
    """auto: usa CUDA se existir e funcionar, sen?o CPU. Retorna (modelo, dispositivo_usado)."""
    import numpy as np
    from faster_whisper import WhisperModel
    if device in ("auto", "cuda"):
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                m = WhisperModel(nome, device="cuda", compute_type="float16")
                list(m.transcribe(np.zeros(16000, dtype="float32"), language="pt")[0])  # teste real
                return m, "cuda"
            motivo = "nenhuma GPU CUDA encontrada"
        except Exception as e:
            motivo = f"{type(e).__name__}: {e}"
        print(f"CUDA indispon?vel ({motivo}); usando CPU.")
    return WhisperModel(nome, device="cpu", compute_type="int8", cpu_threads=threads), "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifesto", default="data/manifests/lapsbm.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--modelo", default="tiny", help="tiny | base | small | medium")
    ap.add_argument("--por-locutor", type=int, default=3)
    ap.add_argument("--snrs", type=float, nargs="+", default=[20, 10, 5, 0])
    ap.add_argument("--frentes", nargs="+", default=FRENTES, choices=FRENTES)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--threads", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--ruido", default="branco", choices=["branco", "rosa", "marrom", "babble"])
    ap.add_argument("--saida", default="resultados_asr")
    a = ap.parse_args()
    if "nenhum" not in a.frentes:
        a.frentes = ["nenhum"] + a.frentes

    modelo, disp = carregar_modelo(a.modelo, a.device, a.threads)
    print(f"Dispositivo: {disp}")

    def asr(audio):
        segs, _ = modelo.transcribe(audio, language="pt", beam_size=a.beam, condition_on_previous_text=False,
                                    without_timestamps=True)
        return " ".join(s.text.strip() for s in segs)

    falas = escolher_falas(a.manifesto, a.split, a.por_locutor)
    CFG["tipo"] = a.ruido
    if a.ruido == "babble":
        CFG["pool"] = [(r.speaker, (lambda x: x - x.mean())(sf.read(r.wav, dtype="float32")[0])) for r in falas.itertuples()]
    print(f"{len(falas)} falas ({falas.speaker.nunique()} locutores) | modelo {a.modelo} | frentes {a.frentes} | SNRs {a.snrs}")
    df = rodar(asr, falas, a.frentes, [int(s) if float(s).is_integer() else s for s in a.snrs])

    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"falas_{a.modelo}.csv", index=False)
    resumo = resumir(df, a.frentes)
    resumo.to_csv(out / f"resumo_{a.modelo}.csv", index=False)
    grafico(resumo, out / f"wer_{a.modelo}.png", a.modelo)

    dur = falas.dur_s.sum() * len(a.frentes) * (1 + len(a.snrs))
    print(f"\nTempo de reconhecimento: {df.attrs['tempo_asr_s'] / 60:.1f} min para {dur / 60:.0f} min de áudio "
          f"(fator {df.attrs['tempo_asr_s'] / dur:.2f}x)")
    print("\n=== WER (%) — linhas: técnica | colunas: condição ===")
    print(resumo.pivot(index="frente", columns="condicao", values="WER").reindex(a.frentes)[list(dict.fromkeys(resumo.condicao))].round(1).to_string())
    print("\n=== Variação do WER vs. 'nenhum' (pontos percentuais; IC 95%) ===")
    d = resumo[resumo.frente != "nenhum"].copy()
    d["delta"] = d.apply(lambda r: f"{r.delta_vs_nenhum:+.1f} [{r.delta_lo:+.1f}, {r.delta_hi:+.1f}] {r.efeito}", axis=1)
    print(d.pivot(index="frente", columns="condicao", values="delta").reindex(a.frentes[1:])[list(dict.fromkeys(resumo.condicao))].to_string())
    print(f"\nSalvo em {out.resolve()}")


if __name__ == "__main__":
    main()
