"""
benchmark_asr.py — mede o WER do faster-whisper em várias condições (áudio limpo / com ruído / com realce).

Exemplos:
    # linha de base: áudio limpo, sem realce
    python benchmark_asr.py --dataset lapsbm --modelo tiny --snr limpo --realce nenhum

    # ruído branco em vários níveis, sem realce, e com Wiener
    python benchmark_asr.py --dataset lapsbm --modelo tiny --snr limpo 20 10 5 --realce nenhum wiener

Condições = produto de --snr × --realce. A primeira é a referência para as comparações pareadas.
  --snr    : "limpo" (sem ruído) ou SNR em dB do ruído branco adicionado (20, 10, 5, 0...)
  --realce : nenhum | wiener | wiener+pm | pm_seletivo | calor | perona_malik   (aplicado DEPOIS do ruído)

Saídas em resultados/: resumo (WER, CER, IC 95 %, diferença pareada) e detalhe por fala.
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from textutil import normalizar_texto

SR = 16000
REALCES = ["nenhum", "wiener", "wiener+pm", "pm_seletivo", "calor", "perona_malik"]


# ----------------------------------------------------------------------
# Condições: ruído + realce
# ----------------------------------------------------------------------
def aplicar_realce(y, realce, p):
    if realce == "nenhum":
        return y
    import edp_lab as L
    if not hasattr(L, "ganho_reducao"):
        raise SystemExit("Seu edp_lab.py está desatualizado (sem Wiener / PM seletivo). "
                         "Extraia o edp_audio_lab.zip mais recente por cima da pasta.")
    if realce == "wiener":
        r = L.aplicar_edp(y, None, pre="wiener")
    elif realce == "wiener+pm":
        r = L.aplicar_edp(y, "perona_malik", pre="wiener", kappa=p["kappa"], dt=p["dt"], n_iter=p["iter"])
    elif realce == "pm_seletivo":
        r = L.aplicar_edp(y, "perona_malik", seletivo=True, kappa=p["kappa"], dt=p["dt"], n_iter=p["iter_sel"])
    elif realce == "calor":
        r = L.aplicar_edp(y, "calor", D=p["D"], n_iter=p["iter"])
    elif realce == "perona_malik":
        r = L.aplicar_edp(y, "perona_malik", kappa=p["kappa"], dt=p["dt"], n_iter=p["iter"])
    else:
        raise ValueError(realce)
    return r["y_out"]


def preparar_audio(y, snr, realce, semente, p):
    if snr != "limpo":
        import edp_lab as L
        y = L.adicionar_ruido(y, float(snr), seed=semente)
    return aplicar_realce(y, realce, p).astype(np.float32)


# ----------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------
def erros_por_fala(refs, hips):
    """(erros, nº de palavras da referência) por fala, via jiwer."""
    import jiwer
    err, n = [], []
    for r, h in zip(refs, hips):
        o = jiwer.process_words(r, h)
        err.append(o.substitutions + o.deletions + o.insertions)
        n.append(o.hits + o.substitutions + o.deletions)
    return np.array(err, float), np.array(n, float)


def wer_ic(err, n, B=2000, seed=0):
    """WER (%) e IC 95 % por bootstrap sobre as falas."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), (B, len(err)))
    w = 100 * err[idx].sum(1) / n[idx].sum(1)
    return 100 * err.sum() / n.sum(), *np.percentile(w, [2.5, 97.5])


def delta_ic(err, n, err0, n0, B=2000, seed=0):
    """Diferença de WER (condição − referência) em pontos percentuais, com IC 95 % pareado."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(err), (B, len(err)))
    d = 100 * err[idx].sum(1) / n[idx].sum(1) - 100 * err0[idx].sum(1) / n0[idx].sum(1)
    return 100 * err.sum() / n.sum() - 100 * err0.sum() / n0.sum(), *np.percentile(d, [2.5, 97.5])


# ----------------------------------------------------------------------
# Execução
# ----------------------------------------------------------------------
def criar_transcritor(modelo, threads, beam):
    from faster_whisper import WhisperModel
    m = WhisperModel(modelo, device="cpu", compute_type="int8", cpu_threads=threads)

    def transcrever(audio):
        segs, _ = m.transcribe(audio, language="pt", beam_size=beam, without_timestamps=True)
        return " ".join(s.text.strip() for s in segs)
    return transcrever


def rodar(df, transcrever, condicoes, params, log=print):
    """Roda todas as condições. Retorna (detalhe por fala, tempo por condição)."""
    audios = [sf.read(w, dtype="float32")[0] for w in df["wav"]]
    linhas, tempos = [], {}
    for snr, realce in condicoes:
        nome = f"{snr}|{realce}"
        t0 = time.perf_counter()
        for i, (y, (_, r)) in enumerate(zip(audios, df.iterrows())):
            ya = preparar_audio(y, snr, realce, semente=i, p=params)
            hip = transcrever(ya)
            linhas.append(dict(condicao=nome, id=r["id"], speaker=r["speaker"], dur_s=r["dur_s"],
                               ref=r["texto_norm"], hip_bruta=hip, hip=normalizar_texto(hip)))
            if (i + 1) % 100 == 0:
                log(f"  [{nome}] {i + 1}/{len(df)} falas")
        tempos[nome] = time.perf_counter() - t0
        log(f"{nome}: {tempos[nome]:.0f} s")
    return pd.DataFrame(linhas), tempos


def resumir(det, tempos, ref_nome=None):
    import jiwer
    conds = list(dict.fromkeys(det["condicao"]))
    ref_nome = ref_nome or conds[0]
    base = det[det.condicao == ref_nome]
    e0, n0 = erros_por_fala(base["ref"].tolist(), base["hip"].tolist())
    linhas = []
    for c in conds:
        d = det[det.condicao == c]
        e, n = erros_por_fala(d["ref"].tolist(), d["hip"].tolist())
        w, lo, hi = wer_ic(e, n)
        cer = 100 * jiwer.cer(d["ref"].tolist(), d["hip"].tolist())
        dw, dlo, dhi = delta_ic(e, n, e0, n0)
        linhas.append(dict(condicao=c, falas=len(d), WER=w, WER_ic_inf=lo, WER_ic_sup=hi, CER=cer,
                           delta_vs_ref=dw, delta_ic_inf=dlo, delta_ic_sup=dhi,
                           significativo=("sim" if (dlo > 0 or dhi < 0) else "não") if c != ref_nome else "—",
                           tempo_s=tempos.get(c, np.nan)))
    return pd.DataFrame(linhas).round(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifesto", default="data/manifests/todos.csv")
    ap.add_argument("--dataset", default="lapsbm")
    ap.add_argument("--split", default=None, help="test | dev | train (padrão: todas)")
    ap.add_argument("--modelo", default="tiny")
    ap.add_argument("--snr", nargs="+", default=["limpo"])
    ap.add_argument("--realce", nargs="+", default=["nenhum"], choices=REALCES)
    ap.add_argument("--n", type=int, default=None, help="usar só N falas (sorteadas, semente fixa)")
    ap.add_argument("--max-dur", type=float, default=30.0, help="ignora falas mais longas que isso (s)")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--threads", type=int, default=max(4, (os.cpu_count() or 8) // 2))
    ap.add_argument("--iter", type=int, default=3)
    ap.add_argument("--iter-sel", type=int, default=10)
    ap.add_argument("--D", type=float, default=0.2)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--saida", default="resultados")
    a = ap.parse_args()

    df = pd.read_csv(a.manifesto)
    df = df[df.dataset == a.dataset]
    if a.split:
        df = df[df.split == a.split]
    df = df[df.dur_s <= a.max_dur].reset_index(drop=True)
    if a.n and a.n < len(df):
        df = df.sample(a.n, random_state=0).reset_index(drop=True)
    if df.empty:
        raise SystemExit("Nenhuma fala encontrada com esses filtros. Rode antes o preparar_dados.py.")
    condicoes = [(s, r) for s in a.snr for r in a.realce]
    params = dict(iter=a.iter, iter_sel=a.iter_sel, D=a.D, kappa=a.kappa, dt=a.dt)
    print(f"{len(df)} falas de {df.speaker.nunique()} locutores | modelo: {a.modelo} | "
          f"condições: {[f'{s}|{r}' for s, r in condicoes]}")

    transcrever = criar_transcritor(a.modelo, a.threads, a.beam)
    det, tempos = rodar(df, transcrever, condicoes, params)
    res = resumir(det, tempos)

    out = Path(a.saida)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.dataset}_{a.modelo}_{time.strftime('%Y%m%d_%H%M%S')}"
    det.to_csv(out / f"{tag}_falas.csv", index=False, encoding="utf-8")
    res.to_csv(out / f"{tag}_resumo.csv", index=False, encoding="utf-8")
    print("\n=== RESUMO (WER e CER em %; delta = condição − primeira condição, em pontos percentuais) ===")
    print(res.to_string(index=False))
    print(f"\nSalvo em {out.resolve()}  ({tag}_resumo.csv e {tag}_falas.csv)")


if __name__ == "__main__":
    main()
