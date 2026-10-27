"""
preparar_dados.py — baixa datasets FalaBrasil do Hugging Face, padroniza o áudio (16 kHz, mono, 16 bits)
e gera um MANIFESTO (CSV) com texto, locutor, divisão e duração.

    python preparar_dados.py --dataset lapsbm coddef
    python preparar_dados.py --dataset cetuc --max-locutores 5     # só 5 locutores por divisão

Saídas:
    data/proc/<dataset>/<divisão>/<locutor>/<arquivo>.wav     áudio padronizado
    data/manifests/<dataset>.csv                              um manifesto por dataset
    data/manifests/todos.csv                                  união de todos os manifestos

Os pacotes baixados ficam no cache do Hugging Face (reaproveitados em execuções seguintes).
"""
import argparse
import os
import random
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import librosa  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

from explorar_tar import EXT_AUDIO, eh_lixo, extrair  # noqa: E402
from textutil import ler_texto, normalizar_texto  # noqa: E402

SR = 16000
DATASETS = {"lapsbm": "falabrasil/lapsbm", "coddef": "falabrasil/coddef",
            "constituicao": "falabrasil/constituicao", "cetuc": "falabrasil/cetuc"}
RAIZ = Path("data")


def divisao_e_locutor(dataset, caminho_no_repo):
    """data/<divisão>/<locutor>/<arq>.tar.gz  ->  (divisão, locutor). Pasta 'locutor' = locutor único."""
    partes = Path(caminho_no_repo).parts
    divisao = partes[1] if len(partes) > 2 and partes[0] == "data" else "desconhecida"
    pasta = partes[-2] if len(partes) >= 2 else "x"
    locutor = f"{dataset}-unico" if pasta == "locutor" else pasta
    return divisao, locutor


def processar_tar(tar_path, dataset, divisao, locutor, raiz=RAIZ):
    """Extrai um .tar.gz, converte cada áudio para 16 kHz mono e devolve as linhas do manifesto."""
    linhas, sem_texto = [], 0
    tmp = Path(tempfile.mkdtemp(prefix="extr_"))
    try:
        extrair(tar_path, tmp)
        saida_dir = raiz / "proc" / dataset / divisao / locutor
        saida_dir.mkdir(parents=True, exist_ok=True)
        audios = sorted(p for p in tmp.rglob("*") if p.suffix.lower() in EXT_AUDIO and not eh_lixo(p.name))
        for p in audios:
            txt = p.with_suffix(".txt")
            if not txt.exists():
                sem_texto += 1
                continue
            sr_orig = sf.info(str(p)).samplerate
            y, _ = librosa.load(str(p), sr=SR, mono=True)
            destino = saida_dir / f"{p.stem}.wav"
            sf.write(str(destino), y, SR, subtype="PCM_16")
            texto = ler_texto(txt)
            linhas.append(dict(
                id=f"{dataset}/{divisao}/{locutor}/{p.stem}", dataset=dataset, split=divisao, speaker=locutor,
                wav=destino.as_posix(), texto=texto, texto_norm=normalizar_texto(texto),
                dur_s=round(len(y) / SR, 3), sr_orig=sr_orig))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return linhas, sem_texto


def resumo(df):
    g = df.groupby(["dataset", "split"])
    r = g.agg(falas=("id", "count"), locutores=("speaker", "nunique"), horas=("dur_s", lambda s: s.sum() / 3600),
              dur_media_s=("dur_s", "mean"), dur_max_s=("dur_s", "max"),
              palavras=("texto_norm", lambda s: s.str.split().str.len().sum()))
    r["acima_30s"] = g["dur_s"].apply(lambda s: int((s > 30).sum()))
    return r.round(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", required=True, choices=sorted(DATASETS))
    ap.add_argument("--max-locutores", type=int, default=None, help="máx. de locutores por divisão (sorteados)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=None, help="so estas divisoes (ex.: train dev)")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    (RAIZ / "manifests").mkdir(parents=True, exist_ok=True)
    for ds in a.dataset:
        repo = DATASETS[ds]
        print(f"\n### {repo}")
        tars = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".tar.gz"))
        alvo = [(f, *divisao_e_locutor(ds, f)) for f in tars]
        if a.splits:
            alvo = [x for x in alvo if x[1] in a.splits]
        if a.max_locutores:
            rng = random.Random(a.seed)
            por_div = {}
            for item in alvo:
                por_div.setdefault(item[1], []).append(item)
            alvo = []
            for div, itens in por_div.items():
                alvo += rng.sample(itens, min(a.max_locutores, len(itens)))
        print(f"{len(alvo)} pacotes a processar")
        linhas, sem_texto = [], 0
        for i, (f, div, loc) in enumerate(sorted(alvo, key=lambda x: x[0]), 1):
            print(f"[{i}/{len(alvo)}] {f}")
            p = hf_hub_download(repo, f, repo_type="dataset")
            l, s = processar_tar(p, ds, div, loc)
            linhas += l
            sem_texto += s
        df = pd.DataFrame(linhas)
        dest = RAIZ / "manifests" / f"{ds}.csv"
        if dest.exists():
            df = pd.concat([pd.read_csv(dest), df], ignore_index=True).drop_duplicates("id", keep="last")
        df.to_csv(dest, index=False, encoding="utf-8")
        print(f"\n{ds}: {len(df)} falas salvas em data/manifests/{ds}.csv (áudios sem .txt ignorados: {sem_texto})")
        print(resumo(df).to_string())

    partes = [pd.read_csv(p) for p in sorted((RAIZ / "manifests").glob("*.csv")) if p.name != "todos.csv"]
    todos = pd.concat(partes, ignore_index=True)
    todos.to_csv(RAIZ / "manifests" / "todos.csv", index=False, encoding="utf-8")
    print("\n=== TODOS OS MANIFESTOS ===")
    print(resumo(todos).to_string())


if __name__ == "__main__":
    main()
