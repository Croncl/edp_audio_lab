"""
explorar_tar.py — baixa UM pacote .tar.gz pequeno de cada dataset e mostra o que há dentro
(nomes dos arquivos, formato do áudio, exemplos de transcrição).

    python explorar_tar.py
    python explorar_tar.py --repo lapsbm --arquivo data/test/LapsBM-F004/LapsBM-F004.tar.gz

Extrai para data/amostra/<repo>/<nome-do-tar>/ (pasta nova e isolada).
Salva o resumo em exploracao_tar.txt.
"""
import argparse
import os
import tarfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import soundfile as sf  # noqa: E402

PADRAO = [
    ("lapsbm", "data/test/LapsBM-F004/LapsBM-F004.tar.gz"),
    ("coddef", "data/dev/locutor/locutor.tar.gz"),
]
EXT_AUDIO = (".wav", ".flac", ".mp3", ".ogg")


def eh_lixo(nome):
    """Metadados do macOS dentro de tar: '._arquivo' (AppleDouble) e a pasta '__MACOSX'."""
    partes = Path(nome).parts
    return any(p.startswith("._") or p == "__MACOSX" or p == ".DS_Store" for p in partes)


def extrair(caminho_tar, destino):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    with tarfile.open(caminho_tar, "r:gz") as t:
        todos = t.getmembers()
        membros = [m for m in todos if not eh_lixo(m.name)]
        n_lixo = len(todos) - len(membros)
        try:
            t.extractall(destino, members=membros, filter="data")   # Python 3.12+: bloqueia caminhos perigosos
        except TypeError:                                            # Python mais antigo: checagem manual
            base = destino.resolve()
            for m in membros:
                alvo = (destino / m.name).resolve()
                if not str(alvo).startswith(str(base)):
                    raise RuntimeError(f"caminho suspeito no tar: {m.name}")
            t.extractall(destino, members=membros)
    return [m.name for m in membros], n_lixo


def analisar(caminho_tar, destino, saida):
    membros, n_lixo = extrair(caminho_tar, destino)
    exts = Counter(Path(m).suffix.lower() or "(pasta/sem ext.)" for m in membros)
    saida.append(f"membros úteis no tar: {len(membros)} (ignorados {n_lixo} arquivos '._' do macOS) | por extensão: {dict(exts)}")
    saida.append("primeiros nomes: " + " | ".join(membros[:8]))

    raiz = Path(destino)
    audios = sorted(p for p in raiz.rglob("*") if p.suffix.lower() in EXT_AUDIO and not eh_lixo(p.name))
    textos = sorted(p for p in raiz.rglob("*") if p.suffix.lower() in (".txt", ".lab", ".trans", ".json", ".csv", ".tsv")
                    and not eh_lixo(p.name))
    saida.append(f"áudios: {len(audios)} | arquivos de texto/metadados: {len(textos)}")

    if audios:
        info = sf.info(str(audios[0]))
        saida.append(f"formato do 1º áudio ({audios[0].name}): {info.samplerate} Hz, {info.channels} canal(is), "
                     f"{info.frames / info.samplerate:.1f} s, {info.format}/{info.subtype}")
        durs, ruins = [], []
        for p in audios:
            try:
                i = sf.info(str(p))
                durs.append(i.frames / i.samplerate)
            except Exception:
                ruins.append(p.name)
        if durs:
            saida.append(f"duração ({len(durs)} áudios): mín {min(durs):.1f} s | média {sum(durs)/len(durs):.1f} s | "
                         f"máx {max(durs):.1f} s | total {sum(durs)/60:.1f} min")
        if ruins:
            saida.append(f"ATENÇÃO: {len(ruins)} áudios ilegíveis, ex.: {ruins[:3]}")

    pares = sum(1 for a in audios if a.with_suffix(".txt").exists())
    saida.append(f"áudios com .txt de mesmo nome: {pares} de {len(audios)}")
    nao_utf8 = 0
    for t in textos:
        try:
            t.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            nao_utf8 += 1
    saida.append(f"textos que NÃO são UTF-8 válido: {nao_utf8} de {len(textos)}"
                 + (" (provavelmente latin-1; o manifesto vai tratar)" if nao_utf8 else ""))
    for t in textos[:3]:
        bruto = t.read_bytes()
        try:
            conteudo = bruto.decode("utf-8")
        except UnicodeDecodeError:
            conteudo = bruto.decode("latin-1")
        saida.append(f"  exemplo {t.name}: {conteudo.strip().replace(chr(10), ' ⏎ ')[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo")
    ap.add_argument("--arquivo", help="caminho do .tar.gz dentro do repositório")
    a = ap.parse_args()
    jobs = [(a.repo, a.arquivo)] if a.repo and a.arquivo else PADRAO

    from huggingface_hub import hf_hub_download

    saida = []
    for repo, arquivo in jobs:
        saida += ["=" * 70, f"falabrasil/{repo} :: {arquivo}", "=" * 70]
        try:
            p = hf_hub_download(f"falabrasil/{repo}", arquivo, repo_type="dataset")
            analisar(p, Path("data/amostra") / repo / Path(arquivo).stem.replace(".tar", ""), saida)
        except Exception as e:
            saida.append(f"(falhou: {type(e).__name__}: {e})")
        saida.append("")
    texto = "\n".join(saida)
    print(texto)
    Path("exploracao_tar.txt").write_text(texto, encoding="utf-8")
    print("Salvo em exploracao_tar.txt")


if __name__ == "__main__":
    main()