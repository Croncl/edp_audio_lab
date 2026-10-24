"""
inspecionar_hf.py — mostra COMO cada dataset do Hugging Face está organizado, sem baixar os dados.

    python inspecionar_hf.py
    python inspecionar_hf.py --repos lapsbm coddef

Para cada repositório: o início do README (dataset card) e uma amostra da árvore de pastas
(poucas entradas por pasta, poucos níveis). Salva tudo também em inspecao_hf.txt.
"""
import argparse
import os
from itertools import islice

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

MAX_ENTRADAS = 8      # entradas mostradas por pasta
MAX_PROFUNDIDADE = 4  # níveis de pasta
MAX_SUBPASTAS = 2     # quantas subpastas explorar por nível
LINHAS_README = 45


def tamanho(n):
    if n is None:
        return "?"
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def arvore(api, repo, caminho, nivel, saida):
    try:
        entradas = list(islice(api.list_repo_tree(repo, path_in_repo=caminho, repo_type="dataset", recursive=False),
                               MAX_ENTRADAS + 1))
    except Exception as e:
        saida.append(f"{'  ' * nivel}(erro ao listar '{caminho}': {type(e).__name__})")
        return
    mais = len(entradas) > MAX_ENTRADAS
    pastas = []
    for e in entradas[:MAX_ENTRADAS]:
        nome = e.path.split("/")[-1]
        if e.__class__.__name__ == "RepoFolder":
            saida.append(f"{'  ' * nivel}[pasta] {nome}/")
            pastas.append(e.path)
        else:
            saida.append(f"{'  ' * nivel}{nome}  ({tamanho(getattr(e, 'size', None))})")
    if mais:
        saida.append(f"{'  ' * nivel}... (há mais entradas nesta pasta)")
    if nivel < MAX_PROFUNDIDADE:
        for p in pastas[:MAX_SUBPASTAS]:
            arvore(api, repo, p, nivel + 1, saida)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", default=["lapsbm", "coddef", "constituicao", "cetuc"])
    a = ap.parse_args()
    api = HfApi()
    saida = []
    for nome in a.repos:
        repo = nome if "/" in nome else f"falabrasil/{nome}"
        saida.append("=" * 70)
        saida.append(f"REPO: {repo}")
        saida.append("=" * 70)
        try:
            info = api.dataset_info(repo)
            saida.append(f"privado: {info.private} | gated: {info.gated} | licença/tags: "
                         f"{[t for t in (info.tags or []) if t.startswith(('license', 'language', 'task'))]}")
        except Exception as e:
            saida.append(f"(não consegui ler os metadados: {type(e).__name__}: {e})")
        try:
            p = hf_hub_download(repo, "README.md", repo_type="dataset")
            with open(p, encoding="utf-8", errors="replace") as f:
                saida.append("--- README (início) ---")
                saida.extend(l.rstrip() for l in islice(f, LINHAS_README))
        except Exception as e:
            saida.append(f"(sem README acessível: {type(e).__name__})")
        saida.append("--- árvore (amostra) ---")
        arvore(api, repo, "", 0, saida)
        saida.append("")

    texto = "\n".join(saida)
    print(texto)
    with open("inspecao_hf.txt", "w", encoding="utf-8") as f:
        f.write(texto)
    print("\nSalvo em inspecao_hf.txt")


if __name__ == "__main__":
    main()
