"""
textutil.py — leitura e normalização de texto para comparar transcrições (WER).

As referências do FalaBrasil vêm em minúsculas, sem pontuação e com números por extenso
("cento e setenta bilhões"), enquanto o Whisper escreve "170 bilhões." Normalizamos os DOIS lados
com a mesma função, senão o WER é inflado por diferenças de formatação e não por erros de reconhecimento.
"""
import re
import unicodedata
from pathlib import Path

try:
    from num2words import num2words
except ImportError:  # sem o pacote, os números ficam como dígitos (o WER pode ficar pior)
    num2words = None


def ler_texto(caminho):
    """Lê um .txt em UTF-8; se falhar, tenta latin-1."""
    bruto = Path(caminho).read_bytes()
    try:
        return bruto.decode("utf-8").strip()
    except UnicodeDecodeError:
        return bruto.decode("latin-1").strip()


def _por_extenso(m):
    n = m.group(0)
    if num2words is None:
        return n
    try:
        return " " + num2words(int(n), lang="pt_BR") + " "
    except Exception:
        return n


def normalizar_texto(t):
    """minúsculas, números por extenso, sem pontuação, espaços únicos (acentos preservados)."""
    t = unicodedata.normalize("NFC", str(t)).lower()
    t = re.sub(r"(?<=\d)\.(?=\d{3}(?!\d))", "", t)          # 1.000 -> 1000
    t = re.sub(r"(\d+),(\d+)", r"\1 vírgula \2", t)          # 2,5 -> 2 vírgula 5
    t = t.replace("%", " por cento ")
    t = re.sub(r"\d+", _por_extenso, t)
    t = re.sub(r"[^\w\s]", " ", t)                           # tira pontuação
    t = t.replace("_", " ")
    return re.sub(r"\s+", " ", t).strip()


def wer_cer(refs, hips):
    """WER e CER (%) sobre listas de textos JÁ normalizados. Usa jiwer."""
    import jiwer
    pares = [(r, h) for r, h in zip(refs, hips) if r.strip()]
    r, h = [p[0] for p in pares], [p[1] for p in pares]
    return 100 * jiwer.wer(r, h), 100 * jiwer.cer(r, h)
