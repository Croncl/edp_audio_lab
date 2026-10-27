"""
dados.py: Dataset de treino com ruído gerado na hora (nada de ruído pré-gravado).

Cada item: um trecho aleatório de fala limpa + ruído de tipo e SNR sorteados.
  tipos: branco, rosa, marrom, babble (soma de falas de outros trechos)
  SNR  : uniforme em [snr_min, snr_max] dB
Entrada e alvo são normalizados pelo MESMO fator (pico do ruidoso = 0,9), que é também
o que se faz na inferência. No modo `fixo` (validação) tudo é determinístico.
"""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

SR = 16000
TIPOS = ("branco", "rosa", "marrom", "babble")


def gerar_ruido(n, tipo, rng, pool=None, k=6):
    if tipo == "branco":
        return rng.standard_normal(n)
    if tipo in ("rosa", "marrom"):
        X = np.fft.rfft(rng.standard_normal(n))
        f = np.arange(len(X), dtype=np.float64)
        f[0] = 1.0
        return np.fft.irfft(X / f ** (0.5 if tipo == "rosa" else 1.0), n)
    r = np.zeros(n)
    for j in rng.choice(len(pool), size=min(k, len(pool)), replace=False):
        a = np.asarray(pool[j], dtype=np.float64)
        a = np.tile(a, int(np.ceil(n / len(a)) + 1))
        ini = int(rng.integers(0, len(a) - n))
        r += a[ini:ini + n]
    return r


def ler_trecho(caminho, n, rng, aleatorio=True):
    """Lê `n` amostras (início aleatório); completa com zeros se o áudio for mais curto."""
    total = sf.info(caminho).frames
    ini = int(rng.integers(0, total - n)) if (aleatorio and total > n) else 0
    y, _ = sf.read(caminho, start=ini, frames=n, dtype="float32", always_2d=False)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return y - y.mean()          # remove DC (alguns datasets tem DC grande)


class DadosRuido(Dataset):
    def __init__(self, manifesto, split, duracao=3.0, n_amostras=4000, snr=(-5.0, 20.0),
                 tipos=TIPOS, fixo=False, pool_babble=200, seed=0, max_locutores=None):
        df = pd.read_csv(manifesto)
        df = df[df.split == split]
        if max_locutores:
            df = df[df.speaker.isin(sorted(df.speaker.unique())[:max_locutores])]
        assert len(df), f"nenhuma fala em {manifesto} com split={split}"
        self.arqs = df.wav.tolist()
        self.n = int(duracao * SR)
        self.n_amostras, self.snr, self.tipos, self.fixo = n_amostras, snr, tuple(tipos), fixo
        rng = np.random.default_rng(seed)
        self.ordem = rng.permutation(len(self.arqs))
        self.pool = []
        if "babble" in self.tipos:
            for i in rng.choice(len(self.arqs), size=min(pool_babble, len(self.arqs)), replace=False):
                self.pool.append(ler_trecho(self.arqs[i], self.n, rng))

    def __len__(self):
        return self.n_amostras

    def __getitem__(self, i):
        rng = np.random.default_rng(10_000 + i) if self.fixo else np.random.default_rng()
        arq = self.arqs[self.ordem[i % len(self.arqs)]] if self.fixo else self.arqs[int(rng.integers(len(self.arqs)))]
        limpo = ler_trecho(arq, self.n, rng, aleatorio=True).astype(np.float64)
        limpo = limpo - limpo.mean()
        tipo = self.tipos[int(rng.integers(len(self.tipos)))]
        snr = float(rng.uniform(*self.snr))
        r = gerar_ruido(self.n, tipo, rng, self.pool)
        p_alvo = np.mean(limpo ** 2) / (10 ** (snr / 10)) + 1e-20
        r = r * np.sqrt(p_alvo / (np.mean(r ** 2) + 1e-20))
        ruidoso = limpo + r
        g = 0.9 / (np.max(np.abs(ruidoso)) + 1e-9)
        return (torch.from_numpy((ruidoso * g).astype(np.float32)),
                torch.from_numpy((limpo * g).astype(np.float32)), torch.tensor(snr, dtype=torch.float32))
