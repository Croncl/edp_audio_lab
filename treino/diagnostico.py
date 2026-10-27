"""diagnostico.py: por que o SNR da U-Net e do Wiener fica quase constante no avaliar.py?

    python treino/diagnostico.py
Para 3 falas, ruído branco a 20 dB: mostra SNR bruto, escala ótima, atraso e DC.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import correlate

RAIZ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(RAIZ), str(RAIZ / "treino")]
import edp_lab as L  # noqa: E402
from modelo import carregar, realcar  # noqa: E402


def snr(a, b):
    return 10 * np.log10(np.sum(a ** 2) / (np.sum((a - b) ** 2) + 1e-12))


modelo, _ = carregar("modelos/unet_v1/best.pt")
df = pd.read_csv("data/manifests/lapsbm.csv").sort_values("id").groupby("speaker").head(1).head(3)
for r in df.itertuples():
    y, _ = sf.read(r.wav, dtype="float32")
    y = y[:64000]
    y = y * (0.9 / np.abs(y).max())
    yr = L.adicionar_ruido_tipo(y, 20, "branco", seed=1)
    print(f"\n{r.speaker}: DC/desvio = {abs(y.mean()) / y.std():.3f} | RMS ref {y.std():.3f} | RMS ruidoso {yr.std():.3f}")
    print(f"  ruidoso: SNR {snr(y.astype(float), yr.astype(float)):.1f} dB")
    saidas = {"unet": realcar(modelo, yr)}
    try:
        saidas["wiener"] = L.aplicar_edp(yr, "perona_malik", pre="wiener", n_iter=0)["y_out"]
    except Exception as e:
        print("  wiener indisponível:", repr(e))
    for nome, o in saidas.items():
        n = min(len(y), len(o))
        a, b = y[:n].astype(float), np.asarray(o[:n], dtype=float)
        alfa = (a @ b) / (b @ b + 1e-12)
        c = correlate(a, b, mode="full", method="fft")
        atraso = int(np.argmax(c) - (n - 1))
        print(f"  {nome}: SNR {snr(a, b):+.1f} dB | RMS saída {b.std():.3f} | escala ótima {alfa:.2f} "
              f"-> SNR {snr(a, alfa * b):+.1f} dB | atraso {atraso} amostras | tamanho {len(o)}/{len(y)}")
