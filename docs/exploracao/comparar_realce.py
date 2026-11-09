import sys
import pandas as pd
from benchmark_asr_tinybase import erros_por_fala, delta_ic

d = pd.read_csv(sys.argv[1], dtype=str).fillna("")
for snr in ["limpo", "20", "10", "5"]:
    a = d[d.condicao == f"{snr}|nenhum"].sort_values("id")
    b = d[d.condicao == f"{snr}|wiener"].sort_values("id")
    e0, n0 = erros_por_fala(a["ref"].tolist(), a["hip"].tolist())
    e1, n1 = erros_por_fala(b["ref"].tolist(), b["hip"].tolist())
    dw, lo, hi = delta_ic(e1, n1, e0, n0)
    print(f"{snr:>6} dB: Wiener - nenhum = {dw:+.2f} pp  IC95 [{lo:+.2f}, {hi:+.2f}]  {'significativo' if lo > 0 or hi < 0 else 'não significativo'}")
