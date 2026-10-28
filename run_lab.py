"""
run_lab.py — executa o laboratório completo e salva figuras, áudios e métricas em outputs/.

Exemplos:
    python run_lab.py --sintetico                      # teste de fumaça, sem internet
    python run_lab.py --audio meu_audio.wav            # arquivo seu
    python run_lab.py --pasta meus_audios --n 5 --benchmark   # pasta local, offline, + benchmark

Métodos comparados (todos aplicados ao MESMO áudio ruidoso):
    calor, perona_malik     EDPs puras no espectrograma
    wiener                  redução de ruído clássica (baseline que funciona)
    wiener+pm               Wiener e depois Perona-Malik como acabamento
    pm_seletivo             Perona-Malik só onde há ruído (máscara); a fala fica intacta
Os .wav de saída têm o MESMO volume (RMS) do áudio limpo, para a comparação de escuta ser justa.
"""
import argparse
from pathlib import Path

import pandas as pd

import edp_lab as L


def rodar_um(y, nome, saida: Path, snr, D, kappa, dt, n_iter, n_iter_sel):
    saida.mkdir(parents=True, exist_ok=True)
    y_ruid = L.adicionar_ruido(y, snr)
    ref = L.stft(y).ref

    L.fig_onda_e_espectrograma(y, saida=saida / f"{nome}_01_onda_espectrograma.png")
    L.fig_resolucao_stft(y, saida=saida / f"{nome}_02_resolucao_stft.png")
    _, ys = L.fig_magnitude_fase(y, saida=saida / f"{nome}_03_magnitude_fase.png")
    L.fig_funcao_difusao(kappa, saida=saida / f"{nome}_04_funcao_difusao.png")
    L.fig_evolucao_edp(y_ruid, D=D, kappa=kappa, dt=dt, saida=saida / f"{nome}_05_evolucao_edp.png")

    res = L.variantes(y_ruid, D=D, kappa=kappa, dt=dt, n_iter=n_iter, n_iter_sel=n_iter_sel)
    L.fig_comparacao(y, y_ruid, res["calor"], res["perona_malik"], rotulo=f"Com ruído (SNR {snr:g} dB)",
                     saida=saida / f"{nome}_06_comparacao.png")
    L.fig_variantes(y, y_ruid, res, rotulo=f"Ruidoso (SNR {snr:g} dB)", saida=saida / f"{nome}_10_todas_variantes.png")
    L.fig_mascara(y, y_ruid, saida=saida / f"{nome}_11_mascara_seletiva.png")
    L.fig_perfil_espectral(y, y_ruid, res["calor"]["y_out"], res["perona_malik"]["y_out"], ref=ref,
                           saida=saida / f"{nome}_07_perfil_espectral.png")
    L.fig_consistencia(res["calor"], titulo="Calor", saida=saida / f"{nome}_08_consistencia.png")

    df = L.varredura_iteracoes(y, y_ruid, ref, D=D, kappa=kappa, dt=dt)
    base = L.metricas(y, y_ruid, ref)
    L.fig_curvas(df, base=base, saida=saida / f"{nome}_09_curvas_iteracoes.png")
    df.to_csv(saida / f"{nome}_varredura.csv", index=False)

    # .wav com volume igualado ao do limpo (só para escuta; as métricas usam o sinal sem ajuste)
    L.salvar_wav(y, saida / f"{nome}_limpo.wav")
    L.salvar_wav(L.igualar_nivel(y, y_ruid), saida / f"{nome}_ruidoso.wav")
    for k, v in res.items():
        L.salvar_wav(L.igualar_nivel(y, v["y_out"]), saida / f"{nome}_{k.replace('+', '_mais_')}.wav")

    tabela = {"ruidoso": base}
    tabela.update({k: L.metricas(y, v["y_out"], ref) for k, v in res.items()})
    tab = pd.DataFrame(tabela).T.round(3)
    tab.to_csv(saida / f"{nome}_metricas.csv")
    print(f"\n=== {nome} (SNR de entrada {snr:g} dB; EDP {n_iter} it., seletivo {n_iter_sel} it.) ===")
    print(tab.to_string())
    return tab


def benchmark(ys, saida: Path, snrs, D, kappa, dt, n_iter, n_iter_sel):
    linhas = []
    for nome, y in ys.items():
        ref = L.stft(y).ref
        for snr in snrs:
            yr = L.adicionar_ruido(y, snr, seed=1)
            res = L.variantes(yr, D=D, kappa=kappa, dt=dt, n_iter=n_iter, n_iter_sel=n_iter_sel)
            linhas.append(dict(audio=nome, snr_entrada=snr, metodo="ruidoso", **L.metricas(y, yr, ref)))
            for m, v in res.items():
                linhas.append(dict(audio=nome, snr_entrada=snr, metodo=m, **L.metricas(y, v["y_out"], ref)))
    df = pd.DataFrame(linhas)
    df.to_csv(saida / "benchmark.csv", index=False)
    resumo = df.groupby(["snr_entrada", "metodo"])[["SNR_dB", "LSD_dB", "STOI"]].agg(["mean", "std"]).round(3)
    print("\n=== BENCHMARK (média ± desvio entre áudios) ===")
    print(resumo.to_string())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--audio", nargs="+", help="um ou mais arquivos de áudio")
    g.add_argument("--pasta", help="pasta local com áudios (busca .wav/.flac/.mp3/.ogg recursivamente) — 100%% offline")
    g.add_argument("--sintetico", action="store_true", help="sinal sintético (só para testar o código)")
    ap.add_argument("--n", type=int, default=3, help="quantos áudios usar da pasta")
    ap.add_argument("--duracao", type=float, default=4.0, help="segundos de cada áudio")
    ap.add_argument("--snr", type=float, default=10.0, help="SNR do ruído adicionado (dB)")
    ap.add_argument("--iter", type=int, default=3, help="iterações de calor / Perona-Malik / wiener+pm")
    ap.add_argument("--iter-sel", type=int, default=10, help="iterações do PM seletivo")
    ap.add_argument("--D", type=float, default=0.2)
    ap.add_argument("--kappa", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--saida", default="outputs")
    a = ap.parse_args()
    if not (a.audio or a.pasta or a.sintetico):
        ap.error("informe --audio, --pasta ou --sintetico")

    saida = Path(a.saida)
    if a.sintetico:
        ys = {"sintetico": L.audio_sintetico(duracao=a.duracao)}
    else:
        if a.audio:
            caminhos = a.audio
        else:
            caminhos = L.listar_audios_pasta(a.pasta, n=a.n)
            print("Áudios encontrados:", *caminhos, sep="\n  ")
        ys = {Path(c).stem: L.carregar_audio(c, duracao=a.duracao) for c in caminhos}

    nomes = list(ys)
    rodar_um(ys[nomes[0]], nomes[0], saida, a.snr, a.D, a.kappa, a.dt, a.iter, a.iter_sel)
    if a.benchmark and len(ys) > 0:
        benchmark(ys, saida, snrs=(20, 10, 5, 0), D=a.D, kappa=a.kappa, dt=a.dt, n_iter=a.iter, n_iter_sel=a.iter_sel)
    print(f"\nArquivos em: {saida.resolve()}")


if __name__ == "__main__":
    main()
