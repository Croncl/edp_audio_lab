# EDP + STFT em áudio pt-BR: realce de fala clássico e neural

Trabalho de disciplina. Trata o espectrograma de fala como imagem e testa várias formas de reduzir ruído: **EDPs de difusão** (calor e Perona-Malik), **Wiener** e uma **U-Net** treinada. Cada técnica é medida de dois jeitos: por **qualidade do sinal** (SNR, distância log-espectral, STOI) e pelo **reconhecimento de fala** (WER do Whisper, local).

O relatório completo, com métodos, tabelas, o que funcionou e o que não funcionou, está em [`docs/RELATORIO.md`](docs/RELATORIO.md).

## Principais resultados

- **EDP sozinha não remove ruído.** O calor piora o sinal (difunde igual em tudo). O Perona-Malik é quase neutro. A melhor EDP é o PM **seletivo** (só age onde há ruído): +1,6 dB de SNR com ruído branco a 10 dB.
- **Wiener** é bem melhor que qualquer EDP sozinha (+5 dB). A melhor combinação clássica é **Wiener + PM seletivo**.
- **U-Net** (1,9 M de parâmetros, treinada no CETUC, testada no LapsBM) é a melhor com ruído forte: com ruído branco a 0 dB, +10,4 dB de SNR e +0,13 de STOI sobre o ruidoso. Com ruído fraco (20 dB), processar piora. A EDP depois da U-Net não ajuda.
- **Para o ASR, nenhuma técnica ajudou.** O Whisper já é robusto a ruído, e o realce introduz distorções. A U-Net piora muito com babble a 0 dB (+16 pontos de WER). Ganhar em SNR/STOI não implica reduzir o WER.
- **Limitação principal:** o ruído de treino e o de teste são sintéticos e da mesma família. O ganho em ruído real não foi medido.

> **Aviso (DC).** O LapsBM tem um deslocamento DC grande. Isso invalidou os SNRs *rotulados* dos primeiros experimentos de ASR, que foram refeitos (relatório, seção 14). O código atual remove o DC.

## Estrutura

```
edp_lab.py              biblioteca: STFT, EDPs, Wiener, máscaras, ruídos, métricas, gráficos
run_lab.py              laboratório de espectrogramas e benchmark de qualidade do sinal
edp_stft_passo_a_passo.ipynb   notebook explicado (comece por aqui)
preparar_dados.py       baixa LapsBM/CETUC/coddef, converte para 16 kHz e gera manifestos CSV
textutil.py             normalização de texto (números por extenso, sem pontuação) e WER
benchmark_asr.py        WER do Whisper por técnica, ruído e SNR (com IC 95%)
transcrever.py          transcreve um áudio com faster-whisper (CPU ou GPU NVIDIA)
gravar.py               grava sua voz pelo microfone
treino/                 U-Net: dados.py, modelo.py, treinar.py, avaliar.py, diagnostico.py
registro.py             camada fina sobre o MLflow
modelos/unet_v1/best.pt modelo treinado (época 25)
docs/                   RELATORIO.md, resultados/ (CSVs e gráficos), exploracao/
inspecionar_hf.py, explorar_tar.py   exploram a estrutura dos datasets (já usados)
```

`data/`, `outputs*/`, `resultados_*/`, `meus_audios*/`, `mlflow.db` e `mlartifacts/` são gerados localmente e não ficam no Git.

## Instalação

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`.

O `torch` instalado pelo `pip` pode vir sem CUDA. Para treinar em GPU NVIDIA, instale a versão com CUDA pelo seletor de [pytorch.org](https://pytorch.org). Os scripts de ASR (`benchmark_asr.py`, `transcrever.py`) usam CUDA automaticamente se existir e funcionar; senão, caem para a CPU. GPU AMD não é aproveitada (o `faster-whisper` só tem CUDA e CPU).

## Passo a passo

### 1. Teste rápido (sem internet)

```powershell
python run_lab.py --sintetico
```

### 2. Laboratório com áudio real

```powershell
python run_lab.py --audio minha_fala.wav --snr 10 --iter 3
python run_lab.py --pasta meus_audios --n 5 --benchmark
python gravar.py --seg 6 --saida meus_audios/fala1.wav     # pip install sounddevice
```

### 3. Datasets pt-BR (FalaBrasil, Hugging Face, licença MIT)

```powershell
python inspecionar_hf.py
python preparar_dados.py --dataset lapsbm
python preparar_dados.py --dataset cetuc --splits train --max-locutores 8
python preparar_dados.py --dataset cetuc --splits dev --max-locutores 3
```

Gera `data/manifests/*.csv` com áudio em 16 kHz mono. LapsBM: 35 locutores, 0,9 h. CETUC usado no treino: 8 locutores (11,3 h) para treino e 3 (4,6 h) para validação.

### 4. Reconhecimento de fala

```powershell
python transcrever.py data/proc/lapsbm/test/<locutor>/<fala>.wav --modelo small
python benchmark_asr.py --modelo small --por-locutor 2 --snrs 10 5 0 --ruido babble `
    --frentes nenhum wiener+pm_sel unet unet+pm_sel --saida resultados_asr/babble
```

Frentes: `nenhum`, `perona_malik`, `wiener`, `wiener+pm`, `pm_seletivo`, `wiener+pm_sel`, `unet`, `unet+pm_sel`. Ruídos: `branco`, `rosa`, `marrom`, `babble`. Na primeira vez, o faster-whisper baixa os pesos do modelo; depois roda offline. Use o modelo `small` ou maior: com o `tiny`, o Whisper alucina em ruído forte e o experimento mede o modelo, não o realce.

### 5. Treino e avaliação da U-Net

```powershell
python treino/treinar.py --teste-rapido                    # 1 a 2 min, só para ver se roda
python treino/treinar.py --nome unet_v1 --epocas 30 --mlflow
python treino/avaliar.py --modelo modelos/unet_v1/best.pt --mlflow
```

Na CPU, o treino leva ~9 min por época; numa GPU NVIDIA é bem mais rápido.

### 6. Painel MLflow

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre em `http://localhost:5000`, com as curvas de treino e as métricas da avaliação.

## Métodos

| Nome | O que é |
|---|---|
| `calor`, `perona_malik` | EDPs na magnitude do espectrograma (dB) |
| `wiener` | Redução de ruído clássica (ruído estimado nas pausas) |
| `wiener+pm` | Wiener, depois Perona-Malik em todo o espectrograma |
| `pm_seletivo` | Perona-Malik só onde há ruído (máscara pelo piso de ruído) |
| `wiener+pm_sel` | Wiener, depois PM seletivo (melhor combinação clássica) |
| `pm_vales` | PM só nos vales entre harmônicos (**resultado negativo**, mantido como registro) |
| `unet` | Máscara aprendida por uma U-Net sobre o espectrograma |
| `unet+pm_sel` | U-Net, depois PM seletivo (não ajuda) |

## Parâmetros principais (`run_lab.py`)

| Flag | Padrão | O que faz |
|---|---|---|
| `--snr` | 10 | SNR do ruído branco adicionado (dB) |
| `--iter` | 3 | Iterações da EDP (mais não é melhor) |
| `--iter-sel` | 10 | Iterações do PM seletivo |
| `--D`, `--dt` | 0.2 | Calor e passo do PM (máx. 0.25) |
| `--kappa` | 10 | Limiar de borda do PM, em dB |
| `--duracao` | 4 | Segundos analisados por áudio |
| `--benchmark` | | SNR 20/10/5/0 dB em todos os áudios |

## Limitações

Ruído sintético (branco, rosa, marrom, babble montado), um único treino e uma semente, 11 h de treino, 70 falas no ASR, um modelo Whisper (`small`). Detalhes e a lista completa na seção 19 do relatório.

## Referências

- FalaBrasil, Speech Datasets in Brazilian Portuguese: github.com/falabrasil/speech-datasets
- Perona & Malik (1990), *Scale-space and edge detection using anisotropic diffusion*
- Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*
- Taal et al. (2011), *An algorithm for intelligibility prediction of time-frequency weighted noisy speech* (STOI)
