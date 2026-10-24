# EDP + STFT em áudio pt-BR

Aplica EDPs (calor e Perona-Malik) ao espectrograma de áudio de fala, reconstrói o som e compara imagens, áudios e métricas. Também mede se o realce ajuda ou atrapalha o reconhecimento de fala (Whisper local).

## Arquivos

| Arquivo | Para quê |
|---|---|
| `edp_stft_passo_a_passo.ipynb` | Notebook explicado (comece por aqui) |
| `edp_lab.py` | Biblioteca: STFT, EDPs, Wiener, métricas, gráficos |
| `run_lab.py` | Roda o laboratório de espectrogramas e salva em `outputs/` |
| `gravar.py` | Grava sua voz pelo microfone (offline) |
| `inspecionar_hf.py`, `explorar_tar.py` | Exploram a estrutura dos datasets do Hugging Face |
| `preparar_dados.py` | Baixa LapsBM/coddef, converte para 16 kHz e gera manifestos CSV |
| `textutil.py` | Normalização de texto (números por extenso, sem pontuação) e WER |
| `transcrever.py` | Transcreve um áudio com faster-whisper, em CPU |
| `benchmark_asr.py` | WER do Whisper com ruído e cada técnica de realce |
| `exploracao_tar.txt`, `inspecao_hf.txt` | Resultado das explorações dos datasets |

`data/`, `outputs/` e `resultados_asr/` são gerados localmente e não ficam no Git.

## Passo a passo

### 1. Ambiente virtual e dependências

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate`.
Para ler `.mp3`, o `librosa` pode precisar do `ffmpeg`.

### 2. Teste rápido (sem internet)

```powershell
python run_lab.py --sintetico
```

### 3. Com áudio real, tudo local

```powershell
python run_lab.py --audio minha_fala.wav --snr 10 --iter 3
python run_lab.py --pasta meus_audios --n 5 --benchmark
pip install sounddevice
python gravar.py --seg 6 --saida meus_audios/fala1.wav
```

### 4. Datasets pt-BR (FalaBrasil, Hugging Face)

Ids confirmados, todos com licença MIT: `falabrasil/lapsbm` (35 locutores, ~0,9 h, falas de ~4 s), `falabrasil/coddef` (1 locutor, ~1,4 h, falas longas), `falabrasil/constituicao`, `falabrasil/cetuc`. O áudio vem em pacotes `.tar.gz`.

```powershell
python inspecionar_hf.py
python explorar_tar.py
python preparar_dados.py --dataset lapsbm coddef
```

Gera `data/manifests/lapsbm.csv`, `coddef.csv` e `todos.csv`, com áudio em 16 kHz mono.

### 5. Reconhecimento de fala (local)

```powershell
python transcrever.py meus_audios/fala1.wav --modelo tiny
python benchmark_asr.py --modelo tiny --por-locutor 3
python benchmark_asr.py --modelo base --por-locutor 20
```

Na primeira vez, o faster-whisper baixa os pesos do modelo; depois roda offline. O benchmark compara WER (com IC 95% por bootstrap) de: sem realce, Perona-Malik, Wiener, Wiener+PM e PM seletivo, em áudio limpo e com ruído branco a 20/10/5/0 dB. Resultados em `resultados_asr/`.

Atenção: o LapsBM foi gravado em ambiente não controlado, então o "limpo" já tem algum ruído real.

## Métodos

| Nome | O que é |
|---|---|
| `calor`, `perona_malik` | EDPs na magnitude do espectrograma (dB) |
| `wiener` | Redução de ruído clássica (ruído estimado nas pausas) |
| `wiener+pm` | Wiener, depois Perona-Malik como acabamento |
| `pm_seletivo` | Perona-Malik só onde há ruído (máscara) |

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

## Resultados preliminares (sinal sintético, não é fala)

Com 3 iterações e SNR de entrada de 10 dB, o Perona-Malik fica em ~10,3 dB e o calor cai para ~5,8 dB. Com muitas iterações os dois pioram. As EDPs deixam o espectrograma mais liso, mas quase não baixam o ruído de fundo. Confirmar com fala real.

## Referências

* FalaBrasil, Speech Datasets in Brazilian Portuguese: github.com/falabrasil/speech-datasets
* Perona & Malik (1990), Scale-space and edge detection using anisotropic diffusion
