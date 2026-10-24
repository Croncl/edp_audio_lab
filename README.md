# EDP + STFT em áudio pt-BR — passo a passo

Aplica EDPs (calor e Perona-Malik) ao espectrograma de um áudio de fala, reconstrói o som e compara
imagens, áudios e métricas. Tudo com explicação matemática no notebook.

## Conteúdo

| Arquivo | Para quê |
|---|---|
| `edp_stft_passo_a_passo.ipynb` | **Comece por aqui.** Notebook explicado (9 gráficos, áudios, métricas) |
| `edp_lab.py` | Biblioteca: STFT, EDPs, pipeline, métricas, gráficos |
| `run_lab.py` | Executa tudo por linha de comando e salva em `outputs/` |
| `gravar.py` | Grava sua voz pelo microfone (offline) |
| `requirements.txt` | Dependências |
| `outputs/` | Resultado de um teste com **sinal sintético** (não é fala real) |

## Passo a passo

### 1. Criar o ambiente virtual

Linux / macOS:
```bash
cd edp_audio_lab
python3 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
```powershell
cd edp_audio_lab
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalar dependências
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Para ler `.mp3` o `librosa` pode precisar do `ffmpeg` instalado no sistema. `.wav`/`.flac` funcionam sem ele.

### 3. Teste rápido (sem internet, sinal sintético)
```bash
python run_lab.py --sintetico
```
Gera 9 figuras, 4 `.wav` e `varredura.csv` em `outputs/`.

### 4. Com áudio pt-BR real — tudo local

O pipeline inteiro (STFT, EDPs, métricas, gráficos) roda **no seu computador, sem internet e sem nenhum modelo de IA**:
são métodos numéricos clássicos. Só o download de datasets precisa de rede.

**Opção A — sua própria voz (mais simples):**
```powershell
pip install sounddevice
python gravar.py --seg 6 --saida meus_audios/fala1.wav     # fale uma frase em português
python gravar.py --seg 6 --saida meus_audios/fala2.wav
python run_lab.py --pasta meus_audios --n 5 --benchmark
```

**Opção B — pasta com áudios que você já tem** (dataset baixado, etc.):
```powershell
python run_lab.py --pasta C:\caminho\da\pasta --n 5 --benchmark
```
**Opção C — um arquivo:**
```powershell
python run_lab.py --audio minha_fala.wav --snr 10 --iter 3
```
**Opção D — dataset FalaBrasil, baixado uma vez e usado offline.** Ids confirmados: `falabrasil/mf` (15 min, 2 locutores)
e `falabrasil/coddef` (≈ 1 h 25 min, 1 locutor). Com `git` + `git lfs`:
```powershell
git lfs install
git clone https://huggingface.co/datasets/falabrasil/mf
python run_lab.py --pasta mf --n 5 --benchmark
```
(ou tente o download direto: `python run_lab.py --hf falabrasil/mf --n 5`). Confira a licença antes de usar.

### 5. Abrir o notebook
```bash
jupyter notebook edp_stft_passo_a_passo.ipynb
```
Na primeira célula de dados, troque `FONTE = "sintetico"` por `"hf"` ou `"arquivo"`.
No Google Colab: suba `edp_lab.py` (ou rode a célula `%%writefile`) e descomente a linha `!pip install`.

## Métodos comparados

| Nome | O que é |
|---|---|
| `calor`, `perona_malik` | EDPs puras na magnitude do espectrograma |
| `wiener` | redução de ruído clássica (ruído estimado nas pausas da fala) |
| `wiener+pm` | Wiener e depois Perona-Malik como acabamento |
| `pm_seletivo` | Perona-Malik só onde há ruído (máscara); a fala fica protegida |

Os `.wav` de saída têm o mesmo volume (RMS) do áudio limpo, para comparar de ouvido sem o viés do volume.

## Parâmetros principais

| Flag | Padrão | O que faz |
|---|---|---|
| `--snr` | 10 | SNR do ruído branco adicionado (dB). Dá a referência limpa para medir |
| `--iter` | 3 | Iterações da EDP. **Mais não é melhor** (veja a varredura) |
| `--iter-sel` | 10 | Iterações do PM seletivo |
| `--D` | 0.2 | Coeficiente do calor (≤ 0.25) |
| `--kappa` | 10 | Limiar de borda do Perona-Malik, em dB |
| `--dt` | 0.2 | Passo do Perona-Malik (≤ 0.25) |
| `--duracao` | 4 | Segundos analisados por áudio |
| `--benchmark` | – | Roda SNR 20/10/5/0 dB em todos os áudios → `benchmark.csv` |

## Metodologia sugerida

1. Fala limpa pt-BR → ruído controlado (SNR 20/10/5/0 dB) → referência limpa disponível.
2. Comparar: sem EDP, calor, Perona-Malik (+ baselines: gaussiano, mediana, subtração espectral).
3. Ajustar hiperparâmetros em validação, relatar em teste, com média ± desvio entre locutores.
4. Métricas: SNR, distância log-espectral, STOI (e PESQ / teste de escuta).

## O que o teste com sinal sintético mostrou

(Sinal sintético ≠ fala; confirme com áudio real.)

* O pipeline é correto: com 0 iterações a reconstrução é idêntica (SNR > 100 dB).
* Com 3 iterações e SNR de entrada de 10 dB: Perona-Malik ≈ 10,3 dB (ruidoso: 10,0); calor ≈ 5,8 dB (piorou).
  Com 20 iterações os dois pioram. As EDPs deixam o espectrograma mais liso (LSD melhora ~1,3-1,6 dB), mas **não baixam o nível do
  ruído de fundo** e a SNR de forma de onda só melhora muito pouco, e só com poucas iterações.
* Moral: as EDPs regularizam a imagem; para remover ruído de fato, combine com subtração espectral/Wiener, ou use as EDPs em *inpainting*.

## Referências usadas

* FalaBrasil — Speech Datasets in Brazilian Portuguese (github.com/falabrasil/speech-datasets)
* Tagarela: a Portuguese speech dataset from podcasts (arXiv 2603.15326)
* Perona & Malik (1990), *Scale-space and edge detection using anisotropic diffusion*

## Se o download do Hugging Face falhar

* `401 / Repository Not Found` = id errado, repositório privado/gated ou exige login. Não é problema do seu código.
* Veja o que existe no repo: `python run_lab.py --listar-hf falabrasil/mf`
* Ou pule o Hugging Face: use as opções A, B ou C acima (offline).
* Se o repo exigir login: `huggingface-cli login`.
