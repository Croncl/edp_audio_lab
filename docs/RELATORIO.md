# Relatório: EDPs no espectrograma de fala pt-BR

Registro do projeto `edp_audio_lab`: o que foi feito, os resultados, o que funcionou, o que não funcionou e por quê. Escrito também como material de estudo.

---

## 1. Objetivo e pergunta de pesquisa

Tratar o espectrograma de um áudio como uma **imagem** e aplicar **equações diferenciais parciais (EDPs)** de difusão nela (equação do calor e Perona-Malik), para ver:

1. Se a EDP melhora a **qualidade do sinal** de fala com ruído (SNR, distância log-espectral, STOI).
2. Se o realce melhora o **reconhecimento automático de fala** (WER do Whisper).
3. Como se comparam com um baseline clássico (Wiener) e com combinações.

**Resposta curta:** as EDPs puras não melhoram o som de forma útil. A melhor EDP é o Perona-Melik seletivo (age só onde há ruído). O Wiener é muito melhor que qualquer EDP sozinha. A melhor combinação foi **Wiener + PM seletivo**. Nenhuma técnica melhorou o WER do Whisper.

---

## 2. Pipeline

```
áudio x(t) --STFT--> X(t,f) = |X| · e^{jφ}
|X| em dB  =  imagem u(t,f)
u evolui por uma EDP (ou por filtro Wiener, ou os dois)   ->   u'
áudio' = ISTFT( 10^{u'/20} · e^{jφ} )      (fase φ original mantida)
```

Parâmetros: 16 kHz, janela de 512 amostras (32 ms), passo de 128 (8 ms), janela de Hann.

### 2.1 STFT
Corta o sinal em janelas curtas sobrepostas e faz a FFT de cada uma. Resultado: matriz complexa tempo × frequência. A **janela** controla o compromisso: janela curta enxerga bem o tempo e borra a frequência; janela longa o contrário (princípio de incerteza).

### 2.2 Por que dB e por que manter a fase
- A magnitude varia em ordens de grandeza; em dB (`20·log10`) a imagem fica mais próxima do que o ouvido percebe e a difusão se comporta melhor.
- A fase carrega o alinhamento temporal. Com fase aleatória ou zero o som se desfaz (testado no notebook). Por isso a EDP age só em `|X|`.
- **Inconsistência da STFT:** nem toda matriz tempo-frequência é a STFT de algum sinal. Depois da EDP, o ISTFT "projeta" o resultado num sinal real, e o espectrograma do áudio reconstruído difere do que a EDP produziu. Isso gera artefatos além do efeito da EDP.

### 2.3 As técnicas

**Equação do calor.** `∂u/∂τ = D·Δu`. Esquema explícito: `u ← u + D·(soma das diferenças aos 4 vizinhos)`. Estável se `D ≤ 0,25`. Difunde **igualmente em todo lugar**: suaviza o ruído, mas também apaga harmônicos e transientes.

**Perona-Malik.** `∂u/∂τ = div( c(|∇u|) ∇u )`, com `c(g) = exp(-(g/κ)²)`. Onde o gradiente é pequeno (região lisa, ruído) `c ≈ 1` e suaviza; onde é grande (borda, harmônico) `c ≈ 0` e preserva. `κ` (em dB) é o limiar do que conta como borda.

**Wiener (baseline clássico).** Estima o ruído nos 10% de quadros de menor energia (pausas), assume ruído **estacionário**, e multiplica cada bin por um ganho `G = ξ/(1+ξ)`, `ξ = max(Py/Pn − 1, 0)`. Piso de ganho 0,1 e suavização no tempo para reduzir "ruído musical".

**PM seletivo.** Perona-Malik multiplicado por uma máscara (0 a 1) que vale ≈ 1 onde o espectrograma está perto do piso de ruído (+8 dB de margem) e ≈ 0 onde está bem acima (fala). A fala fica protegida.

**PM vales (ideia do usuário, adaptada).** Máscara pela **forma** do espectro: age só nos pontos ≥ 10 dB abaixo do máximo local em frequência (entre harmônicos), picos intocados.

**Combinações:** `wiener+pm` (PM em tudo depois do Wiener) e `wiener+pm_sel` (PM seletivo depois do Wiener).

---

## 3. Dados

- **LapsBM** (FalaBrasil, Hugging Face, licença MIT): 35 locutores, 700 falas, ~0,9 h, falas de ~4,6 s, 22,05 kHz originais, convertidas para 16 kHz mono. Gravado em ambiente não controlado, então o "limpo" já tem algum ruído real.
- **coddef**: 1 locutor, ~1,4 h, falas de 7 a 77 s (16 delas > 30 s, limite do Whisper). Preparado, mas não usado nos experimentos.
- Textos normalizados dos dois lados (minúsculas, sem pontuação, números por extenso) para o WER não ser inflado por formatação. As referências têm erros de digitação (ex.: "quatoze"), então o WER nunca chega a 0.

---

## 4. Métricas (glossário)

| Métrica | O que mede | Melhor |
|---|---|---|
| **SNR (dB)** | Razão entre a energia do sinal limpo e a do erro na forma de onda | maior |
| **LSD (dB)** | Distância log-espectral: erro médio entre espectrogramas em dB | menor |
| **STOI (0 a 1)** | Inteligibilidade estimada: correlação dos envelopes de energia (bandas, janelas de ~400 ms) com o limpo; calibrada com testes de escuta | maior |
| **WER (%)** | Taxa de erro de palavras do reconhecedor: (substituições + inserções + remoções) / palavras da referência. Pode passar de 100% se o modelo inventar palavras | menor |
| **IC 95%** | Intervalo de confiança por bootstrap pareado (mesmas falas e mesmo ruído). Se o intervalo da diferença não cruza 0, o efeito é estatisticamente claro | |

---

## 5. Experimento A: qualidade do sinal (sem ASR)

35 locutores (uma fala de cada), 4 s por fala, ruído branco, mesma semente para todos os métodos. EDPs: 3 iterações; métodos seletivos: 10 iterações; D = 0,2, κ = 10, dt = 0,2. Média entre os 35.

### 5.1 Resultados com SNR de entrada de 10 dB

| Método | SNR (dB) | LSD (dB) | STOI |
|---|---|---|---|
| ruidoso (sem processar) | 10,04 | 18,55 | 0,774 |
| calor | 5,50 | 17,06 | 0,757 |
| perona_malik | 9,62 | 17,39 | 0,772 |
| pm_seletivo | 11,67 | 17,22 | 0,784 |
| pm_vales | 9,31 | 19,14 | 0,763 |
| wiener | 15,01 | 12,70 | 0,789 |
| wiener+pm | 11,70 | 12,03 | 0,788 |
| **wiener+pm_sel** | **15,80** | **11,12** | **0,792** |

### 5.2 Resultados com SNR de entrada de 5 dB e 0 dB

| Método | SNR 5 dB | STOI 5 dB | SNR 0 dB | STOI 0 dB |
|---|---|---|---|---|
| ruidoso | 5,04 | 0,679 | 0,04 | 0,575 |
| calor | 4,02 | 0,659 | 1,14 | 0,555 |
| perona_malik | 6,00 | 0,676 | 1,56 | 0,573 |
| pm_seletivo | 6,93 | 0,693 | 2,09 | 0,590 |
| pm_vales | 4,51 | 0,666 | −0,43 | 0,563 |
| wiener | 10,84 | 0,704 | 6,40 | 0,604 |
| wiener+pm | 9,88 | 0,704 | 6,97 | 0,605 |
| **wiener+pm_sel** | **12,45** | **0,710** | **8,79** | **0,613** |

### 5.3 Resultados com SNR de entrada de 20 dB

| Método | SNR | LSD | STOI |
|---|---|---|---|
| ruidoso | 20,04 | 11,67 | 0,912 |
| pm_seletivo | 20,44 | 10,59 | 0,909 |
| wiener | 21,96 | 9,64 | 0,906 |
| wiener+pm_sel | 21,22 | 9,74 | 0,901 |
| calor | 6,28 | 10,53 | 0,895 |

### 5.4 Leitura

1. **Calor piora o SNR em todos os níveis** (10 → 5,5 dB). Confirma a observação de que "espalha": difunde igual em tudo, os picos (harmônicos) perdem energia e o vale (onde está o ruído) ganha. Só melhora o LSD, ou seja, a imagem fica mais lisa, mas o som fica pior.
2. **Perona-Malik é quase neutro:** preserva bordas, mas ainda borra o resto. SNR ligeiramente pior que o ruidoso a 10 dB, STOI praticamente igual.
3. **PM seletivo é a melhor EDP:** +1,6 dB de SNR e +0,01 de STOI a 10 dB; +1,9 dB e +0,014 a 5 dB. Agir só onde há ruído funciona melhor do que agir em tudo.
4. **Wiener é muito superior a qualquer EDP sozinha:** +5 dB de SNR a 10 dB e LSD muito menor. Cuidado: assume ruído estacionário; com ruído branco isso é verdade.
5. **Wiener + PM (em tudo) piora o SNR em relação ao Wiener sozinho** (11,7 contra 15,0 dB a 10 dB): o PM aplicado em tudo borra a fala que o Wiener tinha preservado.
6. **Wiener + PM seletivo é a melhor combinação** com ruído de 0 a 10 dB: sobre o Wiener sozinho, +0,8 dB (10 dB), +1,6 dB (5 dB), +2,4 dB (0 dB) de SNR, LSD menor em todos os níveis e STOI ligeiramente maior. O ganho cresce quanto mais ruído, o que faz sentido: sobra mais ruído residual para a EDP limpar.
7. **Com pouco ruído (20 dB) nada supera o sinal sem processar em STOI.** Processar só adiciona artefato. O próprio Wiener+PM sel perde 0,8 dB de SNR para o Wiener a 20 dB.

### 5.5 Ressalva estatística
Os desvios entre locutores são grandes (STOI ≈ 0,065), maiores que as diferenças entre os métodos mais próximos (0,789 contra 0,792). Como todos os métodos processam as mesmas falas e o mesmo ruído, uma comparação pareada com IC seria mais informativa, mas **não foi calculada** para este experimento. Tratar diferenças de STOI de 0,01 como indício, não como prova. As diferenças de SNR (décimos a vários dB) são mais robustas.

---

## 6. Experimento B: ASR (WER do faster-whisper)

70 falas do LapsBM (2 por locutor, 35 locutores), áudio **sem** normalização de pico, mesmo ruído para todas as frentes, IC 95% por bootstrap pareado. Realce igualado em RMS antes de transcrever. CPU, int8.

### 5.1 Primeira tentativa: modelo `tiny` (resultado descartado)

Com ruído branco, WER sem realce de 22,4% (limpo), **99,5% (10 dB)** e **118,9% (0 dB)**. Ao inspecionar as transcrições, o modelo **alucinava**: texto fluente sem relação com a referência, com repetições ("e a sua e a sua e a sua"). O "efeito" do Wiener a 10 dB (−15,8 pontos) era artefato dessa alucinação. **Lição:** com um modelo fraco demais, o experimento mede o modelo, não o realce.

### 6.2 Modelo `small`, ruído branco (SNR 20/15/10 dB)

| Frente | limpo | 20 dB | 15 dB | 10 dB |
|---|---|---|---|---|
| nenhum | 8,9 | 12,1 | 21,7 | 39,1 |
| wiener | 7,8 | 13,6 | 26,5 (PIORA) | 44,6 |
| perona_malik | 8,5 | 12,3 | 23,9 (PIORA) | 42,7 |
| wiener+pm | 8,2 | 13,2 | 23,2 | 42,7 |
| pm_seletivo | 8,8 | 14,7 (PIORA) | 25,1 (PIORA) | 41,8 |

("PIORA" = IC 95% da diferença acima de zero.)

### 6.3 Modelo `small`, ruído babble (várias falas de outros locutores somadas)

| Frente | limpo | 15 dB | 10 dB | 5 dB |
|---|---|---|---|---|
| nenhum | 8,9 | 8,9 | 9,3 | 14,8 |
| wiener | 7,8 | 8,8 | 10,4 | 19,1 (PIORA) |
| perona_malik | 8,5 | 9,1 | 9,5 | 15,1 |
| wiener+pm | 8,2 | 8,9 | 10,8 | 18,2 (PIORA) |

### 6.4 Leitura

1. **Nenhuma frente melhorou o WER com significância**, em nenhum tipo de ruído. Várias pioraram.
2. **O Whisper já é robusto a ruído**: foi treinado com áudio muito variado. O babble quase não o atrapalha (9,3% a 10 dB), ao contrário do ruído branco (39,1%).
3. **Por que o realce piora o ASR:** o realce remove ruído, mas introduz distorções (ruído musical, fase original reaproveitada numa magnitude alterada, inconsistência da STFT) que o modelo não viu no treino. O custo supera o benefício.
4. **Wiener piora com babble** porque o ruído não é estacionário: o ruído estimado nas pausas não representa o ruído durante a fala, e o filtro corta fala junto.
5. **Contradição aparente com o Experimento A:** o realce melhora SNR e STOI, mas não o WER. Não é contradição: SNR e STOI medem a proximidade com o limpo para humanos e para a forma de onda; o Whisper tem outro "ouvido".

### 6.5 Limitações do Experimento B
Só 70 falas (intervalos largos, ex.: −0,7 a +12,2 a 10 dB), um único modelo robusto (`small`), ruídos sintéticos, e `wiener+pm_sel`, `pm_vales` e `calor` **não foram testados em ASR**. A conclusão correta é "não há evidência de benefício nessas condições", e não "nunca ajuda".

---

## 7. Experimentos de ajuste

### 7.1 `pm_vales` (ideia: agir só nos vales, picos intactos)
**Falhou.** SNR 9,31 dB e LSD 19,14 dB a 10 dB, pior que o ruidoso (10,04 e 18,55) e pior que o `pm_seletivo` em tudo. Motivos:
- Suavizar vales os puxa para cima, em direção aos picos vizinhos, e **enche o espaço entre harmônicos** (que no limpo é profundo). A fala fica borrada onde mais importava.
- A máscara por **forma** não distingue sinal de ruído: num espectrograma ruidoso há "vales" e "picos" de ruído também. A máscara do `pm_seletivo`, baseada no **nível relativo ao piso de ruído estimado**, distingue.

Conclusão: o critério certo é "quão acima do ruído estou", não "qual é a forma local".

### 7.2 Sobre "fazer o processo contrário" (difusão reversa)
Não foi testado. Difusão reversa é mal posta (amplifica ruído de alta frequência). O que existe de bem comportado: filtros de choque e *unsharp masking*, ainda não implementados. Ideia para o futuro.

### 7.3 Varredura de `margem` e `kappa` do PM seletivo (SNR médio / STOI, 35 locutores)

| Configuração | 10 dB | 5 dB |
|---|---|---|
| margem 8, κ 10 (padrão) | 11,67 / 0,784 | 6,93 / 0,693 |
| margem 12, κ 10 | 11,82 / 0,773 | 7,17 / 0,679 |
| margem 4, κ 10 | 10,82 / 0,786 | 5,95 / 0,695 |

κ = 10 foi o melhor em quase todas as margens. A margem é um compromisso entre SNR (margens maiores) e STOI (menores). **Margem 8 é a única que fica acima do ruidoso nos dois critérios nos dois níveis.** Ajustar parâmetros muda décimos de dB: o teto do PM seletivo é ~+1,6 dB, muito abaixo do Wiener (+5 dB).

---

## 8. O que funcionou e o que não

| Funcionou | Não funcionou |
|---|---|
| Wiener como baseline: +5 dB de SNR | Equação do calor: piora tudo (difunde igual, borra) |
| PM seletivo: melhor EDP | PM em tudo: neutro ou levemente pior |
| **Wiener + PM seletivo: melhor resultado geral com ruído** | `pm_vales`: pior que não processar |
| Pipeline STFT → dB → EDP → ISTFT (com 0 iterações a reconstrução é idêntica) | Wiener + PM em tudo: perde 3 dB para o Wiener sozinho |
| Máscara baseada no piso de ruído | Qualquer realce para o WER do Whisper |
| Poucas iterações (3) | Muitas iterações (10 a 20): pioram quase tudo |

**Princípio geral:** a EDP é um regularizador de imagem, não um removedor de ruído. Ela alisa o espectrograma, mas não baixa o nível do ruído de fundo. Para tirar ruído, é preciso um método que estime e subtraia o ruído (Wiener, subtração espectral). A EDP funciona melhor como acabamento local, **onde há ruído residual**, que é exatamente o `wiener+pm_sel`.

---

## 9. Limitações gerais

1. **Ruído sintético** (branco, rosa, marrom, babble montado). Ruído real (vento no microfone, eco, clipping, ruído não estacionário) pode se comportar de outro jeito. Rosa e marrom foram implementados e ouvidos, mas não rodados nos benchmarks.
2. **Dados:** o LapsBM já traz algum ruído real e possível processamento do gravador. Isso impõe um teto à melhora medida, mas não invalida a comparação (o "limpo" é igual para todos os métodos).
3. **Amostra:** 35 falas (qualidade) e 70 falas (ASR), 4 s cada. Intervalos largos em ASR; qualidade sem IC pareado.
4. **Pré-processamento diferente entre experimentos:** o benchmark de qualidade normaliza o pico a 0,9 e corta 4 s; o de ASR não normaliza.
5. **Parâmetros** das EDPs afinados só para o `pm_seletivo`, e só com ruído branco.
6. **Fase original** reaproveitada: limita o que qualquer método de magnitude pode fazer.
7. **Estimador de ruído** simples (10% dos quadros de menor energia): assume ruído estacionário e que existem pausas.

---

## 10. Próximos passos sugeridos

1. Testar `wiener+pm_sel` em **ASR** (modelo `small`), com ruído branco e babble.
2. Calcular **IC pareado** para as métricas de qualidade.
3. Rodar com ruído **marrom/rosa** e, principalmente, com **gravações reais ruidosas** (sua voz em ambiente ruim, via `gravar.py`).
4. Estimador de ruído adaptativo (não estacionário), que deve ajudar o Wiener com babble.
5. Reconstrução com **Griffin-Lim** ou fase estimada, para reduzir a inconsistência da STFT.
6. EDP para **inpainting** (preencher trechos apagados), onde ela deve ter vantagem sobre filtros clássicos.
7. Modelos maiores (`medium`, `large-v3`) na RTX 4050, com `--por-locutor 10` ou mais.
8. Realce de bordas controlado (filtro de choque) em vez de difusão reversa.

---

## 11. Arquivos do projeto

| Arquivo | Para quê |
|---|---|
| `edp_lab.py` | STFT, EDPs, Wiener, máscaras, métricas, gráficos, ruídos |
| `run_lab.py` | Figuras, áudios e benchmark de qualidade |
| `benchmark_asr.py` | WER do Whisper por técnica e condição, com IC 95% |
| `transcrever.py` | Transcrição local; usa CUDA se existir, senão CPU |
| `preparar_dados.py`, `textutil.py` | Preparação dos datasets e normalização de texto |
| `docs/resultados/` | CSVs e gráficos dos resultados |

## 12. Referências

- FalaBrasil, Speech Datasets in Brazilian Portuguese: github.com/falabrasil/speech-datasets
- Perona & Malik (1990), *Scale-space and edge detection using anisotropic diffusion*
- Taal et al. (2011), *An algorithm for intelligibility prediction of time-frequency weighted noisy speech* (STOI)
