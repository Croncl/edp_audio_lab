# Relatório: EDPs no espectrograma de fala pt-BR

Registro do projeto `edp_audio_lab`: o que foi feito, os resultados, o que funcionou, o que não funcionou e por quê. Escrito também como material de estudo.

> **Aviso de correção (leia antes das tabelas de ASR da Parte 1).** Depois de escrever a Parte 1, descobri que os áudios do LapsBM têm um deslocamento DC grande e que o benchmark de ASR não o removia. Por isso o SNR **rotulado** nos experimentos de ASR das seções 6.2, 6.3 e 6.5 estava errado (o SNR real da fala ficou de −6,5 a +6 dB quando se pedia 10 dB). Essas tabelas estão marcadas como **obsoletas** e foram refeitas na Parte 2 (seção 17). As conclusões qualitativas não mudaram. O Experimento A (qualidade do sinal) provavelmente não foi afetado. Detalhes na seção 14.

---

## 1. Objetivo e pergunta de pesquisa

Tratar o espectrograma de um áudio como uma **imagem** e aplicar **equações diferenciais parciais (EDPs)** de difusão nela (equação do calor e Perona-Malik), para ver:

1. Se a EDP melhora a **qualidade do sinal** de fala com ruído (SNR, distância log-espectral, STOI).
2. Se o realce melhora o **reconhecimento automático de fala** (WER do Whisper).
3. Como se comparam com um baseline clássico (Wiener) e com combinações.

**Resposta curta:** as EDPs puras não melhoram o som de forma útil. A melhor EDP é o Perona-Malik seletivo (age só onde há ruído). O Wiener é muito melhor que qualquer EDP sozinha. A melhor combinação foi **Wiener + PM seletivo**. Nenhuma técnica melhorou o WER do Whisper (nem a U-Net da Parte 2).

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

> **Obsoleto nos SNRs:** as seções 6.2, 6.3 e 6.5 usam SNR nominal que, por causa do DC dos áudios, não correspondia ao SNR real. Mantidas só como registro. Valores corretos: Parte 2, seção 17.

70 falas do LapsBM (2 por locutor, 35 locutores), áudio **sem** normalização de pico, mesmo ruído para todas as frentes, IC 95% por bootstrap pareado. Realce igualado em RMS antes de transcrever. CPU, int8.

### 6.1 Primeira tentativa: modelo `tiny` (resultado descartado)

Com ruído branco, WER sem realce de 22,4% (limpo), **99,5% (10 dB)** e **118,9% (0 dB)**. Ao inspecionar as transcrições, o modelo **alucinava**: texto fluente sem relação com a referência, com repetições ("e a sua e a sua e a sua"). O "efeito" do Wiener a 10 dB (−15,8 pontos) era artefato dessa alucinação. **Lição:** com um modelo fraco demais, o experimento mede o modelo, não o realce.

### 6.2 [OBSOLETA] Modelo `small`, ruído branco (SNR nominal 20/15/10 dB)

| Frente | limpo | 20 dB | 15 dB | 10 dB |
|---|---|---|---|---|
| nenhum | 8,9 | 12,1 | 21,7 | 39,1 |
| wiener | 7,8 | 13,6 | 26,5 (PIORA) | 44,6 |
| perona_malik | 8,5 | 12,3 | 23,9 (PIORA) | 42,7 |
| wiener+pm | 8,2 | 13,2 | 23,2 | 42,7 |
| pm_seletivo | 8,8 | 14,7 (PIORA) | 25,1 (PIORA) | 41,8 |

("PIORA" = IC 95% da diferença acima de zero.)

### 6.3 [OBSOLETA] Modelo `small`, ruído babble (SNR nominal)

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

### 6.5 [OBSOLETA] `wiener+pm_sel` em ASR (SNR nominal)

| Ruído | Frente | limpo | 20 dB | 15 dB | 10 dB | 5 dB |
|---|---|---|---|---|---|---|
| branco | nenhum | 8,9 | 12,5 | 21,7 | 39,9 | n/d |
| branco | wiener+pm_sel | 8,2 | 13,3 | 25,8 (PIORA, +4,1) | 41,0 | n/d |
| babble | nenhum | 8,9 | n/d | 8,9 | 9,3 | 14,8 |
| babble | wiener+pm_sel | 8,2 | n/d | 9,3 | 10,3 | 20,4 (PIORA, +5,6) |

**Resultado:** mesmo a melhor combinação em qualidade do sinal **não ajuda o Whisper**. Piora de forma significativa em 15 dB (branco) e 5 dB (babble); nas demais condições a diferença é inconclusiva, com sinal positivo (pior) em quase todas. No limpo a diferença (−0,7) também é inconclusiva.

Isso reforça o ponto da seção 6.4: ganhar em SNR/LSD/STOI (que o `wiener+pm_sel` ganha) não se traduz em menor WER, porque as distorções do realce pesam mais para o reconhecedor do que o ruído removido.

**Nota de reprodutibilidade:** a linha `nenhum` do ruído branco saiu um pouco diferente da rodada anterior (12,5 e 39,9 contra 12,1 e 39,1). Causa provável: o ruído passou a ser gerado por `adicionar_ruido_tipo`, que escala o ruído para a potência exata pedida, enquanto a função antiga usava o desvio-padrão teórico. É uma variação de sorteio, não de método. A comparação entre frentes dentro de uma mesma rodada continua válida (mesmo ruído, pareada). Os números do branco das seções 6.2 e 6.5 não devem ser misturados numa mesma tabela de diferenças.

### 6.6 Limitações do Experimento B
Só 70 falas (intervalos largos, ex.: −0,7 a +12,2 a 10 dB), um único modelo robusto (`small`), ruídos sintéticos, e `calor` e `pm_vales` **não foram testados em ASR** (já eram piores em qualidade do sinal). A conclusão correta é "não há evidência de benefício nessas condições", e não "nunca ajuda".

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
8. **DC dos áudios:** o LapsBM tem DC/desvio de 1,2 a 6,6 (seção 14). Corrigido na Parte 2; as tabelas antigas de ASR ficaram obsoletas.

---

## 10. Próximos passos sugeridos

1. (feito: `wiener+pm_sel` em ASR não ajudou, seção 6.5.)
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

---

# Parte 2: modelo neural (U-Net) e correção do DC

## 13. Resumo da Parte 2

1. **Achado metodológico:** os áudios do LapsBM têm um deslocamento DC grande. Isso distorceu o SNR dos experimentos de ASR e a medida de SNR da U-Net. Foi detectado, medido e corrigido (seção 14).
2. **U-Net treinada** (1,9 M de parâmetros, CETUC, 11,3 h) para prever uma máscara sobre o espectrograma. Em **qualidade do sinal**, ela supera os métodos clássicos com ruído forte (0 a 5 dB): +10,4 dB de SNR e +0,13 de STOI sobre o sinal ruidoso com ruído branco a 0 dB. Com ruído fraco (20 dB), perde para o sinal sem processar e para o Wiener.
3. **Em ASR (Whisper), a U-Net não ajuda:** piora de forma significativa com babble a 5 e 0 dB (+6,0 e +16,3 pontos de WER). Só o WER sem ruído melhora (−1,1 ponto), com evidência fraca.
4. **Conclusão central, válida para as três famílias testadas (EDP, Wiener e rede neural):** melhorar SNR e STOI não implica reduzir o WER de um reconhecedor moderno e robusto como o Whisper.
5. **A EDP não acrescenta nada depois da U-Net:** `unet+pm_sel` é igual ou levemente pior que `unet` em todas as condições.

---

## 14. O erro do DC: o que foi, como apareceu, como foi corrigido

**O que é DC.** É a média do sinal (frequência zero), um nível constante que não se ouve. Um áudio gravado com um circuito mal ajustado pode ter DC grande. Aqui, o DC/desvio (média absoluta dividida pelo desvio padrão) ficou entre 1,2 e 6,6 nos locutores medidos. Isso significa que de 60% a 98% da "energia" do sinal era só DC.

**Como apareceu.** No primeiro `avaliar.py`, o SNR da U-Net ficava preso em ~1,4 dB e o do Wiener em ~3 a 4 dB, qualquer que fosse o SNR de entrada (0 a 20 dB). Um realce de verdade não se comporta assim, e a validação do treino tinha dado +10 dB. O diagnóstico (`treino/diagnostico.py`) mostrou DC/desvio de 2,9 a 4,8: a U-Net e o Wiener removem o DC (corretamente), e a métrica de SNR, que compara com uma referência que contém o DC, contava essa remoção como erro. A conta bate: com DC/desvio = 4,8, remover o DC deixa um SNR de ~0,2 dB, e a U-Net deu +0,3 dB.

**O efeito sobre o ruído sintético (o mais grave).** `adicionar_ruido_tipo` calculava a potência do sinal como `mean(y²)`, que inclui o DC. A potência do ruído saía muito maior que a pedida. Medição com 12 locutores, pedindo SNR de 10 dB:

| Situação | SNR real da fala |
|---|---|
| Antes da correção (12 locutores) | de −6,5 a +6,1 dB (11 de 12 entre −6,5 e +0,7 dB) |
| Depois da correção | 10,0 dB em todos |

**O que foi afetado e o que não foi**

| Item | Afetado? |
|---|---|
| ASR da Parte 1 (seções 6.2, 6.3, 6.5) | **Sim.** Rótulos de SNR errados; refeito na seção 17 |
| Comparação entre frentes dentro de uma rodada | Não: todas receberam o mesmo ruído |
| Experimento A (qualidade, 35 locutores) | Provavelmente não: no benchmark do `run_lab.py`, o Wiener deu +14 dB no locutor F004 (DC/desvio ≈ 5), o que só é possível com o DC já removido em `carregar_audio` (inferência, não verificada no código) |
| Treino da U-Net | Praticamente não: o sinal limpo já tinha o DC removido. Ressalva: o ruído babble do treino pode ter carregado algum DC dos áudios do CETUC (corrigido em `dados.py` para os próximos treinos) |

**Teste de consistência.** O WER sem realce com ruído branco a 10 dB (SNR real) saiu 12,6%, quase igual ao do "20 dB" da rodada antiga (12,5%): o rótulo antigo estava ~10 dB acima do real, como a medição indicou. Além disso, só remover o DC reduziu o WER sem ruído de 8,9% para 8,5%.

**Correções aplicadas**

1. `adicionar_ruido_tipo`: potência do sinal calculada com `np.var` (sem DC).
2. `benchmark_asr.py`: remove o DC ao ler cada fala e o conjunto do babble.
3. `avaliar.py` e `modelo.realcar`: removem o DC da referência e da entrada.
4. `treino/dados.py`: remove o DC dos trechos do CETUC.

**Lição.** Sempre conferir o SNR *real* do ruído adicionado e inspecionar métricas que ficam "constantes" quando deveriam variar. Um resultado suspeito de ser bom ou ruim demais é um convite a medir, não a interpretar.

---

## 15. A U-Net de realce

### 15.1 Ideia
É a mesma ideia do projeto (editar a magnitude e manter a fase), mas com a **máscara aprendida** em vez de calculada por Wiener:

```
áudio ruidoso --STFT--> |X|, fase
entrada da rede = (20·log10(|X|+1e-5) + 20) / 40         (espectrograma em dB, normalizado)
máscara = U-Net(entrada)        (valores em 0..1, mesmo tamanho do espectrograma)
|X̂| = máscara · |X|             áudio = ISTFT(|X̂| · e^{j·fase ruidosa})
```

### 15.2 Arquitetura (`treino/modelo.py`)
U-Net 2D com 4 níveis: cada nível tem duas convoluções 3×3 com BatchNorm e LeakyReLU; o caminho de descida usa max-pooling e o de subida usa convolução transposta, com **conexões de atalho** (skip connections) que levam os detalhes finos da entrada à saída. A última camada é uma convolução 1×1 com sigmoide. **1.942.289 parâmetros**, 16 canais na primeira camada (dobrando a cada nível).

### 15.3 Dados (`treino/dados.py`)
- **CETUC** (FalaBrasil, MIT): 8 locutores de treino (8.000 falas, 11,3 h) e 3 de validação (3.000 falas, 4,6 h). **Teste final: LapsBM**, outro dataset, com locutores que a rede nunca viu.
- **Ruído gerado na hora, a cada passo**: branco, rosa, marrom e babble (soma de 6 trechos de fala de outros locutores), com SNR sorteado entre −5 e 20 dB. A rede nunca vê duas vezes o mesmo ruído, o que reduz o risco de decorar.
- Trechos de 3 s, com início aleatório; DC removido; entrada e alvo normalizados pelo mesmo fator.
- Validação com conjunto **fixo** (400 trechos, mesma semente), para as épocas serem comparáveis.

### 15.4 Perda e treino (`treino/treinar.py`)
- **Perda:** erro absoluto em dB entre o espectrograma estimado e o limpo (dividido por 40) **mais** erro absoluto na magnitude comprimida (potência 0,3). A primeira domina onde o sinal é fraco; a segunda, onde é forte.
- **Otimizador:** AdamW (taxa 1e-3, decaimento de pesos 1e-4) com **OneCycle** (aquecimento de 10% e decaimento), corte de gradiente em 5,0, lote de 16, 4.000 trechos por época (250 passos), 30 épocas, semente fixa. Precisão mista só na GPU.
- **Parada antecipada** (paciência 8): não disparou, as 30 épocas rodaram. O **melhor modelo** (menor perda de validação) é o da época 25.
- **Hardware:** CPU (Ryzen), ~8 a 9 min por época, ~4,5 h de treino efetivo. O relógio total de 917 min inclui uma suspensão do computador.
- **Registro:** MLflow (parâmetros, métricas por época, modelo).

### 15.5 Curva de treino

| Época | Perda treino | Perda validação | SNR validação (entrada → saída) | Ganho |
|---|---|---|---|---|
| 1 | 0,430 | 0,400 | 7,2 → 12,9 dB | +5,6 dB |
| 5 | 0,245 | 0,255 | 7,2 → 16,7 dB | +9,5 dB |
| 10 | 0,231 | 0,243 | 7,2 → 16,9 dB | +9,7 dB |
| 20 | 0,219 | 0,238 | 7,2 → 17,0 dB | +9,8 dB |
| 25 (melhor) | 0,215 | 0,236 | 7,2 → 17,2 dB | +10,0 dB |
| 30 | 0,212 | 0,238 | 7,2 → 17,2 dB | +10,0 dB |

**Leitura:** a perda de treino e a de validação ficam próximas (diferença de ~0,03 no final), então **não há overfitting relevante**. A curva se estabiliza por volta da época 20: mais épocas, com os mesmos dados e o mesmo modelo, trariam pouco. O que provavelmente ajudaria mais é mais dados (mais locutores), um modelo maior ou uma perda de forma de onda (SI-SDR).

---

## 16. Avaliação de qualidade do sinal no LapsBM (sem DC)

35 locutores (1 fala por locutor), 4 s, ruído branco e babble, SNR de entrada 0/5/10/20 dB, DC removido, mesma semente para todos os métodos. Média entre as falas. Arquivos: `resultados_unet/`.

### 16.1 Ruído branco (SNR em dB / STOI)

| Método | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|
| ruidoso | 0,0 / 0,572 | 5,0 / 0,675 | 10,0 / 0,771 | 20,0 / 0,910 |
| wiener | 6,3 / 0,602 | 10,8 / 0,701 | 15,0 / 0,787 | 21,9 / 0,905 |
| wiener+pm_sel | 8,7 / 0,612 | 12,4 / 0,708 | 15,8 / 0,790 | 21,2 / 0,900 |
| **unet** | **10,4 / 0,704** | **12,6 / 0,765** | 14,5 / **0,817** | 17,3 / 0,895 |
| unet+pm_sel | 10,4 / 0,703 | 12,5 / 0,763 | 14,4 / 0,813 | 17,2 / 0,887 |

### 16.2 Ruído babble (SNR em dB / STOI)

| Método | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|
| ruidoso | 0,0 / 0,531 | 5,0 / 0,657 | 10,0 / 0,757 | 20,0 / 0,892 |
| wiener | 1,3 / 0,534 | 6,3 / 0,660 | 11,2 / 0,754 | 20,0 / 0,875 |
| wiener+pm_sel | 1,4 / 0,534 | 6,2 / 0,659 | 11,0 / 0,751 | 19,3 / 0,867 |
| **unet** | **4,5 / 0,600** | **8,8 / 0,733** | **12,2 / 0,805** | 16,6 / 0,880 |
| unet+pm_sel | 4,5 / 0,599 | 8,8 / 0,732 | 12,1 / 0,802 | 16,4 / 0,874 |

### 16.3 Diferenças pareadas (média, IC 95% por bootstrap; as mesmas falas em todos os métodos)

| Comparação | Ruído | 0 dB | 5 dB | 10 dB | 20 dB |
|---|---|---|---|---|---|
| unet − wiener+pm_sel, SNR | branco | **+1,7** | +0,2 (sem dif.) | **−1,2** | **−3,8** |
| unet − wiener+pm_sel, STOI | branco | **+0,092** | **+0,056** | **+0,027** | −0,005 |
| unet − wiener+pm_sel, SNR | babble | **+3,1** | **+2,6** | **+1,2** | **−2,7** |
| unet − wiener+pm_sel, STOI | babble | **+0,066** | **+0,074** | **+0,054** | **+0,013** |
| unet − ruidoso, SNR | branco | **+10,4** | **+7,6** | **+4,5** | **−2,7** |
| unet − ruidoso, SNR | babble | **+4,5** | **+3,8** | **+2,2** | **−3,4** |

(Negrito = intervalo de confiança não inclui zero. Sinal positivo = a U-Net é melhor.)

### 16.4 Leitura

1. **Ruído forte (0 a 5 dB): a U-Net é claramente a melhor**, em SNR e em STOI. Com babble, onde o Wiener praticamente não ajuda (o ruído não é estacionário), a U-Net ganha +4,5 dB a 0 dB.
2. **Ruído médio (10 dB):** com babble, a U-Net ainda ganha. Com branco, o `wiener+pm_sel` ganha em SNR (+1,2 dB), mas a U-Net mantém mais STOI (+0,027).
3. **Ruído fraco (20 dB): processar piora.** A U-Net perde de 2,7 a 3,4 dB de SNR para o sinal ruidoso e fica atrás do Wiener. Com pouco ruído, qualquer máscara que remove um pouco de fala custa mais do que ganha.
4. **`unet+pm_sel` é igual ou levemente pior que `unet`** em todas as condições (≤0,14 dB de SNR e ≤0,008 de STOI; significativo, mas irrelevante). A EDP não acrescenta nada depois de uma máscara aprendida.
5. **Por que a U-Net supera o Wiener com babble:** o Wiener estima o ruído nas pausas e assume que ele é estacionário. A U-Net aprende o padrão de "o que é fala" no espectrograma, e não precisa dessa hipótese.

---

## 17. ASR corrigido (SNR real): Parte 1 refeita e U-Net

`faster-whisper` `small`, 70 falas do LapsBM (2 por locutor, 35 locutores), CPU, int8, DC removido, SNR **real** de 10/5/0 dB. IC 95% por bootstrap pareado. Arquivos: `resultados_asr/v2_*` e `v3_*`.

### 17.1 Ruído branco (WER %)

| Frente | limpo | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| nenhum | 8,5 | 12,6 | 21,7 | 38,4 |
| wiener | 7,4 | 13,9 | 22,4 | 40,5 |
| perona_malik | 8,4 | 14,1 (PIORA) | 23,3 | 39,6 |
| wiener+pm | 8,0 | 13,4 | 22,9 | 40,5 |
| pm_seletivo | 8,2 | 13,6 | 21,7 | 37,4 |
| wiener+pm_sel | 8,2 | 13,6 | 22,5 | 39,4 |
| **unet** | **7,4 (MELHORA)** | 13,9 | 23,2 | 43,8 |
| unet+pm_sel | 7,8 | 14,7 | 23,5 | 44,9 |

### 17.2 Ruído babble (WER %)

| Frente | limpo | 10 dB | 5 dB | 0 dB |
|---|---|---|---|---|
| nenhum | 8,5 | 10,4 | 16,6 | 53,8 |
| wiener | 7,4 | 11,7 | 20,0 (PIORA) | 66,1 (PIORA) |
| perona_malik | 8,4 | 12,2 (PIORA) | 17,4 | 53,8 |
| wiener+pm | 8,0 | 11,2 | 20,7 | 63,8 (PIORA) |
| pm_seletivo | 8,2 | 11,4 | 18,5 | 57,9 |
| wiener+pm_sel | 8,2 | 10,7 | 20,4 | 60,8 a 61,2 (PIORA) |
| unet | **7,4 (MELHORA)** | 12,2 | 22,6 (PIORA) | **70,1 (PIORA)** |
| unet+pm_sel | 7,8 | 12,1 | 21,9 (PIORA) | **72,6 (PIORA)** |

"PIORA"/"MELHORA" = intervalo de confiança 95% da diferença para `nenhum` não inclui zero. As frentes presentes nas duas rodadas (`nenhum`, `wiener+pm_sel`) reproduziram os valores quase exatamente (a variação máxima foi de 0,4 ponto, em `wiener+pm_sel` com babble a 0 dB: 61,2 contra 60,8).

### 17.3 Leitura

1. **Nenhuma técnica reduz o WER com ruído, com significância.** Os resultados mais próximos de "ajudar" são o `pm_seletivo` (neutro em todas as condições) e a U-Net no ruído branco a 10 dB (+1,2, sem diferença clara).
2. **O babble é onde o realce faz mais estrago, e a U-Net é o pior caso:** +16,3 pontos a 0 dB (53,8% para 70,1%), mesmo sendo a melhor em SNR e STOI nesse mesmo ruído. O Wiener também pior (+12,3).
3. **O único "MELHORA"** (a U-Net no WER sem ruído, −1,1 [−2,3, −0,1]) é um efeito pequeno, com intervalo que quase toca o zero, entre dezenas de comparações. Tratar como indício, não como resultado. É plausível: sem ruído adicionado, a máscara só remove o ruído de fundo próprio das gravações do LapsBM.
4. **Contradição aparente com a seção 16, e por que ela é real.** SNR e STOI medem a distância ao sinal limpo. O Whisper não "ouve" assim. Hipóteses, não testadas: (a) o Whisper foi treinado com muito áudio ruidoso e já separa fala de babble, melhor do que uma máscara 0 a 1 que só atenua; (b) uma máscara por bin gera artefatos (fala "esburacada", ruído musical) que o modelo não viu no treino; (c) com babble, a fala-alvo e o ruído têm o mesmo espectro, então a máscara apaga também partes da fala, e uma frase com palavra faltando custa mais ao reconhecedor do que um ruído constante ao fundo.

---

## 18. Consolidação: o que funcionou, o que não funcionou e por quê

| Objetivo | Melhor método | Observação |
|---|---|---|
| Qualidade do sinal, ruído forte (0 a 5 dB) | **U-Net** | +10,4 dB (branco 0 dB), +4,5 dB (babble 0 dB), STOI +0,09 a +0,13 |
| Qualidade do sinal, ruído médio (10 dB) | U-Net (babble), `wiener+pm_sel` (branco) | diferenças pequenas |
| Qualidade do sinal, ruído fraco (20 dB) | **Não processar** (ou Wiener) | todo realce degrada um pouco |
| Reduzir o WER do Whisper | **Não processar** | nenhuma frente ajuda; várias pioram |
| EDP sozinha (calor, Perona-Malik) | — | borra; ganho de LSD sem ganho de som |
| EDP como acabamento seletivo (`pm_seletivo`) | melhor EDP | só depois de um Wiener; depois da U-Net, não ajuda |

**Conclusão geral.** Para o objetivo "o ouvido ou uma métrica de sinal prefere o resultado", uma máscara aprendida (U-Net) supera as técnicas clássicas, e a EDP é dispensável. Para o objetivo "o reconhecimento de fala melhora", nenhum pré-processamento testado ajudou, o que aponta que o ganho possível está dentro do próprio reconhecedor (por exemplo, ajuste fino do Whisper com áudio ruidoso), e não numa etapa anterior. A EDP funciona como regularizador de imagem do espectrograma, que é o seu papel natural, e não como removedor de ruído.

---

## 19. Limitações da Parte 2

1. **O ruído do treino e o do teste são da mesma família** (branco, rosa, marrom e babble sintéticos, gerados pelo mesmo código). Os locutores e o dataset do teste são diferentes, mas o ruído não. O ganho da U-Net em ruído **real** (rua, ambiente, microfone ruim) deve ser menor e **não foi medido**.
2. **Um único treino, uma semente**, sem barras de erro entre treinos.
3. **Pouco dado de treino** (11,3 h, 8 locutores) e um modelo pequeno. A curva estabilizou, mas isso vale para esta configuração.
4. **O babble do teste é feito de falas do próprio LapsBM** de outros locutores, o que pode ser mais fácil do que fala sobreposta real.
5. **STOI e SNR são métricas de proximidade com o limpo**, sem peso para o que importa ao reconhecedor. Não há medida de qualidade subjetiva (teste de escuta).
6. **ASR:** 70 falas, intervalos largos (por exemplo, ±6 pontos), um modelo (`small`), dezenas de comparações por tabela, sem correção para comparações múltiplas.
7. **O `unet_v1` pode ter visto algum DC no babble do treino** (seção 14); a influência é pequena e não foi medida.
8. **A máscara usa a fase ruidosa.** Com ruído forte, isso limita o resultado de qualquer método de magnitude.

---

## 20. Próximos passos sugeridos

1. **Ruído real:** gravar a própria voz em ambiente ruim (`gravar.py`) e comparar `nenhum`, `wiener+pm_sel` e `unet` por WER e de ouvido. É o teste que mais falta.
2. **Treinar melhor, com a RTX 4050:** U-Net maior (`--base 32`), mais locutores, mais épocas, e perda de forma de onda (SI-SDR). Comparar com `unet_v1` pelo MLflow.
3. **Baseline neural pré-treinado** (DeepFilterNet, RNNoise ou Demucs) como referência de estado da arte.
4. **Treinar a U-Net com o ASR no loop** (por exemplo, com perda que penaliza o erro do Whisper), já que a perda de sinal não se traduz em WER.
5. **Ajuste fino do Whisper** com áudio ruidoso, para atacar o ponto onde o ganho parece estar.
6. **Teste de escuta** (com colegas) das saídas `unet` e `wiener+pm_sel`.
7. **Importar os CSVs de ASR para o MLflow**, para comparar tudo no mesmo painel.

---

## 21. Arquivos novos e como reproduzir

| Arquivo | Para quê |
|---|---|
| `treino/dados.py` | Dataset de treino com ruído gerado na hora |
| `treino/modelo.py` | U-Net, STFT/ISTFT e `realcar()` |
| `treino/treinar.py` | Treino (CPU ou GPU), validação, parada antecipada, MLflow |
| `treino/avaliar.py` | Comparação no LapsBM (SNR, LSD, STOI, IC pareado) |
| `treino/diagnostico.py` | Diagnóstico de escala, atraso e DC |
| `registro.py` | Camada fina do MLflow |
| `modelos/unet_v1/best.pt` | Modelo treinado (época 25, ~8 MB) |

```powershell
python preparar_dados.py --dataset cetuc --splits train --max-locutores 8
python preparar_dados.py --dataset cetuc --splits dev --max-locutores 3
python treino/treinar.py --nome unet_v1 --epocas 30 --mlflow
python treino/avaliar.py --modelo modelos/unet_v1/best.pt --mlflow
python benchmark_asr.py --modelo small --por-locutor 2 --snrs 10 5 0 --ruido branco --frentes nenhum wiener+pm_sel unet unet+pm_sel --saida resultados_asr/v3_branco
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## 22. Referências adicionais

- Ronneberger, Fischer & Brox (2015), *U-Net: Convolutional Networks for Biomedical Image Segmentation*.
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* (AdamW).
- Smith & Topin (2019), *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates* (OneCycle).
- FalaBrasil, CETUC e LapsBM: huggingface.co/falabrasil.
