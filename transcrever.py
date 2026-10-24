"""
transcrever.py — transcreve um áudio com faster-whisper, 100% local (CPU) e mede a velocidade.

    pip install faster-whisper
    python transcrever.py meus_audios/fala1.wav
    python transcrever.py meus_audios/fala1.wav --modelo base

Na primeira vez com cada modelo, o faster-whisper baixa os pesos do Hugging Face (precisa de internet
uma única vez); depois roda offline a partir do cache.
"""
import argparse
import os
import time

import librosa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--modelo", default="tiny", help="tiny | base | small | medium")
    ap.add_argument("--threads", type=int, default=max(4, (os.cpu_count() or 8) // 2),
                    help="threads de CPU (padrão: metade dos núcleos lógicos)")
    a = ap.parse_args()

    from faster_whisper import WhisperModel

    # Lemos o áudio nós mesmos (mono, 16 kHz, float32) e entregamos o array ao Whisper.
    # Isso evita o decodificador PyAV do faster-whisper, que quebra com algumas versões do pacote `av`.
    audio, sr = librosa.load(a.audio, sr=16000, mono=True)
    dur = len(audio) / sr
    print(f"Áudio: {a.audio}  ({dur:.1f} s)   modelo: {a.modelo}   threads: {a.threads}")

    t0 = time.perf_counter()
    modelo = WhisperModel(a.modelo, device="cpu", compute_type="int8", cpu_threads=a.threads)
    t_carga = time.perf_counter() - t0

    t0 = time.perf_counter()
    segmentos, _ = modelo.transcribe(audio, language="pt", beam_size=5)
    texto = " ".join(s.text.strip() for s in segmentos)  # o gerador só executa aqui
    t_trans = time.perf_counter() - t0

    print(f"\nTexto: {texto}\n")
    print(f"Carregar modelo: {t_carga:.1f} s | Transcrever: {t_trans:.1f} s | "
          f"fator de tempo real: {t_trans / dur:.2f}x (abaixo de 1 = mais rápido que o áudio)")


if __name__ == "__main__":
    main()