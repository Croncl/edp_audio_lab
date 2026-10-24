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


def carregar_modelo(nome, device, threads):
    """auto: usa CUDA se existir e funcionar, senao CPU. Retorna (modelo, dispositivo_usado)."""
    import numpy as np
    from faster_whisper import WhisperModel
    if device in ("auto", "cuda"):
        try:
            import ctranslate2
            if ctranslate2.get_cuda_device_count() > 0:
                m = WhisperModel(nome, device="cuda", compute_type="float16")
                list(m.transcribe(np.zeros(16000, dtype="float32"), language="pt")[0])  # teste real
                return m, "cuda"
            motivo = "nenhuma GPU CUDA encontrada"
        except Exception as e:
            motivo = f"{type(e).__name__}: {e}"
        print(f"CUDA indisponivel ({motivo}); usando CPU.")
    return WhisperModel(nome, device="cpu", compute_type="int8", cpu_threads=threads), "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--modelo", default="tiny", help="tiny | base | small | medium")
    ap.add_argument("--threads", type=int, default=max(4, (os.cpu_count() or 8) // 2),
                    help="threads de CPU (padrão: metade dos núcleos lógicos)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = ap.parse_args()

    from faster_whisper import WhisperModel

    # Lemos o áudio nós mesmos (mono, 16 kHz, float32) e entregamos o array ao Whisper.
    # Isso evita o decodificador PyAV do faster-whisper, que quebra com algumas versões do pacote `av`.
    audio, sr = librosa.load(a.audio, sr=16000, mono=True)
    dur = len(audio) / sr
    print(f"Áudio: {a.audio}  ({dur:.1f} s)   modelo: {a.modelo}   threads: {a.threads}")

    t0 = time.perf_counter()
    modelo, disp = carregar_modelo(a.modelo, a.device, a.threads)
    print(f"Dispositivo: {disp}")
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