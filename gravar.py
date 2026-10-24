"""
gravar.py — grava sua voz pelo microfone (100% local) em um .wav mono 16 kHz.

    pip install sounddevice
    python gravar.py                       # 5 s -> minha_fala.wav
    python gravar.py --seg 8 --saida meus_audios/fala1.wav
    python gravar.py --dispositivos        # lista os microfones

Dica: grave num ambiente silencioso (a referência "limpa" é a gravação; o ruído é adicionado depois,
de forma controlada) e diga uma frase completa em português.
"""
import argparse
from pathlib import Path

import soundfile as sf

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=float, default=5.0, help="duração em segundos")
    ap.add_argument("--saida", default="minha_fala.wav")
    ap.add_argument("--dispositivo", type=int, default=None, help="índice do microfone")
    ap.add_argument("--dispositivos", action="store_true", help="lista dispositivos e sai")
    a = ap.parse_args()
    try:
        import sounddevice as sd
    except Exception as e:  # PortAudio ausente ou pacote não instalado
        raise SystemExit(f"sounddevice indisponível ({e}). Instale com: pip install sounddevice")
    if a.dispositivos:
        print(sd.query_devices())
        return
    input(f"Pressione ENTER e fale por {a.seg:g} s...")
    audio = sd.rec(int(a.seg * SR), samplerate=SR, channels=1, dtype="float32", device=a.dispositivo)
    sd.wait()
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    sf.write(a.saida, audio, SR)
    print(f"Salvo em {Path(a.saida).resolve()}")


if __name__ == "__main__":
    main()
