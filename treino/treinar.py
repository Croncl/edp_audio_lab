"""
treinar.py: treina a U-Net de realce.

    python treino/treinar.py --mlflow
    python treino/treinar.py --epocas 40 --base 24 --nome unet_v2 --mlflow
    python treino/treinar.py --teste-rapido          # 1 minuto, só para ver se tudo roda

Usa GPU (CUDA) se existir, senão CPU. Treina com ruído gerado na hora; valida com um conjunto
FIXO de trechos (locutores de `dev`). Salva o melhor modelo (menor perda de validação).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from registro import Registro  # noqa: E402
from dados import DadosRuido  # noqa: E402
from modelo import UNet, contar_parametros, feature, istft, stft  # noqa: E402


def passo(modelo, ruidoso, limpo, janela, amp):
    Xn, Xc = stft(ruidoso, janela), stft(limpo, janela)
    mn, mc = Xn.abs(), Xc.abs()
    ft = feature(mn)[:, None]
    with torch.autocast(device_type=ruidoso.device.type, enabled=amp):
        mask = modelo(ft)[:, 0]
    mask = mask.float()
    me = mask * mn
    dbe, dbc = 20 * torch.log10(me + 1e-5), 20 * torch.log10(mc + 1e-5)
    p_db = (dbe - dbc).abs().mean() / 40
    p_mag = ((me + 1e-6) ** 0.3 - (mc + 1e-6) ** 0.3).abs().mean()
    return p_db + p_mag, p_db, p_mag, mask, Xn


def snr_db(ref, y):
    e = ref - y
    return 10 * torch.log10((ref ** 2).sum(-1) / ((e ** 2).sum(-1) + 1e-12) + 1e-12)


@torch.no_grad()
def validar(modelo, loader, janela, dev, amp):
    modelo.eval()
    tot, n, s_in, s_out = 0.0, 0, 0.0, 0.0
    for ruidoso, limpo, _ in loader:
        ruidoso, limpo = ruidoso.to(dev), limpo.to(dev)
        l, _, _, mask, Xn = passo(modelo, ruidoso, limpo, janela, amp)
        y = istft(mask * Xn, janela, ruidoso.shape[-1])
        b = ruidoso.shape[0]
        tot += l.item() * b
        s_in += snr_db(limpo, ruidoso).sum().item()
        s_out += snr_db(limpo, y).sum().item()
        n += b
    return tot / n, s_in / n, s_out / n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifesto", default="data/manifests/cetuc.csv")
    ap.add_argument("--nome", default="unet_v1")
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--amostras-epoca", type=int, default=4000, help="trechos de treino por época")
    ap.add_argument("--amostras-val", type=int, default=400)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--duracao", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base", type=int, default=16, help="canais da 1a camada (16 -> ~1,9 M parametros)")
    ap.add_argument("--niveis", type=int, default=4)
    ap.add_argument("--paciencia", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--experimento", default="treino_unet")
    ap.add_argument("--saida", default="modelos")
    ap.add_argument("--teste-rapido", action="store_true")
    a = ap.parse_args()
    if a.teste_rapido:
        a.epocas, a.amostras_epoca, a.amostras_val, a.workers, a.batch = 2, 64, 32, 0, 8

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    print(f"Dispositivo: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if amp else ""))

    treino = DadosRuido(a.manifesto, "train", a.duracao, a.amostras_epoca, seed=a.seed)
    val = DadosRuido(a.manifesto, "dev", a.duracao, a.amostras_val, fixo=True, seed=a.seed + 1)
    lt = DataLoader(treino, a.batch, shuffle=False, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0, pin_memory=amp)
    lv = DataLoader(val, a.batch, shuffle=False, num_workers=min(a.workers, 2))

    cfg = dict(base=a.base, niveis=a.niveis)
    modelo = UNet(**cfg).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epocas * len(lt), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=amp)
    janela = torch.hann_window(512, device=dev)
    pasta = Path(a.saida) / a.nome
    pasta.mkdir(parents=True, exist_ok=True)

    params = dict(vars(a), parametros_modelo=contar_parametros(modelo), dispositivo=dev)
    reg = Registro(a.mlflow, a.experimento, a.nome, params)
    print(f"Parâmetros do modelo: {contar_parametros(modelo):,} | passos por época: {len(lt)}")

    melhor, sem_melhora, t0 = 1e9, 0, time.perf_counter()
    for ep in range(1, a.epocas + 1):
        modelo.train()
        acc = np.zeros(3)
        for ruidoso, limpo, _ in lt:
            ruidoso, limpo = ruidoso.to(dev, non_blocking=True), limpo.to(dev, non_blocking=True)
            l, pdb, pmag, _, _ = passo(modelo, ruidoso, limpo, janela, amp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            acc += [l.item(), pdb.item(), pmag.item()]
        acc /= len(lt)
        vl, s_in, s_out = validar(modelo, lv, janela, dev, amp)
        m = dict(train_loss=acc[0], train_loss_db=acc[1], train_loss_mag=acc[2], val_loss=vl,
                 val_snr_in=s_in, val_snr_out=s_out, val_ganho_snr=s_out - s_in, lr=opt.param_groups[0]["lr"])
        reg.metricas(m, ep)
        marca = ""
        if vl < melhor:
            melhor, sem_melhora, marca = vl, 0, " *"
            torch.save(dict(modelo=modelo.state_dict(), config=cfg, epoca=ep, val_loss=vl), pasta / "best.pt")
        else:
            sem_melhora += 1
        print(f"época {ep:3d}/{a.epocas} | treino {acc[0]:.4f} | val {vl:.4f} | "
              f"SNR {s_in:+.1f} -> {s_out:+.1f} dB (ganho {s_out - s_in:+.1f}) | {(time.perf_counter() - t0) / 60:.1f} min{marca}")
        if sem_melhora >= a.paciencia:
            print("Parada antecipada: sem melhora na validação.")
            break
    torch.save(dict(modelo=modelo.state_dict(), config=cfg, epoca=ep, val_loss=vl), pasta / "last.pt")
    reg.metricas(dict(melhor_val_loss=melhor, tempo_treino_min=(time.perf_counter() - t0) / 60))
    reg.arquivo(pasta / "best.pt", "modelo")
    reg.fim()
    print(f"\nMelhor val_loss: {melhor:.4f}. Modelo em {(pasta / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()
