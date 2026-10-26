"""
modelo.py: U-Net 2D que prevê uma MÁSCARA (0 a 1) sobre o espectrograma de magnitude ruidoso.

    áudio ruidoso --STFT--> |X|, fase
    feature = (20·log10(|X|+1e-5) + 20) / 40          (dB normalizado)
    máscara = U-Net(feature)                           (sigmoide, mesmo formato)
    |X̂| = máscara · |X|     áudio = ISTFT(|X̂| · e^{j·fase ruidosa})

É a mesma ideia do projeto (editar a magnitude e manter a fase), com a máscara aprendida
em vez de calculada por Wiener.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

SR, N_FFT, HOP = 16000, 512, 128
EPS = 1e-5


def stft(y, janela):
    return torch.stft(y, N_FFT, HOP, window=janela, return_complex=True)


def istft(X, janela, n):
    return torch.istft(X, N_FFT, HOP, window=janela, length=n)


def feature(mag):
    return (20 * torch.log10(mag + EPS) + 20) / 40


class Bloco(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.1, True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.LeakyReLU(0.1, True))

    def forward(self, x):
        return self.f(x)


class UNet(nn.Module):
    def __init__(self, base=16, niveis=4):
        super().__init__()
        self.niveis = niveis
        ch = [base * 2 ** i for i in range(niveis + 1)]
        self.enc = nn.ModuleList(Bloco(1 if i == 0 else ch[i - 1], ch[i]) for i in range(niveis))
        self.fundo = Bloco(ch[niveis - 1], ch[niveis])
        self.ups = nn.ModuleList(nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(niveis)))
        self.dec = nn.ModuleList(Bloco(ch[i] * 2, ch[i]) for i in reversed(range(niveis)))
        self.saida = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):                       # x: (B, 1, F, T)
        F, T = x.shape[-2:]
        m = 2 ** self.niveis
        x = nnf.pad(x, (0, (-T) % m, 0, (-F) % m), mode="replicate")
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = nnf.max_pool2d(x, 2)
        x = self.fundo(x)
        for up, dec, s in zip(self.ups, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], dim=1))
        return torch.sigmoid(self.saida(x))[..., :F, :T]


def contar_parametros(modelo):
    return sum(p.numel() for p in modelo.parameters())


def carregar(caminho, device="cpu"):
    ck = torch.load(caminho, map_location=device, weights_only=False)
    m = UNet(**ck["config"]).to(device)
    m.load_state_dict(ck["modelo"])
    return m.eval(), ck


@torch.no_grad()
def realcar(modelo, y, device="cpu"):
    """numpy float32 (16 kHz) -> numpy float32 realçado, no mesmo volume da entrada."""
    modelo.eval()
    janela = torch.hann_window(N_FFT, device=device)
    y = np.asarray(y, dtype=np.float32)
    g = 0.9 / (np.max(np.abs(y)) + 1e-9)
    yt = torch.from_numpy(y * g).to(device)[None]
    X = stft(yt, janela)
    mask = modelo(feature(X.abs())[:, None])[:, 0]
    out = istft(mask * X, janela, yt.shape[-1])[0].cpu().numpy() / g
    return out.astype(np.float32)
