import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import time

def parallel_lcse(log_input, log_coeff):

    t, b, d = log_input.shape

    t_log_coeff = torch.arange(t, device=log_coeff.device)[:, None] * log_coeff[None, :]
    t_log_coeff = t_log_coeff.unsqueeze(1)

    return t_log_coeff + torch.logcumsumexp(log_input - t_log_coeff, dim=0)


def conv(_input, log_coeff):

    t, b, d = _input.shape

    t_log_coeff = torch.arange(t - 1, -1, -1, device=log_coeff.device)[:, None] * log_coeff[None, :]
    kernel_transpose = torch.diag_embed(t_log_coeff.exp())
    kernel = kernel_transpose.permute(1, 2, 0)

    # T B D -> B D T
    input_pad = F.pad(_input.permute(1, 2, 0), (t-1, 0, 0, 0, 0, 0))
    # B D T -> T B D
    return F.conv1d(input_pad, kernel.to(dtype=torch.complex64)).permute(2, 0, 1)


class LRU(nn.Module):

    def __init__(self, dim, r_min=0.5, r_max=0.95, max_phase=6.28):

        super().__init__()
        u1 = torch.rand(size=(dim,), dtype=torch.float32)
        u2 = torch.rand(size=(dim,), dtype=torch.float32)

        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))

        self.b_linear = nn.Linear(dim, dim, bias=False)
        self.b_linear.weight = nn.Parameter((torch.randn((dim, dim)) + 1j * torch.randn((dim, dim))) / np.sqrt(2 * dim))
        self.c_linear = nn.Linear(dim, dim, bias=False)
        self.c_linear.weight = nn.Parameter((torch.randn((dim, dim)) + 1j * torch.randn((dim, dim))) / np.sqrt(dim))
        self.d = nn.Parameter(torch.randn((dim,)))

        _lambda = torch.exp(-torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)).detach()
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(1 - torch.abs(_lambda) ** 2)))

        self.forward = self.lcse

    def lcse(self, x: torch.Tensor, h: torch.Tensor = None, eps: float = 1e-10):

        log_lambda = -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
        x_complex = x.to(dtype=torch.complex64)
        bx = self.b_linear(x_complex) + eps

        if h is not None:
            x_in = torch.cat([h.log(), self.gamma_log + bx.log()], dim=0)
            ht = parallel_lcse(x_in, log_lambda)[1:].exp()
        else:
            x_in = self.gamma_log + bx.log()
            ht = parallel_lcse(x_in, log_lambda).exp()

        y = self.c_linear(ht.to(dtype=torch.complex64)).real + self.d * x

        return y, ht[-1]

    def conv(self, x: torch.Tensor, h: torch.Tensor = None):

        if h is None:
            h = torch.zeros_like(x[0])
        log_lambda = -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)

        x_complex = x.to(dtype=torch.complex64)
        bx = self.gamma_log.exp() * self.b_linear(x_complex)
        ht = conv(bx, log_lambda)
        y = self.c_linear(ht.to(dtype=torch.complex64)).real + self.d * x

        return y, ht[-1]

    def seq(self, x: torch.Tensor, h: torch.Tensor = None):

        if h is None:
            h = torch.zeros_like(x[0])

        log_lambda = -torch.exp(self.nu_log) + 1j * torch.exp(self.theta_log)
        x_complex = x.to(dtype=torch.complex64)
        bx = self.gamma_log.exp() * self.b_linear(x_complex)

        ht = []

        _lambda = log_lambda.exp()
        for t in range(x.size(0)):
            ht.append(h * _lambda + bx[t])
            h = ht[-1]

        ht = torch.stack(ht)

        y = self.c_linear(ht.to(dtype=torch.complex64)).real + self.d * x

        return y, ht[-1]


class LRUBlock(nn.Module):

    def __init__(self, dim: int):

        super().__init__()
        self.prenorm = nn.LayerNorm(dim)
        self.rnn = LRU(dim)
        self.linear = nn.Linear(dim, dim * 2)

    def forward(self, x, h: torch.Tensor = None):

        z = self.prenorm(x)
        z, h = self.rnn(z)
        z = F.gelu(z)
        z1, z2 = self.linear(z).chunk(2, dim=-1)
        z = z1 * torch.sigmoid(z2)
        return z + x, h


if __name__ == '__main__':

    torch.set_default_device('cuda')

    dim = 5
    x = torch.randn((10, 10, dim), dtype=torch.float32)

    layer = LRU(dim)
    y_lcse = layer.lcse(x)
    y_conv = layer.conv(x)
    y_seq = layer.seq(x)
    print('LCSE/SEQ output allclose:', torch.allclose(y_lcse[0], y_seq[0], atol=1e-4))
    print('CONV/SEQ output allclose:', torch.allclose(y_conv[0], y_seq[0], atol=1e-4))
    y_conv[0].sum().backward()
    conv_grad = {}
    for n, p in layer.named_parameters():
        conv_grad[n] = p.grad
        p.grad = None

    y_lcse[0].sum().backward()
    lcse_grad = {}
    for n, p in layer.named_parameters():
        lcse_grad[n] = p.grad
        p.grad = None

    y_seq[0].sum().backward()
    seq_grad = {}
    for n, p in layer.named_parameters():
        seq_grad[n] = p.grad
        p.grad = None

    for k in conv_grad.keys():
        print(f'CONV/SEQ {k} grad allclose:', torch.allclose(conv_grad[k], seq_grad[k], atol=1e-4))
        print(f'LCSE/SEQ {k} grad allclose:', torch.allclose(lcse_grad[k], seq_grad[k], atol=1e-4))

    repeats = 10
    log_x = np.arange(8, 16)
    y_lcse = []
    y_conv = []
    y_seq = []

    for log_length in log_x:
        print('Seq length:', 2 ** log_length)
        x = torch.randn((2 ** log_length, 1, dim), dtype=torch.float32)
        t0 = time.time()
        for _ in range(repeats):
            layer.lcse(x)
        delta_t = time.time() - t0
        print('lcse:', delta_t)
        y_lcse.append(delta_t)
        t0 = time.time()
        for _ in range(repeats):
            layer.conv(x)
        delta_t = time.time() - t0
        print('conv:', delta_t)
        y_conv.append(delta_t)
        t0 = time.time()
        for _ in range(repeats):
            layer.seq(x)
        delta_t = time.time() - t0
        print('seq:', time.time() - t0)
        y_seq.append(delta_t)

    plt.plot(log_x, y_lcse, label='LogCumSumExp')
    plt.plot(log_x, y_conv, label='Conv')
    plt.plot(log_x, y_seq, label='Sequential')
    plt.ylabel('Runtime')
    plt.xlabel('Log Sequence Length')
    plt.legend()
    plt.savefig('profile.png')
