import os

import numpy as np
import matplotlib.pyplot as plt
import torch


def kuma_pdf(x, a, b):
    pdf = np.zeros_like(x, dtype=float)
    valid_x = (x > 0) & (x < 1)
    x_valid = x[valid_x]

    term1 = a * b
    term2 = x_valid ** (a - 1)
    term3 = (1 - x_valid ** a) ** (b - 1)

    pdf[valid_x] = term1 * term2 * term3

    return pdf


def plot_kuma_pdfs(a, b):
    x = np.linspace(1e-5, 1 - 1e-2, 500)
    plt.figure(figsize=(12, 8))

    pdf_values = kuma_pdf(x, a, b)

    plt.plot(x, pdf_values, lw=2.5)

    plt.title('Kumaraswamy Distribution PDF ', fontsize=16)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.legend(fontsize=11)
    plt.xlim(0, 1)
    plt.ylim(bottom=0)
    plt.show()


if __name__ == '__main__':
    output_dir = "/path/"
    step = 150
    a = torch.load(os.path.join(output_dir, f"a_step={step}.pt"))
    b = torch.load(os.path.join(output_dir, f"b_step={step}.pt"))
    layer = 2
    head = 7

    a_clamp = a.clamp(1e-6, 100.0)  # extreme values could result in NaNs
    b_clamp = b.clamp(1e-6, 100.0)  # extreme values could result in NaNs

    plot_kuma_pdfs(a_clamp[layer,head].item(),b_clamp[layer,head].item())