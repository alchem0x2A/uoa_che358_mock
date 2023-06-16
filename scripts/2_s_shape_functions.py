"""Use least square fit
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.special import erf

def erf_scaled(x):
    return (erf(x) + 1) / 2

def logistic(x):
    return 1 / (1 + np.exp(-x))

def tanh_scaled(x):
    return (np.tanh(x) + 1) / 2

def atan_scaled(x):
    return (np.arctan(x) / np.pi) + 0.5


def plot_main():
    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    x = np.linspace(-5, 5, 256)
    ax.plot(x, tanh_scaled(x), label=r"$(\mathrm{tanh}(x) + 1) / 2$", lw=2, alpha=0.6)
    ax.plot(x, atan_scaled(x), label=r"$(\mathrm{arctan}(x) / \pi) + 1 / 2$", lw=2, alpha=0.6)
    ax.plot(x, erf_scaled(x), label=r"$(\mathrm{erf}(x) + 1) / 2$", lw=2, alpha=0.6 )
    ax.plot(x, logistic(x), label=r"$(1 + e^{-x})^{-1}$", lw=3, color="k")

    ax.legend()
    fig.savefig("sigmoid_funs.pdf")

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    x = np.linspace(-5, 5, 256)
    ax.plot(x, logistic(x), label=r"$(1 + e^{-x})^{-1}$", lw=3, color="k")

    # ax.legend()
    fig.savefig("logistic_fun_alone.pdf")
    return

if __name__ == "__main__":
    plot_main()
