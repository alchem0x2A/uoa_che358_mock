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
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    ax.plot([-5, 0], [0.5, 0.5], ls="--", color="grey", lw=2)
    ax.set_xlabel("$z$")
    ax.set_ylabel("Normalized $p(z)$")
    x = np.linspace(-5, 5, 256)
    (l1,) = ax.plot(
        x, tanh_scaled(x), label=r"$(\mathrm{tanh}(z) + 1) / 2$", lw=2, alpha=1
    )
    (l2,) = ax.plot(
        x, atan_scaled(x), label=r"$(\mathrm{arctan}(z) / \pi) + 1 / 2$", lw=2, alpha=1
    )
    (l3,) = ax.plot(
        x, erf_scaled(x), label=r"$(\mathrm{erf}(z) + 1) / 2$", lw=2, alpha=1
    )
    (l4,) = ax.plot(x, logistic(x), label=r"$1/ (1 + e^{-z})$", lw=2, color="k")

    legend_ = ax.legend()
    fig.tight_layout()
    fig.savefig("sigmoid_funs.pdf")

    l1.set_alpha(0.1)
    l2.set_alpha(0.1)
    l3.set_alpha(0.1)
    l4.set_linewidth(4)
    legend_.remove()

    ax.text(
        x=0.4,
        y=0.5,
        va="center",
        s=r"Logistic function: $p(z) = \dfrac{1}{1 + e^{-z}}$",
    )

    fig.savefig("sigmoid_funs_emphasize_logistic.pdf")

    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.set_xlim(-5, 5)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 0.5, 1])
    ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    ax.plot([-5, 0], [0.5, 0.5], ls="--", color="grey", lw=2)
    ax.plot([0], [0.5], "o")
    ax.set_xlabel("$z$")
    ax.set_ylabel("$p(z)$")
    x = np.linspace(-5, 5, 256)
    ax.plot(x, logistic(x), lw=3, color="k")
    ax.text(x=-4.6, y=0.90, va="top", ha="left", s=r"$p(z) = \dfrac{1}{1 + e^{-z}}$")
    ax.text(x=0.4, y=0.45, s="Point of inflection")
    ax.text(x=0, y=1.12, s="Decision boundary $z=0$", ha="center", va="bottom")

    # ax.legend()
    fig.tight_layout()
    fig.savefig("logistic_fun_alone.pdf")
    return


if __name__ == "__main__":
    plot_main()
