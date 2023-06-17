"""Use least square fit
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.special import erf

def log1(p):
    return -np.log(p)

def log2(p):
    return -np.log(1-p)



def plot_main():
    fig, ax = plt.subplots(figsize=(5, 8))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 2.5)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.set_ylabel(r"$l(\theta|y=1)$")
    ax.set_xlabel(r"$p(\theta)$")
    x = np.linspace(0, 1, 256)
    ax.plot(x, log1(x), lw=2)
    fig.savefig("loss_1.pdf")

    fig, ax = plt.subplots(figsize=(5, 8))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 2.5)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.set_ylabel(r"$l(\theta|y=0)$")
    ax.set_xlabel(r"$p(\theta)$")
    x = np.linspace(0, 1, 256)
    ax.plot(x, log2(x), lw=2)
    fig.savefig("loss_2.pdf")

    return

if __name__ == "__main__":
    plot_main()
