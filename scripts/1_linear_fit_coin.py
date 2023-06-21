"""Use least square fit
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = np.load("result_coin.npz", allow_pickle=True)
r, p, y = data["r"], data["p"], data["flip"]


def plot_main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax, ax2 = axes[0], axes[1]

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 1.3)
    # ax.set_yticks([0, 1])
    ax.axhline(y=1.0, ls="--", color="grey", lw=1.5)
    ax.axhline(y=0.0, ls="--", color="grey", lw=1.5)
    ax.set_xlabel("$x$ (tail weight ratio)")
    ax.set_ylabel("$y$ (head or tail)")
    (l1,) = ax.plot(r[y == 1], y[y == 1], "o", alpha=0.2)
    (l2,) = ax.plot(r[y == 0], y[y == 0], "o", alpha=0.2)

    fit = np.polyfit(r, y, deg=1)
    fit_fun = np.poly1d(fit)
    xx = np.linspace(-5, 5, 100)
    yy = fit_fun(xx)
    y_fit = fit_fun(r)
    r2 = r2_score(y, y_fit)
    print(f"R^2: {r2}")
    ax.fill_between([0, 1], 1.0, 1.3, color=l1.get_c(), alpha=0.5)
    ax.fill_between([0, 1], -0.3, 0, color=l2.get_c(), alpha=0.5)
    ax.plot(xx, yy, lw=3.0, color="#7f5c00")
    ax.text(
        x=0.5, y=fit_fun(0.5) - 0.08, s=f"$p = {{{fit[1]:.2f}}} + {{{fit[0]:.2f}}}x$"
    )
    ax.text(x=0.5, y=fit_fun(0.5) - 0.25, s=f"$R^2 = {{{r2:.2f}}}$")
    ax.text(x=0.5, y=1.15, va="center", ha="center", style="italic", s="Invalid Region")
    ax.text(
        x=0.5, y=-0.15, va="center", ha="center", style="italic", s="Invalid Region"
    )
    # print(xx, yy)
    # fig.tight_layout()
    # fig.savefig("coin_linear_fit.pdf")

    residuals = y - y_fit
    # ax.hist(residuals, bins=20)
    ax2.plot(y_fit, residuals, "o", color="grey")
    ax2.set_xlabel("Predicted values")
    ax2.set_ylabel("Residuals")

    fig.tight_layout()

    fig.savefig("coin_linear_combined.pdf")

    return


if __name__ == "__main__":
    plot_main()
