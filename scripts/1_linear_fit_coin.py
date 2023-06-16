"""Use least square fit
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

data = np.load("result_coin.npz", allow_pickle=True)
r, p, y = data["r"], data["p"], data["flip"]

def plot_main():
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.axhline(y=1.0, ls="--", color="grey", lw=1.5)
    ax.axhline(y=0.0, ls="--", color="grey", lw=1.5)
    ax.set_xlabel("$r$")
    ax.set_ylabel("$y$")
    ax.plot(r[y == 1], y[y == 1], "o", alpha=0.2 )
    ax.plot(r[y == 0], y[y == 0], "o", alpha=0.2 )

    fit = np.polyfit(r, y, deg=1)
    fit_fun = np.poly1d(fit)
    xx = np.linspace(-5, 5, 100)
    yy = fit_fun(xx)
    y_fit = fit_fun(r)
    r2 = r2_score(y, y_fit)
    print(f"R^2: {r2}")
    ax.plot(xx, yy, lw=2.0, color="#7f5c00")
    # print(xx, yy)
    fig.savefig("coin_linear_fit.pdf")

    residuals = y - y_fit
    fig, ax = plt.subplots(figsize=(4, 4))
    # ax.hist(residuals, bins=20)
    ax.plot(y_fit, residuals, "o")
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Residuals")
    fig.tight_layout()
    fig.savefig("coin_linear_res.pdf")
    
    return

if __name__ == "__main__":
    plot_main()
