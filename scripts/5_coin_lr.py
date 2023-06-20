"""Use least square fit
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = np.load("result_coin.npz", allow_pickle=True)
r, p, y = data["r"], data["p"], data["flip"]
rr = r.reshape(-1, 1)


def plot_main():
    fig, ax = plt.subplots(figsize=(7, 6.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, ls="--", color="grey", lw=1.5)
    ax.axhline(y=0.0, ls="--", color="grey", lw=1.5)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$p(x)$")
    ax.plot(r[y == 1], y[y == 1], "o", alpha=0.2)
    ax.plot(r[y == 0], y[y == 0], "o", alpha=0.2)
    lr = LogisticRegression(penalty="l2", C=1.0)
    clr = lr.fit(rr, y)
    accuracy = clr.score(rr, y)
    print(accuracy)
    xx_p = np.linspace(0, 1).reshape(-1, 1)
    yy_p = clr.predict_proba(xx_p)[:, 1]
    w, b = clr.coef_[0][0], clr.intercept_[0]
    print(w, b)
    x0 = -b / w
    ax.plot(xx_p, yy_p, lw=2.0, color="#7f5c00")
    ax.axvline(x=x0, color="grey", alpha=0.8, lw=2.5, ls="-")
    ax.text(
        x=0.02,
        y=0.75,
        ha="left",
        va="bottom",
        s=f"$p(x) = \\dfrac{{1}}{{1 + e^{{({{{b:.2f}}} + {{{w:.2f}}}x)}}}}$",
    )
    ax.text(
        x=0.02, y=0.65, ha="left", va="bottom", s=f"Accuracy: {accuracy * 100:.1f}%"
    )
    ax.text(x=x0, y=1.25, s=f"Decision Boundary", va="bottom", ha="center")
    ax.text(x=x0, y=1.15, s=f"$x={x0:.2f}$", va="bottom", ha="center")
    print(x0)
    fig.tight_layout()
    fig.savefig("coin_fit_lr.pdf")

    return


if __name__ == "__main__":
    plot_main()
