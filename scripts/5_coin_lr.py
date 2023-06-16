"""Use least square fit
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

data = np.load("result_coin.npz", allow_pickle=True)
r, p, y = data["r"], data["p"], data["flip"]
rr = r.reshape(-1, 1)

def plot_main():
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, ls="--", color="grey", lw=1.5)
    ax.axhline(y=0.0, ls="--", color="grey", lw=1.5)
    ax.set_xlabel("$r$")
    ax.set_ylabel("$y$")
    ax.plot(r[y == 1], y[y == 1], "o", alpha=0.2)
    ax.plot(r[y == 0], y[y == 0], "o", alpha=0.2)
    lr = LogisticRegression(penalty="l2", C=1.0)
    clr = lr.fit(rr, y)
    xx_p = np.linspace(0, 1).reshape(-1, 1)
    yy_p = clr.predict_proba(xx_p)[:, 1]
    print(clr.coef_, clr.intercept_)
    ax.plot(xx_p, yy_p, lw=2.0, color="#7f5c00")
    # print(xx, yy)
    # ax.hist(residuals, bins=20)
    fig.savefig("coin_fit_lr.pdf")
    
    return

if __name__ == "__main__":
    plot_main()
