import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LogisticRegression

def f(x):
    """sign function"""
    return (np.sign(x) + 1) / 2

def p(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))

def log_loss(x, y, w, b):
    losses = []
    for xi, yi in zip(x, y):
        pi = p(xi, w, b)
        losses.append(yi * np.log(pi) + (1 - yi) * np.log(1 - pi))
    return -np.sum(losses)
    

x = np.hstack([np.linspace(-2, -1.0, 50), np.linspace(1.0, 2, 50)])
y = f(x)

def plot_main():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    ax.plot(x[y > 0.95], y[y > 0.95], "o", alpha=0.8)
    ax.plot(x[y < 0.05], y[y < 0.05], "o", alpha=0.8)
    weights = [2, 5, 10, 15]
    xx = np.linspace(-3, 3, 200)
    for w in weights:
        loss = log_loss(x, y, w, 0)
        print(w, loss)
        ax.plot(xx, p(xx, w, 0), lw=2, alpha=0.8, label=f"{w}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.savefig("perfect_sep.pdf")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    ax.plot(x[y > 0.95], y[y > 0.95], "o", alpha=0.8)
    ax.plot(x[y < 0.05], y[y < 0.05], "o", alpha=0.8)
    weights = [2, 5, 10, 15]
    xx = np.linspace(-3, 3, 200)
    for w in weights:
        loss = log_loss(x, y, w, 0)
        ax.plot(xx, p(xx, w, 0), lw=1, alpha=0.1, label=f"{w}")

    lr = LogisticRegression(penalty="l2", C=10)
    x_r = x.reshape(-1, 1)
    xx_r = xx.reshape(-1, 1)
    clf = lr.fit(x_r, y)
    print(clf.coef_[0][0])
    print(clf.intercept_)
    ax.plot(xx_r, clf.predict_proba(xx_r)[:, 1], lw=3, alpha=1.0, color="grey", label=f"{w}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.savefig("perfect_sep_reg.pdf")
    return

if __name__ == "__main__":
    plot_main()

