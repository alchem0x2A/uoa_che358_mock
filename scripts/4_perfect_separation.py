import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import LogisticRegression

data = np.load("result_coin.npz", allow_pickle=True)
r, p, y = data["r"], data["p"], data["flip"]
new_r, new_y = [], []
for r_, y_ in zip(r, y):
    if abs(r_ - 0.5) > 0.15:
        if (r_ - 0.5) * (y_ - 0.5) > 0:
            new_r.append(r_)
            new_y.append(y_)
# rr = r.reshape(-1, 1)
# Create a "perfectly separable model"
# condition = np.where(np.abs(r - 0.5) > 0.1)[0]
# print(condition)
r = np.array(new_r)
# r = r[condition]
rr = r.reshape(-1, 1)
y = np.array(new_y)
print(r.shape, y.shape, rr.shape)


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


# x = np.hstack([np.linspace(-2, -1.0, 50), np.linspace(1.0, 2, 50)])
# y = f(x)


def plot_main():
    fig, ax = plt.subplots(figsize=(11.5, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    ax.axvline(x=0.5, ls="--", color="grey", lw=2)
    mask1 = np.where((y > 0.95))[0]
    mask2 = np.where((y < 0.05))[0]
    ax.plot(r[mask1], y[mask1], "o", alpha=0.8)
    ax.plot(r[mask2], y[mask2], "o", alpha=0.8)
    weights = [10, 15, 20, 50, 70]
    xx = np.linspace(-0.2, 1.2, 200)
    old_ls = []
    for w in weights:
        b = w * -0.5
        loss = log_loss(r, y, w, b)
        print(w, loss)
        (l,) = ax.plot(
            xx, p(xx, w, b), lw=2, alpha=0.8, label=f"$\\beta_{{1}}$={w}, loss={loss:.2f}"
        )
        old_ls.append(l)
    arrow = ax.annotate(r'Increasing $\beta_{1}$',
                        ha="right",
                xy=(0.65, 0.65), xytext=(0.45, 0.80),
                arrowprops=dict(arrowstyle='<-'))
    leg_ = ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.savefig("perfect_sep.pdf")
    # Add a new dataset
    new_x = [0.55, 0.58, 0.63, 0.65, 0.68]
    new_y = [0, 0, 0, 0, 0]

    r_nd = np.hstack([r, new_x])
    y_nd = np.hstack([y, new_y])
    rr_nd = r_nd.reshape(-1, 1)

    weights = [10, 15, 20, 50, 70]
    xx = np.linspace(-0.2, 1.2, 200)
    for i, w in enumerate(weights):
        b = w * -0.5
        loss = log_loss(r_nd, y_nd, w, b)
        print(w, loss)
        old_ls[i].set_label(f"$\\beta_{{1}}$={w}, loss={loss:.2f}")

    (l_new,) = ax.plot(new_x, new_y, "^", markersize=15, color="magenta")
    t_new = ax.text(x=0.61, y=0.15, ha="center", s="New data?")
    leg_ = ax.legend()
    fig.savefig("perfect_sep_new_data.pdf")
    l_new.remove()
    t_new.remove()

    # pass2, use regularization

    # fig, ax = plt.subplots(figsize=(10, 6))
    # ax.set_xlim(-2.2, 2.2)
    # ax.set_ylim(-0.1, 1.1)
    # ax.axhline(y=1.0, ls="--", color="grey", lw=2)
    # ax.axhline(y=0.0, ls="--", color="grey", lw=2)
    # ax.axvline(x=0.0, ls="--", color="grey", lw=2)
    # ax.plot(x[y > 0.95], y[y > 0.95], "o", alpha=0.8)
    # ax.plot(x[y < 0.05], y[y < 0.05], "o", alpha=0.8)
    # weights = [2, 5, 10, 15]
    # for w in weights:
    #     loss = log_loss(x, y, w, 0)
    #     ax.plot(xx, p(xx, w, 0), lw=1, alpha=0.1, label=f"{w}")

    lr = LogisticRegression(penalty="l2", C=10)
    x_r = r.reshape(-1, 1)
    xx_r = xx.reshape(-1, 1)
    clf = lr.fit(x_r, y)
    print(clf.coef_[0][0])
    w, b = clf.coef_[0][0], clf.intercept_[0]
    # print(clf.intercept_)
    ax.plot(xx_r, clf.predict_proba(xx_r)[:, 1], lw=3, alpha=1.0, color="grey")
    for l_ in old_ls:
        l_.set_alpha(0.1)
    ax.text(
        x=0.6, y=0.5, ha="left", va="bottom", s=f"Weight: {w:.2f}, L2 $\\lambda=0.1$, "
    )
    leg_.remove()
    fig.savefig("perfect_sep_reg.pdf")
    return


if __name__ == "__main__":
    plot_main()
