import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


curdir = Path(__file__).parent


def gen_all_points(d1, d2, d3, limit=(0.01, 0.01)):
    """Generate 3D point cloud from 2D projections"""
    all_points = []
    for x, z1 in d1:
        for y, z2 in d2:
            if abs(z1 - z2) < limit[0]:
                new_pt = [x, y, (z1 + z2) / 2]
                for y3, x3 in d3:
                    if np.sqrt((x - x3) ** 2 + (y - y3) ** 2) < limit[1]:
                        all_points.append(new_pt)
    return np.array(all_points)


def get_all_data(seed=0):
    """Re-generate 3D point cloud + random noise"""
    # Dataset obtained from publication
    # temp - cap (x, z)
    np.random.seed(seed)
    data1 = np.genfromtxt(
        curdir.parent / "./data/battery/grey-fig1.csv",
        delimiter=",",
    )[:, :2]
    # vol - cap (y, z)
    data2 = np.genfromtxt(
        curdir.parent / "./data/battery/grey-fig2.csv",
        delimiter=",",
    )[:, :2]
    # vol - temp (y, x)
    data3 = np.genfromtxt(
        curdir.parent / "./data/battery/grey-fig3.csv",
        delimiter=",",
    )[:, :2]

    # temp - cap (x, z)
    data4 = np.genfromtxt(
        curdir.parent / "./data/battery/red-fig1.csv",
        delimiter=",",
    )[:, :2]
    # vol - cap (y, z)
    data5 = np.genfromtxt(
        curdir.parent / "./data/battery/red-fig2.csv",
        delimiter=",",
    )[:, :2]
    # vol - temp (y, z)
    data6 = np.genfromtxt(
        curdir.parent / "./data/battery/red-fig3.csv",
        delimiter=",",
    )[:, :2]

    # successful materials
    grey_pts = gen_all_points(data1, data2, data3, limit=(0.003, 0.005))
    label_grey = np.ones(len(grey_pts))
    # failed tests
    red_pts = gen_all_points(data4, data5, data6, limit=(0.013, 0.013))
    label_red = np.zeros(len(red_pts))

    random_data = np.random.uniform([0, 0.3, 0.8], [1, 1, 1], size=(12, 3))
    random_labels = np.random.choice([0, 1], size=12)

    all_data = np.vstack([grey_pts, red_pts, random_data])
    labels = np.hstack([label_grey, label_red, random_labels])

    print(len(data1), len(data4))
    print(len(grey_pts), len(red_pts))
    return all_data, labels


X, y = get_all_data()


def train_model(test_split=0.2, train_split=0.8, l=0.1, X=X, y=y, seed=1):
    N_train = int(len(X) * train_split)
    N_test = int(len(X) * test_split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_train, test_size=N_test, shuffle=True, random_state=seed
    )
    lr = LogisticRegression(penalty="l2", C=1 / l)
    model = lr.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    data = [X_train, X_test, y_train, y_test]
    return accuracy, model, data


def plot_hyper():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    acc_lr = []
    lr_val = np.logspace(-3, 4, 50)
    for i, l in enumerate(lr_val):
        accs = []
        for j in range(8):
            acc, *_ = train_model(train_split=0.7, test_split=0.3, l=l, seed=j)
            accs.append(acc)
        acc_lr.append(accs)
    acc_lr = np.array(acc_lr)
    mean_acc = np.mean(acc_lr, axis=1)
    std_acc = np.std(acc_lr, axis=1)
    (l,) = ax.plot(lr_val, mean_acc, "o-")
    ax.fill_between(
        lr_val, mean_acc - std_acc, mean_acc + std_acc, color=l.get_c(), alpha=0.3
    )
    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel(r"$\lambda$ (L2 regularization penalty)")
    ax.set_ylabel(r"Test set accuracy")
    ax.axvspan(1e-3, 10, color="#eba434", alpha=0.2)
    ax.text(x=0.1, y=1.06, va="bottom", ha="center", s=r"Practical Range for $\lambda$")

    # ax.plot(lr_val, acc_lr, "o")
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(curdir / "batter_hyper.pdf")
    return


def make_scatter(ax, data, first=0, second=2):
    """plot test & train"""
    X_train, X_test, y_train, y_test = data
    ax.scatter(
        X_train[y_train > 0.95, first],
        X_train[y_train > 0.95, second],
        marker="o",
        color="grey",
    )
    ax.scatter(
        X_train[y_train < 0.05, first],
        X_train[y_train < 0.05, second],
        marker="o",
        color="red",
    )
    ax.scatter(
        X_test[y_test > 0.95, first],
        X_test[y_test > 0.95, second],
        marker="o",
        edgecolor="black",
        color="grey",
    )
    ax.scatter(
        X_test[y_test < 0.05, first],
        X_test[y_test < 0.05, second],
        marker="o",
        edgecolor="black",
        color="red",
    )

def make_scatter_3d(ax, data):
    """plot test & train"""
    X_train, X_test, y_train, y_test = data
    ax.scatter(
        X_train[y_train > 0.95, 0],
        X_train[y_train > 0.95, 1],
        X_train[y_train > 0.95, 2],
        s=100,
        marker="o",
        color="grey",
    )
    ax.scatter(
        X_train[y_train < 0.05, 0],
        X_train[y_train < 0.05, 1],
        X_train[y_train < 0.05, 2],
        s=100,
        marker="o",
        color="red",
    )
    ax.scatter(
        X_test[y_test > 0.95, 0],
        X_test[y_test > 0.95, 1],
        X_test[y_test > 0.95, 2],
        s=100,
        marker="o",
        edgecolor="black",
        linewidths=2,
        color="grey",
    )
    ax.scatter(
        X_test[y_test < 0.05, 0],
        X_test[y_test < 0.05, 1],
        X_test[y_test < 0.05, 2],
        s=100,
        marker="o",
        edgecolor="black",
        linewidths=2,
        color="red",
    )
    return

def plot_surf3d(ax, model):
    
    x = np.linspace(0, 1, 32)
    y = np.linspace(0.3, 1, 32)
    z = np.linspace(0.8, 1, 32)

    grid_x, grid_y, grid_z = np.meshgrid(x, y, z)

    grid_points = np.c_[grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]
    probabilities = model.predict_proba(grid_points)[:, 1]
    ax.scatter(grid_points[:, 0], grid_points[:, 1], grid_points[:, 2], s=200,
               c=probabilities, cmap='RdGy', alpha=0.02)
    xx, yy = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0.3, 1, 256))
    zz = np.ones_like(yy) * 0.9
    grid_2d = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    prob_2d = model.predict_proba(grid_2d)[:, 1]
    cc = prob_2d.reshape(yy.shape)
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    cond = np.where(np.abs(prob_2d - 0.5) < 0.05)[0]
    fitfun = np.poly1d(np.polyfit(xx.ravel()[cond], yy.ravel()[cond], deg=1))
    # ax.plot3D(xx.ravel(), fitfun(xx.ravel()), zz.ravel())
    
    

    # ax.pcolor(xx, yy, zz, cmap="RdGy", alpha=0.5)
    return

def plot_boundary(ax, model, first, second, x=None, y=None, z=None):
    if x is None:
        x = np.linspace(0, 1, 100)
    if y is None:
        y = np.linspace(0, 1, 100)
    if z is None:
        z = np.linspace(0, 1, 100)
    xx, yy, zz = np.meshgrid(x, y, z)
    mesh_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    dims = [xx, yy, zz]
    dims2 = [x, y, z]

    # Predict the probabilities for the meshgrid points
    probs = model.predict_proba(mesh_points)[:, 1]
    probs = probs.reshape(xx.shape)
    boundary_points = np.where(np.abs(probs - 0.5) < 0.01)
    # zz = model.predict_proba()

    af, bf = dims[first][boundary_points], dims[second][boundary_points]
    fitfun = np.poly1d(np.polyfit(af, bf, 1))
    ax.plot(dims2[first], fitfun(dims2[first]), ls="--", color="grey", lw=3)

    return


def plot_decision():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]
    acc, model, data = train_model(train_split=0.7, test_split=0.3, l=0.01, seed=42)
    X_train, X_test, y_train, y_test = data
    print(model.coef_, model.intercept_)
    # temp, cap
    make_scatter(ax1, data, 0, 2)
    # vol, cap
    make_scatter(ax2, data, 1, 2)
    # vol, temp
    make_scatter(ax3, data, 1, 0)
    ax1.set_xlabel("Temp consistency")
    ax2.set_xlabel("Vol consistency")
    ax3.set_xlabel("Vol consistency")
    ax1.set_ylabel("Cap consistency")
    ax2.set_ylabel("Cap consistency")
    ax3.set_ylabel("Temp consistency")
    ax1.set_xlim(0, 1.0)
    ax1.set_ylim(0.8, 1.0)
    ax2.set_xlim(0.3, 1.0)
    ax2.set_ylim(0.8, 1.0)
    ax3.set_xlim(0.3, 1.0)
    ax3.set_ylim(0.0, 1.0)
    l1, l2, l3 = 0.7, 0.5, 0.9
    plot_boundary(ax1, model, 0, 2, y=np.ones(100) * l1)
    plot_boundary(ax2, model, 1, 2, x=np.ones(100) * l2)
    plot_boundary(ax3, model, 1, 0, z=np.ones(100) * l3)
    ax1.set_title(f"Vol consistency = {l1:.2f}")
    ax2.set_title(f"Temp consistency = {l2:.2f}")
    ax3.set_title(f"Cap consistency = {l3:.2f}")

    # ax.plot(lr_val, acc_lr, "o")
    fig.tight_layout()
    fig.savefig(curdir / "batter_decision.pdf")
    return

def plot_3d():
    from mpl_toolkits.mplot3d import Axes3D
    acc, model, data = train_model(train_split=0.7, test_split=0.3, l=1, seed=42)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-45)
    ax.set_xlabel('Temp consistency')
    ax.set_ylabel('Vol consistency')
    ax.set_zlabel('Cap consistency')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.3, 1.0)
    ax.set_zlim(0.8, 1.0)
    
    make_scatter_3d(ax, data)
    plot_surf3d(ax, model)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    # fig.tight_layout()
    # plt.show()
    fig.savefig(curdir / "battery_3d.png")
    return
    


if __name__ == "__main__":
    plot_hyper()
    plot_decision()
    plot_3d()
