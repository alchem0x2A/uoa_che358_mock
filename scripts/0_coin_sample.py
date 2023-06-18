import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(1)  # For reproducibility, seed with r
# Define the function f(r)
def f(r):
    v = 1 / (1 + np.exp(-(r - 0.5) * 10))
    # + np.random.uniform(0, 0.5)
    return max(min(v, 0.96), 0.04)

# Set the values of r
# r_values = np.linspace(0.1, 5, 50)
r_values = np.random.uniform(0, 1, 500)
p_values = np.array([f(r) for r in r_values])
res = np.array([np.random.choice([0, 1], p=[1-p, p]) for p in p_values])
zeros = []
ones = []

# Prepare empty lists to store the values of r and flips for the plot
# plot_r_values = []
# plot_flips = []

# Initialize a figure
fig, ax = plt.subplots(figsize=(8, 5))
scatter_one, = ax.plot([], [], "o", alpha=0.2)
scatter_zero, = ax.plot([], [], "o", alpha=0.2)

# Initialization function for the animation
def init():
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['T ($y=0$)', 'H ($y=1$)'])
    ax.axhline(y=1.0, ls="--", color="grey", lw=1.5)
    ax.axhline(y=0.0, ls="--", color="grey", lw=1.5)
    ax.set_xlabel("$x$ (Head weight ratio)")
    ax.set_ylabel("$y$ Head or Tail")
    # ax.grid(True)
    return ax,

# Update function for the animation
def update(i):
    r, flip = r_values[i], res[i]
    if flip == 1:
        ones.append([r, flip])
    else:
        zeros.append([r, flip])
    plot_ones = np.atleast_2d(ones)
    plot_zeros = np.atleast_2d(zeros)
    # ax.clear()
    # init()
        # ax.plot(plot_ones[:, 0], plot_ones[:, 1], 'o', alpha=0.1)
        # ax.plot(plot_zeros[:, 0], plot_zeros[:, 1], 'o', alpha=0.1)
    if plot_ones.shape[1] > 0:
        scatter_one.set_data(plot_ones[:, 0], plot_ones[:, 1])
    if plot_zeros.shape[1] > 0:
        scatter_zero.set_data(plot_zeros[:, 0], plot_zeros[:, 1])
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(len(r_values)), init_func=init, blit=False)

# Show animation
# plt.show()
ani.save('animation.mov', fps=90)
# ani.save('animation.png')
fig.savefig("sample_coin_cover.pdf")
data = {"r": r_values, "p": p_values, "flip": res}
np.savez("result_coin.npz", **data)
