import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# style
plt.style.use(['science','no-latex'])
plt.rcParams.update({
    "font.family": "STIXGeneral",   # widely available, has bold
})

# --- Load for the second subplot (Actions on  Ball) ---
gd = np.load("gd_points.npz")
lgd = np.load("lgd_points.npz")

# pick an action snapshot (last 4 points)
actions = gd["actions"][97:]
unconst = gd["unconst"][97:]
actions_l = lgd["actions"][97:]
unconst_l = lgd["unconst"][97:]

# ℓ₁-ball diamond
diamond = np.array([[ 1,  0], [ 0,  1], [-1,  0], [ 0, -1], [ 1,  0]])

plt.figure(figsize=(4,3))

plt.plot(diamond[:, 0], diamond[:, 1], linewidth=1, color='k')


# Transparent interior and green edge for GD
plt.scatter(
    actions[:, 0], actions[:, 1],
    s=64, marker='o',
    facecolors=(0, 0, 1, 0.6),
    linewidths=1, label="GD"
)
# Transparent interior and green edge for LGD
plt.scatter(
    actions_l[:, 0], actions_l[:, 1],
    s=64, marker='s',
    facecolors=(1, 0.5, 0, 0.6),
    linewidths=1, label="LazyGD"
)


# Draw arrows for movement (GD)
for i in range(len(actions) - 1):
    x0, y0 = actions[i, 0], actions[i, 1]
    x1, y1 = actions[i + 1, 0], actions[i + 1, 1]
    plt.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.025, head_length=0.035, fc='red', length_includes_head=True, alpha=0.8, lw=0.5)

# closing the loop
x1, y1 = actions[0, 0], actions[0, 1]
x0, y0 = actions[ 3, 0], actions[ 3, 1]
plt.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.025, head_length=0.035, fc='red', length_includes_head=True, alpha=0.8, lw=0.5)

# plt.set_box_aspect(1)

plt.xlabel(r"$x_1$", fontsize=18, fontweight='bold')
plt.ylabel(r"$x_2$", fontsize=18, fontweight='bold')
plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')
plt.legend(fontsize=14, frameon=False)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlim(0.65, 1.05)
plt.ylim(-0.15, 0.15)
plt.savefig("actions_ell1.pdf", bbox_inches='tight')  # save first

plt.show()