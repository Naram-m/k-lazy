import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# -----------------------------
# Parameters
# -----------------------------
phase_len = 20
T = 2 * phase_len
x = np.arange(T)

mu1, mu2 = 1.0, -1.0
sigma = np.sqrt(10.0)  # standard deviation = √10

# Mean arrays
y1 = np.full(phase_len, mu1)
y2 = np.full(phase_len, mu2)

brown = "#8B4513"

# Nearest of {+1, -1}
def nearest_sign(y):
    return 1.0 if abs(y - 1) <= abs(y + 1) else -1.0

# -----------------------------
# Fixed illustrative points and their targets
# Phase 1: 3 project to +1, 2 project to -1
# Phase 2: 3 project to -1, 2 project to +1
# -----------------------------
t1 = np.array([8, 10, 12, 14, 16])  # inside phase-1 window [0, 20)
y1_points = np.array([ 2.6, 1.7, 0.2, -2.4, -1.8 ])  # => +1, +1, +1, -1, -1
t2 = np.array([28, 30, 32, 34, 36])                  # inside phase-2 window [20, 40)
y2_points = np.array([ -1.6, -3.4, -0.2,  0.5,  2.8 ])  # => -1, -1, -1, +1, +1

y1_proj = np.array([nearest_sign(v) for v in y1_points])
y2_proj = np.array([nearest_sign(v) for v in y2_points])

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 2.8))

# Std-dev shading (±σ) in brown
ax.fill_between(x[:phase_len], mu1 - sigma, mu1 + sigma, color=brown, alpha=0.12, linewidth=0)
ax.fill_between(x[phase_len:], mu2 - sigma, mu2 + sigma, color=brown, alpha=0.12, linewidth=0)

# Mean lines in brown
ax.plot(x[:phase_len], y1, color=brown, linewidth=2.5, solid_capstyle="round")
ax.plot(x[phase_len:], y2, color=brown, linewidth=2.5, solid_capstyle="round")

# Sample points (black)
ax.scatter(t1, y1_points, s=30, color="grey", zorder=5)
ax.scatter(t2, y2_points, s=30, color="grey", zorder=5)

# Projection lines to nearest {+1, -1} and projected markers (white circles)
for ti, yi, yp in zip(t1, y1_points, y1_proj):
    ax.plot([ti, ti], [yi, yp], linestyle="--", linewidth=1, color="gray", zorder=4)
    if yp == 1:
        ax.scatter([ti], [yp], s=28, facecolors=brown, edgecolors="black", zorder=6)
    if yp == -1:
        ax.scatter([ti], [yp], s=40, facecolors='red', marker='x', zorder=6)

for ti, yi, yp in zip(t2, y2_points, y2_proj):
    ax.plot([ti, ti], [yi, yp], linestyle="--", linewidth=1, color="gray", zorder=4)
    if yp == -1:
        ax.scatter([ti], [yp], s=28, facecolors=brown, edgecolors="black", zorder=6)
    if yp == 1:
        ax.scatter([ti], [yp], s=40, facecolors='red', marker='x', zorder=6)


# Reference line at 0
ax.axhline(0, color="black", linewidth=1)

# Axes (bold; no grid)
ax.set_xlabel("T", fontsize=13, fontweight="bold")
ax.set_ylabel("Signal", fontsize=13, fontweight="bold")
ax.set_xticks([0, 10, 20, 30, 40])

# Y-range wide enough for ±√10 band
ymin = min(mu2 - sigma, mu1 - sigma) - 0.3
ymax = max(mu1 + sigma, mu2 + sigma) + 0.3
ax.set_ylim(ymin-0.5, ymax-0.5)
ax.set_xlim(0, T - 1)

# Y-ticks
yticks = np.arange(np.floor(ymin), np.ceil(ymax) + 1, 2)
ax.set_yticks(yticks)

ax.tick_params(axis='both', which='both', labelsize=12, width=1.5, length=6)
for spine in ax.spines.values():
    spine.set_linewidth(1)

fig.savefig("seq2.pdf", bbox_inches="tight")


fig.tight_layout()
plt.show()
