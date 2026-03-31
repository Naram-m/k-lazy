import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import numpy as np

# --- Parameters ---
phase_len = 20
T = 2 * phase_len
time = np.arange(T)

# Base signal: Phase 1 mostly +1, Phase 2 mostly -1
signal = np.ones(T, dtype=float)
signal[phase_len:] = -1

# Outliers: three adjacent indices centered in each phase
mid1 = phase_len // 2
mid2 = phase_len + phase_len // 2
burst1_idx = np.array([mid1 - 1, mid1, mid1 + 1])     # e.g., 9,10,11
burst2_idx = np.array([mid2 - 1, mid2, mid2 + 1])     # e.g., 29,30,31
bursts = np.concatenate([burst1_idx, burst2_idx])

# Break the main line at outliers (no sloped transitions)
main_signal = signal.copy()
main_signal[bursts] = np.nan

# Opposite-sign values for outliers
outlier_values = np.where(signal[bursts] == 1, -1, 1)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 2.8))

# Higher-contrast phase shading
ax.axvspan(0, phase_len, color="gray", alpha=0.05)   # Phase 1
ax.axvspan(phase_len, T, color="gray", alpha=0.2)   # Phase 2

# Main signal in brown
ax.plot(time, main_signal, color="#8B4513", linewidth=2.5, solid_capstyle="round")

# Outliers as red "x" markers
ax.scatter(burst1_idx, outlier_values[:3], s=60, marker="x", color="red", zorder=4, linewidths=1.5)
ax.scatter(burst2_idx, outlier_values[3:], s=60, marker="x", color="red", zorder=4, linewidths=1.5)

# Axes: bold ticks, labels, and spines
ax.axhline(0, color="black", linewidth=1)
ax.set_yticks([-1, 1], ["-1", "+1"])
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlabel("T", fontsize=13, fontweight="bold")
ax.set_ylabel("Signal", fontsize=13, fontweight="bold")
ax.tick_params(axis='both', which='both', labelsize=12, width=1.5, length=6)
for spine in ax.spines.values():
    spine.set_linewidth(1.1)

# Limits and layout
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, T - 1)
fig.tight_layout()

fig.savefig("seq1.pdf", bbox_inches="tight")


plt.show()
