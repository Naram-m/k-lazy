import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# style
plt.style.use(['science','no-latex'])
plt.rcParams.update({
    "font.family": "STIXGeneral",   # widely available, has bold
})
# --- Load saved values ---
gd_hc_acc  = np.load("./gd_hc_acc.npy")
lgd_hc_acc = np.load("./lgd_hc_acc.npy")
comp_hc_acc = np.load("./u_hc_acc.npy")

# Load all KLAZY variants
k_values = [19]
klazy_hc_acc = {k: np.load(f"./klazy_hc_acc_k{k}.npy") for k in k_values}

# Convert to regret (subtract comparator)
gd_hc_acc  = gd_hc_acc  - comp_hc_acc
lgd_hc_acc = lgd_hc_acc - comp_hc_acc
for k in k_values:
    klazy_hc_acc[k] = klazy_hc_acc[k] - comp_hc_acc

T = len(gd_hc_acc)
t_vals = np.arange(1, T + 1)

plt.figure(figsize=(7, 5))  # small but visible



# --- Plot each KLAZY variant ---
markers = ['d', 'v', '^', '<']  # different markers for each k
for i, k in enumerate(k_values):
    plt.plot(
        t_vals, klazy_hc_acc[k],
        label=f"{k}-LazyGD",
        linewidth=2,
        marker=markers[i % len(markers)],
        markevery=600,
        color='magenta',
        alpha=0.8,
        markersize=8
    )

# --- Plot GD, LGD, Comparator (unchanged) ---
plt.plot(t_vals, gd_hc_acc, label="GD", linewidth=2.5, marker='o',
         markevery=600, color=(0, 0, 1, 0.6))
# plt.plot(t_vals, lgd_hc_acc, label="LazyGD", linewidth=2.5, marker='s',
#          markevery=600, color=(1, 0.5, 0, 0.8))

# --- Axis formatting ---
plt.xlabel("T", fontsize=15, fontweight='bold')
plt.ylabel("Dynamic regret", fontsize=15, fontweight='bold')
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')

# max_tick = (T // 1000) * 1000
max_tick = 3001

if max_tick >= 1000:
    xticks = np.arange(1, max_tick + 1, 500)
    labels = [str(int(x // 1000)) for x in xticks]   # '1','2',...
    plt.xticks(xticks, labels)
    plt.xlabel(r"T$(\times 10^3)$", fontsize=13, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(fontsize=13, frameon=False)  # slightly smaller to fit all
plt.tight_layout()

plt.savefig("hit.pdf", bbox_inches='tight')
# print(gd_hc_acc[-1])
# print(klazy_hc_acc[50][-1])

plt.show()