import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# style
plt.style.use(['science','no-latex'])
plt.rcParams.update({
    "font.family": "STIXGeneral",   # widely available, has bold
})
# --- Load saved values ---
gd_sc_acc  = np.load("gd_sc_acc.npy")
lgd_sc_acc = np.load("lgd_sc_acc.npy")
comp_sc_acc = np.load("u_sc_acc.npy")

# Load all KLAZY variants
k_values = [65, 150, 300, 1500]
klazy_sc_acc = {k: np.load(f"klazy_sc_acc_k{k}.npy") for k in k_values}

T = len(gd_sc_acc)
t_vals = np.arange(1, T + 1)

plt.figure(figsize=(7, 5))  # small but visible



# --- Plot each KLAZY variant ---
markers = ['d', 'v', '^', '<']  # different markers for each k
for i, k in enumerate(k_values):
    plt.plot(
        t_vals, klazy_sc_acc[k],
        label=f"{k}-LazyGD",
        linewidth=2,
        marker=markers[i % len(markers)],
        markevery=3000,
        color='magenta',
        alpha=0.8,
        markersize=8
    )

# --- Plot GD, LGD, Comparator (unchanged) ---
plt.plot(t_vals, gd_sc_acc, label="GD", linewidth=2.5, marker='o',
         markevery=3000, color=(0, 0, 1, 0.6))
plt.plot(t_vals, lgd_sc_acc, label="LazyGD", linewidth=2.5, marker='s',
         markevery=3000, color=(1, 0.5, 0, 0.8))
plt.plot(t_vals, comp_sc_acc, label="Comparator", linewidth=2.5, marker='x', markersize=12,
         markevery=3000, color='brown', alpha=1)

# --- Axis formatting ---
plt.xlabel("T", fontsize=13, fontweight='bold')
plt.ylabel("Switching Cost", fontsize=15, fontweight='bold')
plt.xticks(fontsize=13, fontweight='bold')
# max_tick = (T // 1000) * 1000
max_tick = 61001

if max_tick >= 1000:
    xticks = np.arange(1000, max_tick + 1, 4000)
    labels = [str(int(x // 1000)) for x in xticks]   # '1','2',...
    plt.xticks(xticks, labels)
    plt.xlabel(r"T$(\times 10^3)$", fontsize=15, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=13, frameon=False)  # slightly smaller to fit all
plt.tight_layout()

plt.savefig("swi.pdf", bbox_inches='tight')
print(gd_sc_acc[-1])
print(klazy_sc_acc[65][-1])
print(">>" , comp_sc_acc[-1])
plt.show()