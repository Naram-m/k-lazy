import numpy as np
import matplotlib.pyplot as plt
import scienceplots
# style
plt.style.use(['science','no-latex'])
plt.rcParams.update({
    "font.family": "STIXGeneral",
})

# Load saved values
gd_sc_acc = np.load("gd_sc_acc_ell2.npy")
lgd_sc_acc = np.load("lgd_sc_acc_ell2.npy")


T = len(gd_sc_acc)
t_vals = np.arange(1, T + 1)

plt.figure(figsize=(4, 3))  # small but visible
plt.plot(t_vals, gd_sc_acc, label="GD", linewidth=2.5, marker='o', markevery=20, color=(0, 0, 1, 0.6)    )
plt.plot(t_vals, lgd_sc_acc, label="LazyGD", linewidth=2.5, marker='s', markevery=20, color=(1, 0.5, 0, 0.6))


plt.xlabel(r"$t$", fontsize=13, fontweight='bold')
plt.ylabel("Switching Cost", fontsize=13, fontweight='bold')
plt.xticks( fontsize=12, fontweight='bold')
plt.yticks(np.arange(0, 14.1, 2),fontsize=12, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14, frameon=False)
plt.tight_layout()
plt.savefig("sc_acc_ell2.pdf", bbox_inches='tight')  # save first
plt.show()
