import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# style
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({"font.family": "STIXGeneral"})

# --- Load saved values ---
sader_sc_acc    = np.load("sader_sc_acc.npy")
sader_k_sc_acc  = np.load("sader_k_sc_acc.npy")
comp_sc_acc     = np.load("u_sc_acc.npy")

sader_hc_acc    = np.load("sader_hc_acc.npy")
sader_k_hc_acc  = np.load("sader_k_hc_acc.npy")
comp_hc_acc     = np.load("u_hc_acc.npy")

# --- Regret components ---
sader_switch_regret   = (sader_sc_acc   - comp_sc_acc)
sader_k_switch_regret = (sader_k_sc_acc - comp_sc_acc)

sader_hit_regret      = (sader_hc_acc   - comp_hc_acc)
sader_k_hit_regret    = (sader_k_hc_acc - comp_hc_acc)

sader_total_regret    = sader_switch_regret   + sader_hit_regret
sader_k_total_regret  = sader_k_switch_regret + sader_k_hit_regret

T = len(sader_total_regret)
t_vals = np.arange(1, T + 1)

# --- Shared plotting params (same colors/styles everywhere) ---
SADER_STYLE = dict(label="SAder",   linewidth=2.5, marker='o', markevery=5000, color=(0, 0, 1, 0.6))
SADERK_STYLE= dict(label="SAder-k", linewidth=2.5, marker='s', markevery=5000, color='magenta')

def apply_common_axis_format(ylabel):
    plt.xlabel("T", fontsize=13, fontweight='bold')
    plt.ylabel(ylabel, fontsize=15, fontweight='bold')
    plt.xticks(fontsize=13, fontweight='bold')
    plt.yticks(fontsize=13, fontweight='bold')

    max_tick = 61001
    if max_tick >= 1000:
        xticks = np.arange(1000, max_tick + 1, 4000)
        labels = [str(int(x // 1000)) for x in xticks]
        plt.xticks(xticks, labels)
        plt.xlabel(r"T$(\times 10^3)$", fontsize=15, fontweight='bold')

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=13, frameon=False)
    plt.tight_layout()

def plot_two_curves(y_sader, y_saderk, ylabel, title=None):
    plt.figure(figsize=(7, 5))
    plt.plot(t_vals, y_sader, **SADER_STYLE)
    plt.plot(t_vals, y_saderk, **SADERK_STYLE)
    apply_common_axis_format(ylabel)
    if title is not None:
        pass
        # plt.title(title, fontsize=14, fontweight='bold')
    # plt.show()
    plt.savefig(title+".pdf", bbox_inches='tight')

# --- 1) Hitting cost regret ---
plot_two_curves(sader_hit_regret, sader_k_hit_regret, ylabel="Dynamic Regret", title="Hit_regret")

# --- 2) Switching cost regret ---
plot_two_curves(sader_switch_regret, sader_k_switch_regret, ylabel="Switching Cost", title="Switch_regret")

# --- 3) Total regret ---
plot_two_curves(sader_total_regret, sader_k_total_regret, ylabel="Total Regret", title="Total_regret")

print("Final SADER hit regret     :", sader_hit_regret[-1])
print("Final SADER-K hit regret   :", sader_k_hit_regret[-1])
print("Final SADER switch regret  :", sader_switch_regret[-1])
print("Final SADER-K switch regret:", sader_k_switch_regret[-1])
print("Final SADER total regret   :", sader_total_regret[-1])
print("Final SADER-K total regret :", sader_k_total_regret[-1])