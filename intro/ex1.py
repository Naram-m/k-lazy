from gd import GD
from lgd import LGD
import numpy as np
from klazy import KLAZYGD

T = 100
gd = GD(T)
lgd = LGD(T)
klazy = KLAZYGD(T, k=5)

fixed_sigma = True # make true for plotting points.
g_t = []
g_t.extend([np.array([-1, 0], dtype=float) for _ in range(12)])

cycle = [np.array([1, 0], dtype=float),
         np.array([0, 1], dtype=float),
         np.array([-1, 0], dtype=float),
         np.array([0, -1], dtype=float)]

g_t.extend(cycle[i % 4].copy() for i in range(89))


prev_gd_action = np.zeros(2)
prev_lgd_action = np.zeros(2)
prev_klazy_action = np.zeros(2)
gd_sc_acc = 0
lgd_sc_acc = 0
klazy_sc_acc = 0
gd_hc_acc = 0
lgd_hc_acc = 0
klazy_hc_acc = 0
lgd_sc_acc_list = []
gd_sc_acc_list = []
klazy_sc_acc_list = []
lgd_hc_acc_list = []
gd_hc_acc_list = []
klazy_hc_acc_list = []

# actions
lgd_action_list = []
gd_action_list = []
lgd_actions_unconst_list = []
gd_actions_unconst_list = []

klazy_action_list = []
klazy_actions_unconst_list = []

for t in range(101):
    lgd_action, lgd_action_unconst = np.round(lgd.update(ell2=False, fixed_sigma=fixed_sigma), 5)
    lgd_action_list.append(lgd_action)
    lgd_actions_unconst_list.append(lgd_action_unconst)

    gd_action, gd_action_unconst = np.round(gd.update(ell2=False, fixed_sigma=fixed_sigma), 5)
    gd_action_list.append(gd_action)
    gd_actions_unconst_list.append(gd_action_unconst)

    klazy_action, klazy_action_unconst = np.round(klazy.update(ell2=False, fixed_sigma=fixed_sigma), 5)
    klazy_action_list.append(klazy_action)
    klazy_actions_unconst_list.append(klazy_action_unconst)

    lgd.observe(g_t[t])
    gd.observe(g_t[t])
    klazy.observe(g_t[t])

    print(f"Step {t}")
    print(f"  cost:  {g_t[t]}")
    print(f"  GD action:  {gd_action}")
    print(f"  LGD action: {lgd_action}")
    print(f"  KLAZY action: {klazy_action}")
    # print(f"  GD acc hitting cost:  {gd_hc_acc:.3f}")
    # print(f"  LGD acc hitting cost: {lgd_hc_acc:.3f}")
    gd_hc_acc += gd_action @ g_t[t]
    gd_hc_acc_list.append(gd_hc_acc)

    lgd_hc_acc += lgd_action @ g_t[t]
    lgd_hc_acc_list.append(lgd_hc_acc)

    klazy_hc_acc += klazy_action @ g_t[t]
    klazy_hc_acc_list.append(klazy_hc_acc)

    print(f"    hitting cost GD: {gd_action @ g_t[t]}")
    print(f"    hitting cost LGD: {lgd_action @ g_t[t]}")

    # Compute and print ℓ₂ norm of action difference
    if prev_gd_action is not None:
        gd_diff = np.linalg.norm(np.array(gd_action) - np.array(prev_gd_action))
        gd_sc_acc += gd_diff
        gd_sc_acc_list.append(gd_sc_acc)

        lgd_diff = np.linalg.norm(np.array(lgd_action) - np.array(prev_lgd_action))
        lgd_sc_acc += lgd_diff
        lgd_sc_acc_list.append(lgd_sc_acc)

        klazy_diff = np.linalg.norm(np.array(klazy_action) - np.array(prev_klazy_action))
        klazy_sc_acc += klazy_diff
        klazy_sc_acc_list.append(klazy_sc_acc)

        print(f"  GD Δ norm:  {gd_diff:.3f}")
        print(f"  LGD Δ norm: {lgd_diff:.3f}")
        print(f"  KLAZY Δ norm: {klazy_diff:.3f}")

        print(f"  GD SC acc:  {gd_sc_acc:.3f}")
        print(f"  LGD SC acc: {lgd_sc_acc:.3f}")
        print(f"  KLAZY Δ norm: {klazy_sc_acc:.3f}")

    else:
        print("  Initial step, no Δ norm.")

    # Store current actions as previous for next round
    prev_gd_action = gd_action
    prev_lgd_action = lgd_action
    prev_klazy_action = klazy_action


# np.save("ex1_results/gd_sc_acc.npy", np.array(gd_sc_acc_list))
# np.save("ex1_results/lgd_sc_acc.npy", np.array(lgd_sc_acc_list))
# np.save("ex1_results/klazy_sc_acc.npy", np.array(klazy_sc_acc_list))
# np.save("ex1_results/gd_hc_acc.npy", np.array(gd_hc_acc_list))
# np.save("ex1_results/lgd_hc_acc.npy", np.array(lgd_hc_acc_list))
# np.save("ex1_results/klazy_hc_acc.npy", np.array(klazy_hc_acc_list))
#
#
#
'''
only save and plot points when sigma is fixed (actions snapshot):
'''
# np.savez("ex1_results/gd_points.npz", actions=np.array(gd_action_list), unconst=np.array(gd_actions_unconst_list))
# np.savez("ex1_results/lgd_points.npz", actions=np.array(lgd_action_list), unconst=np.array(lgd_actions_unconst_list))
