from gd import GD
from lgd import LGD
import numpy as np

T = 101

gd = GD(T)
lgd = LGD(T)
g_t = []
fixed_sigma = True # make true for plotting points
np.random.seed(45)
for t in range(1, 12):
    coord_1 = -1
    coord_2 = 0
    g_t.append(np.array([coord_1, coord_2]))


for t in range(12, T + 1):
    coord_1 = -1
    coord_2 = (-1)**t
    g_t.append(np.array([coord_1, coord_2]))

g_t = np.array(g_t)  # shape (T, 2)

prev_gd_action = np.zeros(2)
prev_lgd_action = np.zeros(2)
gd_sc_acc = 0
lgd_sc_acc = 0
gd_hc_acc = 0
lgd_hc_acc = 0
lgd_sc_acc_list=[]
gd_sc_acc_list=[]

lgd_hc_acc_list=[]
gd_hc_acc_list=[]


# actions
lgd_action_list = []
gd_action_list = []
lgd_actions_unconst_list = []
gd_actions_unconst_list = []

for t in range(T):
    lgd_action, lgd_action_unconst = np.round(lgd.update(ell2=True, fixed_sigma=fixed_sigma), 5)
    lgd_action_list.append(lgd_action)
    lgd_actions_unconst_list.append(lgd_action_unconst)

    gd_action, gd_action_unconst = np.round(gd.update(ell2=True, fixed_sigma=fixed_sigma), 5)
    gd_action_list.append(gd_action)
    gd_actions_unconst_list.append(gd_action_unconst)

    lgd.observe(g_t[t])
    gd.observe(g_t[t])

    print(f"Step {t}")
    print(f"  cost:  {g_t[t]}")
    print(f"  GD action:  {gd_action}")
    print(f"  LGD action: {lgd_action}")
    print(f"  GD acc hitting cost:  {gd_hc_acc:.3f}")
    print(f"  LGD acc hitting cost: {lgd_hc_acc:.3f}")
    gd_hc_acc += gd_action@g_t[t]
    gd_hc_acc_list.append(gd_hc_acc)
    lgd_hc_acc += lgd_action@g_t[t]
    lgd_hc_acc_list.append(lgd_hc_acc)
    # Compute and print ℓ₂ norm of action difference
    if prev_gd_action is not None:
        gd_diff = np.linalg.norm(np.array(gd_action) - np.array(prev_gd_action))
        gd_sc_acc += gd_diff
        gd_sc_acc_list.append(gd_sc_acc)
        lgd_diff = np.linalg.norm(np.array(lgd_action) - np.array(prev_lgd_action))
        lgd_sc_acc += lgd_diff
        lgd_sc_acc_list.append(lgd_sc_acc)
        print(f"  GD Δ norm:  {gd_diff:.3f}")
        print(f"  LGD Δ norm: {lgd_diff:.3f}")
        print(f"  GD SC acc:  {gd_sc_acc:.3f}")
        print(f"  LGD SC acc: {lgd_sc_acc:.3f}")
    else:
        print("  Initial step, no Δ norm.")

    # Store current actions as previous for next round
    prev_gd_action = gd_action
    prev_lgd_action = lgd_action


# np.save("ex2_results/gd_sc_acc_ell2.npy", np.array(gd_sc_acc_list))
# np.save("ex2_results/lgd_sc_acc_ell2.npy", np.array(lgd_sc_acc_list))
# np.save("ex2_results/gd_hc_acc_ell2.npy", np.array(gd_hc_acc_list))
# np.save("ex2_results/lgd_hc_acc_ell2.npy", np.array(lgd_hc_acc_list))
'''
only save and plot points when sigma is fixed (actions snapshot):
'''
# np.savez("ex2_results/gd_points_ell2.npz", actions=np.array(gd_action_list), unconst=np.array(gd_actions_unconst_list))
# np.savez("ex2_results/lgd_points_ell2.npz", actions=np.array(lgd_action_list), unconst=np.array(lgd_actions_unconst_list))
