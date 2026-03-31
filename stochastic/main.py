import os
import numpy as np
from gd import GD
from lgd import LGD
from klazy import KLAZYGD


def generate_g(T, d=2, P=2, start_positive=True, var=10, seed=None):
    if seed is not None:
        np.random.seed(int(seed))
    std = np.sqrt(var)
    first = 1.0 if start_positive else -1.0
    mus = first * np.array([1 if i % 2 == 0 else -1 for i in range(P)], float)

    blocks = []
    for mu in mus:
        block = np.random.normal(loc=mu, scale=std, size=(int(T), int(d)))
        norms = np.linalg.norm(block, axis=1, keepdims=True, ord=2)
        block = block / np.maximum(norms, 1e-12)  # unit ℓ2 per row
        blocks.append(block)
    return np.vstack(blocks)


def comparator_sequence(T_phase, d=2, P=10, start_positive=True):
    s0 = 1 if start_positive else -1  # phase-0 mean sign
    signs = s0 * np.array([1 if i % 2 == 0 else -1 for i in range(P)])
    # base = np.ones(int(d), float) / float(d)  # equal coords summing to 1 in L1
    base = np.ones(int(d), float) / np.sqrt(d)  # equal coords summing to 1 in L2
    blocks = [np.tile(-s * base, (int(T_phase), 1)) for s in signs]
    return np.vstack(blocks)


if __name__ == "__main__":
    os.makedirs("./klazy_results_stoch", exist_ok=True)

    # --- experiment parameters (tweak these) ---
    T = 4000
    d = 5  # gradient/action dimension
    pos_len = 100  # length of positive blocks
    neg_len = 10  # length of negative blocks
    start_positive = True
    fixed_sigma = False

    # --- instantiate algorithms (same as your original) ---
    gd = GD(T, d=d)
    lgd = LGD(T, d=d)
    # multiple KLAZY variants
    k_values = [65, 150, 300, 1500]
    klazy_algs = {k: KLAZYGD(T, k=k, d=d) for k in k_values}

    # --- generate sequence ---
    g_t = generate_g(T, d=d, start_positive=start_positive, P=15, seed=42)
    comparator = comparator_sequence(T_phase=T, d=d, P=15, start_positive=start_positive)

    # --- accumulators and storage ---
    prev_gd_action = np.zeros(d)
    prev_lgd_action = np.zeros(d)
    prev_klazy_action = {k: np.zeros(d) for k in k_values}
    prev_u_t = np.zeros(d)

    gd_sc_acc = lgd_sc_acc = u_t_sc_acc = 0.0
    gd_hc_acc = lgd_hc_acc = u_t_hc_acc = 0.0
    klazy_sc_acc = {k: 0.0 for k in k_values}
    klazy_hc_acc = {k: 0.0 for k in k_values}

    gd_sc_acc_list, lgd_sc_acc_list, u_t_sc_acc_list = [], [], []
    gd_hc_acc_list, lgd_hc_acc_list, u_t_hc_acc_list = [], [], []

    klazy_sc_acc_lists = {k: [] for k in k_values}
    klazy_hc_acc_lists = {k: [] for k in k_values}

    gd_action_list, lgd_action_list = [], []
    klazy_action_lists = {k: [] for k in k_values}
    u_t_action_list = []

    # --- minimal header for prints ---
    print("t | cost | GD_action HC Δ | LGD_action HC Δ | KLAZY(k=50)_action HC Δ")

    # --- main loop ---
    for t in range(len(g_t)):
        # get actions for GD/LGD
        lgd_action, lgd_action_unconst = np.round(lgd.update(ell2=True, fixed_sigma=fixed_sigma), 5)
        gd_action, gd_action_unconst = np.round(gd.update(ell2=True, fixed_sigma=fixed_sigma), 5)

        # get actions for each KLAZY variant
        klazy_actions = {}
        for k in k_values:
            action, _ = klazy_algs[k].update(ell2=True, fixed_sigma=fixed_sigma)
            klazy_actions[k] = np.round(action, 5)

        u_t = np.round(comparator[t], 5)

        # save actions
        lgd_action_list.append(lgd_action)
        gd_action_list.append(gd_action)
        for k in k_values:
            klazy_action_lists[k].append(klazy_actions[k])
        u_t_action_list.append(u_t)

        # costs & observe
        cost = g_t[t]
        lgd.observe(cost)
        gd.observe(cost)
        for k in k_values:
            klazy_algs[k].observe(cost)

        # hitting costs (incremental)
        gd_hc = float(np.dot(gd_action, cost))
        lgd_hc = float(np.dot(lgd_action, cost))
        u_t_hc = float(np.dot(u_t, cost))
        klazy_hc_k = {k: float(np.dot(klazy_actions[k], cost)) for k in k_values}

        gd_hc_acc += gd_hc
        lgd_hc_acc += lgd_hc
        u_t_hc_acc += u_t_hc
        for k in k_values:
            klazy_hc_acc[k] += klazy_hc_k[k]

        gd_hc_acc_list.append(gd_hc_acc)
        lgd_hc_acc_list.append(lgd_hc_acc)
        u_t_hc_acc_list.append(u_t_hc_acc)
        for k in k_values:
            klazy_hc_acc_lists[k].append(klazy_hc_acc[k])

        # switching cost: ℓ2 norm between consecutive actions
        gd_diff = float(np.linalg.norm(np.array(gd_action) - np.array(prev_gd_action)))
        lgd_diff = float(np.linalg.norm(np.array(lgd_action) - np.array(prev_lgd_action)))
        u_t_diff = float(np.linalg.norm(np.array(u_t) - np.array(prev_u_t)))
        klazy_diff_k = {k: float(np.linalg.norm(np.array(klazy_actions[k]) - np.array(prev_klazy_action[k])))
                        for k in k_values}

        gd_sc_acc += gd_diff
        lgd_sc_acc += lgd_diff
        u_t_sc_acc += u_t_diff
        for k in k_values:
            klazy_sc_acc[k] += klazy_diff_k[k]

        gd_sc_acc_list.append(gd_sc_acc)
        lgd_sc_acc_list.append(lgd_sc_acc)
        u_t_sc_acc_list.append(u_t_sc_acc)
        for k in k_values:
            klazy_sc_acc_lists[k].append(klazy_sc_acc[k])

        # minimal per-step print (only show KLAZY for k=50)
        if t % 50 == 0:
            k_print = k_values[0]
            print(
                f"{t} | {np.round(cost, 3).tolist()} | "
                f"GD {gd_action.tolist()} HC={gd_hc:.4f} Δ={gd_diff:.3f} | "
                f"LGD {lgd_action.tolist()} HC={lgd_hc:.4f} Δ={lgd_diff:.3f} | "
                f"KLAZY(k={k_print}) {klazy_actions[k_print].tolist()} "
                f"HC={klazy_hc_k[k_print]:.4f} Δ={klazy_diff_k[k_print]:.3f}"
            )

        # update prev
        prev_gd_action = gd_action
        prev_lgd_action = lgd_action
        for k in k_values:
            prev_klazy_action[k] = klazy_actions[k]
        prev_u_t = comparator[t]

    # --- save results ---
    # GD/LGD/U (exactly as before)
    np.save("./klazy_results_stoch/gd_sc_acc.npy", np.array(gd_sc_acc_list))
    np.save("./klazy_results_stoch/lgd_sc_acc.npy", np.array(lgd_sc_acc_list))
    np.save("./klazy_results_stoch/u_sc_acc.npy", np.array(u_t_sc_acc_list))

    np.save("./klazy_results_stoch/gd_hc_acc.npy", np.array(gd_hc_acc_list))
    np.save("./klazy_results_stoch/lgd_hc_acc.npy", np.array(lgd_hc_acc_list))
    np.save("./klazy_results_stoch/u_hc_acc.npy", np.array(u_t_hc_acc_list))

    # KLAZY variants: suffix by k
    for k in k_values:
        np.save(f"./klazy_results_stoch/klazy_sc_acc_k{k}.npy", np.array(klazy_sc_acc_lists[k]))
        np.save(f"./klazy_results_stoch/klazy_hc_acc_k{k}.npy", np.array(klazy_hc_acc_lists[k]))
