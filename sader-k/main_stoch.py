import os
import numpy as np
from gd import GD
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

    # --- generate sequence ---
    g_t = generate_g(T, d=d, start_positive=start_positive, P=15, seed=42)
    comparator = comparator_sequence(T_phase=T, d=d, P=15, start_positive=start_positive)

    # --- instantiate algorithms (same as your original) ---
    eta_values = [1 / np.sqrt(len(g_t)), 2 / np.sqrt(len(g_t)), 4 / np.sqrt(len(g_t)), 8 / np.sqrt(len(g_t)),
                  16 / np.sqrt(len(g_t))]
    gd_algs = {eta: GD(sigma=(1/eta), d=d) for eta in eta_values}

    # multiple KLAZY variants
    # k_values = [65, 150, 300, 1500]
    # k_values = [30, 60, 125, 250, 500]
    k_values = [500, 250, 125, 60, 30]


    klazy_algs = {
        k: KLAZYGD(sigma=(1 / eta), k=k, d=d)
        for k, eta in zip(k_values, eta_values)
    }
    # --- accumulators and storage ---
    prev_gd_actions = {eta: np.zeros(d) for eta in eta_values}
    prev_klazy_action = {k: np.zeros(d) for k in k_values}
    prev_u_t = np.zeros(d)

    u_t_sc_acc = 0.0
    u_t_hc_acc = 0.0

    gd_sc_acc = {eta: 0.0 for eta in eta_values}
    gd_hc_acc = {eta: 0.0 for eta in eta_values}

    klazy_sc_acc = {k: 0.0 for k in k_values}
    klazy_hc_acc = {k: 0.0 for k in k_values}

    u_t_sc_acc_list = []
    u_t_hc_acc_list = []

    gd_sc_acc_lists = {eta: [] for eta in eta_values}
    gd_hc_acc_lists = {eta: [] for eta in eta_values}
    klazy_sc_acc_lists = {k: [] for k in k_values}
    klazy_hc_acc_lists = {k: [] for k in k_values}

    gd_action_lists = {eta: [] for eta in eta_values}
    klazy_action_lists = {k: [] for k in k_values}
    u_t_action_list = []

    #=================================Meta Learning part=================================#
    beta = 2 * np.sqrt(2) / 3 * np.sqrt(5) * 1 / np.sqrt(len(g_t))

    prev_sader_action = np.zeros(d)
    prev_sader_k_action = np.zeros(d)

    sader_sc_acc = 0.0
    sader_hc_acc = 0.0
    sader_k_sc_acc = 0.0
    sader_k_hc_acc = 0.0

    sader_sc_acc_list = []
    sader_hc_acc_list = []
    sader_k_sc_acc_list = []
    sader_k_hc_acc_list = []

    sader_action_list = []
    sader_k_action_list = []

    sader_weights = np.ones(5)/5
    sader_k_weights = np.ones(5)/5


    #==================================================================#

    # --- main loop ---
    for t in range(len(g_t)):
        # get actions for each gd variant
        gd_actions = {}
        for eta in eta_values:
            action, _ = gd_algs[eta].update(ell2=True)
            gd_actions[eta] = np.round(action, 5)

        # get actions for each KLAZY variant
        klazy_actions = {}
        for k in k_values:
            action, _ = klazy_algs[k].update(ell2=True)
            klazy_actions[k] = np.round(action, 5)
        u_t = np.round(comparator[t], 5)

        # Meta-learning:
        sader_action = sum(w * gd_actions[eta] for w, eta in zip(sader_weights, eta_values))
        sader_k_action = sum(w * klazy_actions[k] for w, k in zip(sader_k_weights, k_values))


        # save actions
        for eta in eta_values:
            gd_action_lists[eta].append(gd_actions[eta])
        for k in k_values:
            klazy_action_lists[k].append(klazy_actions[k])
        u_t_action_list.append(u_t)

        sader_action_list.append(sader_action)
        sader_k_action_list.append(sader_k_action)

        # costs & observe
        cost = g_t[t]
        for eta in eta_values:
            gd_algs[eta].observe(cost)
        for k in k_values:
            klazy_algs[k].observe(cost)

        # hitting costs (incremental)
        u_t_hc = float(np.dot(u_t, cost))
        gd_hc_eta = {eta: float(np.dot(gd_actions[eta], cost)) for eta in eta_values}
        klazy_hc_k = {k: float(np.dot(klazy_actions[k], cost)) for k in k_values}
        sader_hc = float(np.dot(sader_action, cost))
        sader_k_hc = float(np.dot(sader_k_action, cost))

        u_t_hc_acc += u_t_hc
        sader_hc_acc += sader_hc
        sader_k_hc_acc += sader_k_hc
        for eta in eta_values:
            gd_hc_acc[eta] += gd_hc_eta[eta]
        for k in k_values:
            klazy_hc_acc[k] += klazy_hc_k[k]

        for eta in eta_values:
            gd_hc_acc_lists[eta].append(gd_hc_acc[eta])
        u_t_hc_acc_list.append(u_t_hc_acc)
        sader_hc_acc_list.append(sader_hc_acc)
        sader_k_hc_acc_list.append(sader_k_hc_acc)
        for k in k_values:
            klazy_hc_acc_lists[k].append(klazy_hc_acc[k])

        # switching cost: ℓ2 norm between consecutive actions
        u_t_diff = float(np.linalg.norm(np.array(u_t) - np.array(prev_u_t)))
        klazy_diff_k = {k: float(np.linalg.norm(np.array(klazy_actions[k]) - np.array(prev_klazy_action[k])))
                        for k in k_values}
        gd_diff_eta = {eta: float(np.linalg.norm(np.array(gd_actions[eta]) - np.array(prev_gd_actions[eta])))
                       for eta in eta_values}

        sader_diff = float(np.linalg.norm(np.array(sader_action) - np.array(prev_sader_action)))
        sader_k_diff = float(np.linalg.norm(np.array(sader_k_action) - np.array(prev_sader_k_action)))

        sader_sc_acc += sader_diff
        sader_k_sc_acc += sader_k_diff
        u_t_sc_acc += u_t_diff
        for eta in eta_values:
            gd_sc_acc[eta] += gd_diff_eta[eta]
        for k in k_values:
            klazy_sc_acc[k] += klazy_diff_k[k]
        ####################################################
        u_t_sc_acc_list.append(u_t_sc_acc)
        sader_sc_acc_list.append(sader_sc_acc)
        sader_k_sc_acc_list.append(sader_k_sc_acc)

        for eta in eta_values:
            gd_sc_acc_lists[eta].append(gd_sc_acc[eta])
        for k in k_values:
            klazy_sc_acc_lists[k].append(klazy_sc_acc[k])

        # minimal per-step print (only show KLAZY for k=50)

        # update prev
        for eta in eta_values:
            prev_gd_actions[eta] = gd_actions[eta]

        for k in k_values:
            prev_klazy_action[k] = klazy_actions[k]
        prev_u_t = comparator[t]
        prev_sader_action = sader_action
        prev_sader_k_action = sader_k_action

        # update weights for sader
        losses = np.array([gd_hc_eta[eta] + gd_diff_eta[eta] for eta in eta_values])
        unnormalized = sader_weights * np.exp(-beta * losses)  # w_t^η e^{-β ℓ_t^η}
        sader_weights = unnormalized / unnormalized.sum()  # normalize to get w_{t+1}^η

        # update weights for sader-k
        losses_sader_k = np.array([klazy_hc_k[k] + klazy_diff_k[k] for k in k_values])
        unnormalized_sader_k = sader_k_weights * np.exp(-beta * losses_sader_k)  # w_t^η e^{-β ℓ_t^η}
        sader_k_weights = unnormalized_sader_k / unnormalized_sader_k.sum()  # normalize to get w_{t+1}^η

        if (t) % 500 == 0:
            print(f"\n=== t = {t + 1} ===")
            print("sader_weights   =", np.round(sader_weights, 3))
            print("sader_k_weights =", np.round(sader_k_weights, 3))

    # --- save results ---
    # GD/LGD/U (exactly as before)
    np.save("./klazy_results_stoch/u_sc_acc.npy", np.array(u_t_sc_acc_list))
    np.save("./klazy_results_stoch/u_hc_acc.npy", np.array(u_t_hc_acc_list))

    np.save("./klazy_results_stoch/sader_sc_acc.npy", np.array(sader_sc_acc_list))
    np.save("./klazy_results_stoch/sader_hc_acc.npy", np.array(sader_hc_acc_list))

    np.save("./klazy_results_stoch/sader_k_sc_acc.npy", np.array(sader_k_sc_acc_list))
    np.save("./klazy_results_stoch/sader_k_hc_acc.npy", np.array(sader_k_hc_acc_list))


