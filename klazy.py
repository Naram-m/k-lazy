import numpy as np


class KLAZYGD:
    """
    k-lazyGD with update:
        x_{t+1} = Π( x_{t - n_t} - g_{t - n_t : t} / σ ),  with  n_t = (t-1) mod k.
    """

    def __init__(self, T, k=1, G=1, R=1, d=2):
        self.d = d
        self.G = G
        self.R = R
        self.k = int(k)

        self.g_history = []
        self.actions = [np.zeros(self.d)]  # x_1 = 0

        self.cum_g_sum = np.zeros(self.d)
        self.sigma = np.sqrt(T)

    def project_onto_l1_ball(self, v, tau=1.0):
        if np.linalg.norm(v, 1) <= tau:
            return v.copy()
        if tau == 0:
            return np.zeros_like(v)

        u = np.abs(v)
        u_sorted = np.sort(u)[::-1]
        cssv = np.cumsum(u_sorted)
        rho = np.nonzero(u_sorted * np.arange(1, len(u) + 1) > (cssv - tau))[0][-1]
        theta = (cssv[rho] - tau) / (rho + 1)
        w = np.sign(v) * np.maximum(u - theta, 0.0)
        return w

    def project_onto_l2_ball(self, v, tau=1.0):
        nrm = np.linalg.norm(v, 2)
        if nrm <= tau or nrm == 0.0:
            return v.copy()
        return v * (tau / nrm)  # not v / nrm

    def update(self, ell2=False, fixed_sigma=False):
        if not fixed_sigma:
            # self.sigma = np.sqrt(len(self.actions))                     # For intro examples
            # self.sigma = np.sqrt(len(self.actions)) / np.sqrt(4 * 15)   # For stochastic sequences experiment
            # self.sigma = np.sqrt(len(self.actions)) / 4                 # For corrupted phases experiment
            self.sigma = np.sqrt(len(self.actions)) / np.sqrt(10)         # For worst-case P_T approx 5

        t = len(self.g_history)  # number of gradients observed so far
        if t == 0:
            base = self.actions[-1]
            g_window_sum = 0.0
        else:
            n_t = (t - 1) % self.k
            start = t - n_t  # "time" of the base point (1-based in math)
            base_idx = start  # x_{start} lives at actions[start]
            g_start_idx = start - 1  # g_{start} lives at g_history[start-1]

            base = self.actions[base_idx]
            g_window_sum = np.sum(self.g_history[g_start_idx:t], axis=0)

        unconst = base - (g_window_sum / self.sigma)

        action = (self.project_onto_l2_ball(unconst) if ell2
                  else self.project_onto_l1_ball(unconst))
        self.actions.append(action)
        return action, unconst

    def observe(self, g_t):
        self.g_history.append(g_t)
        if self.cum_g_sum is None:
            self.cum_g_sum = g_t.copy()
        else:
            self.cum_g_sum += g_t
