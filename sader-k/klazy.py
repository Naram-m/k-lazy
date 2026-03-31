import numpy as np


class KLAZYGD:
    """
    k-lazyGD with update:
        x_{t+1} = Π( x_{t - n_t} - g_{t - n_t : t} / σ ),  with  n_t = (t-1) mod k.
    """

    def __init__(self, sigma, k=1, G=1, R=1, d=2):
        self.d = d
        self.G = G
        self.R = R
        self.k = int(k)

        self.g_history = []
        self.actions = [np.zeros(self.d)]  # x_1 = 0

        self.cum_g_sum = np.zeros(self.d)
        self.sigma = sigma

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
        return v * (tau / nrm)

    def update(self, ell2=False):
        t = len(self.g_history)  # gradients observed so far

        if t == 0:
            action = self.actions[-1]  # x_1 already stored
            unconst = action
            return action, unconst  # do NOT append again

        n_t = (t - 1) % self.k
        start = t - n_t  # 1-based math index

        base = self.actions[start - 1]  # x_start
        g_window_sum = np.sum(self.g_history[start - 1:t], axis=0)  # g_start,...,g_t

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
