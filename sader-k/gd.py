import numpy as np


class GD:
    def __init__(self, sigma, d=2):
        self.d = d
        self.g_history = [np.zeros(self.d)]
        self.actions = [np.zeros(self.d)]
        self.cum_g_sum = np.zeros(self.d)
        self.sigma = sigma

    def project_onto_l1_ball(self, v, tau=1.0):
        # Check if v is already in the l1 ball
        if np.linalg.norm(v, 1) <= tau:
            return v.copy()

        # Compute the projection
        u = np.abs(v)
        if tau == 0:
            return np.zeros_like(v)

        # Sort u in descending order
        u_sorted = np.sort(u)[::-1]
        cssv = np.cumsum(u_sorted)
        rho = np.nonzero(u_sorted * np.arange(1, len(u) + 1) > (cssv - tau))[0][-1]
        theta = (cssv[rho] - tau) / (rho + 1)

        # Perform the shrinkage
        w = np.sign(v) * np.maximum(u - theta, 0)
        return w

    def project_onto_l2_ball(self, v, tau=1.0):
        nrm = np.linalg.norm(v, 2)
        if nrm <= tau or nrm == 0.0:
            return v.copy()
        return v * (tau / nrm)

    def update(self, ell2=False):

        unconst = self.actions[-1] - 1 * self.g_history[-1] / self.sigma
        if ell2:
            action = self.project_onto_l2_ball(unconst)
        else:
            action = self.project_onto_l1_ball(unconst)
        self.actions.append(action)
        return action, unconst

    def observe(self, g_t):
        """
        Call this *after* receiving the actual gradient.
        """
        # 1) Append to history
        self.g_history.append(g_t)

        # 2) Update running cumulative sum of gradients
        if self.cum_g_sum is None:
            self.cum_g_sum = g_t.copy()
        else:
            self.cum_g_sum += g_t