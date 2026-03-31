import numpy as np


class LGD:
    def __init__(self, T, d=2):
        self.d = d
        self.g_history = []
        self.actions = []
        self.cum_g_sum = np.zeros(self.d)
        self.sigma = 1 * np.sqrt(T)

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
        if np.linalg.norm(v, 2) <= tau:
            return v.copy()
        return v / (np.linalg.norm(v, 2))

    def update(self, ell2=False, fixed_sigma=False):

        if not self.actions:
            action = np.zeros(self.d)
            self.actions = [0]
            unconst = action
        else:
            if not fixed_sigma:
                # self.sigma = np.sqrt(len(self.actions))  # For intro examples
                # self.sigma = np.sqrt(len(self.actions)) / np.sqrt(4 * 15)  # For stochastic sequences experiment
                # self.sigma = np.sqrt(len(self.actions)) / 4  # For corrupted phases experiment
                self.sigma = np.sqrt(len(self.actions)) / np.sqrt(10)         # For worst-case P_T approx 5

            unconst = -1 * self.cum_g_sum / self.sigma
            if ell2:
                action = self.project_onto_l2_ball(unconst)
            else:
                action = self.project_onto_l1_ball(unconst)
        self.actions.append(action)
        return (action, unconst)

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