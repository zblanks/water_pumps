import cvxpy as cp


class WaterPointRepair:
    def __init__(self, y, p, c, δ):
        n = len(y)
        self.z = cp.Variable(n, boolean=True)
        w = cp.Variable(n, boolean=True)
        t = cp.Variable(n)

        self.B = cp.Parameter(nonneg=True)
        self.Γ = cp.Parameter(nonneg=True)

        constraints = []
        constraints += [t[i] == self.z[i] + w[i]*y[i] for i in range(n)]
        constraints += [self.z[i] + w[i] == 1 for i in range(n)]

        expr1 = cp.sum(cp.multiply(c, self.z))
        expr2 = self.Γ * cp.norm(cp.multiply(δ, self.z), p=1)
        constraints += [expr1 + expr2 <= self.B]

        objective = cp.Maximize(cp.sum(cp.multiply(p, t)))
        self._problem = cp.Problem(objective, constraints)

    def solve(self):
        self._problem.solve(solver=cp.MOSEK, verbose=True)
        return self.z.value
