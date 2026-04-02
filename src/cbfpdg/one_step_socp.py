import cvxpy as cp
import numpy as np

def solve_socp(Lfh, Lgh, alpha_h,
               LfV, LgV, gamma_V,
               rho2,
               u_dim):
    """
    Solve:
        min ||u||
        s.t.
            Lfh + Lgh @ u >= -alpha_h
            LfV + LgV @ u <= -gamma_V
            ||u|| <= rho2
    """

    # Decision variable
    u = cp.Variable(u_dim)

    # Objective
    objective = cp.Minimize(cp.norm(u, 2))

    # Constraints
    constraints = [
        Lfh + Lgh @ u >= -alpha_h,
        LfV + LgV @ u <= -gamma_V,
        cp.norm(u, 2) <= rho2
    ]

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solve
    prob.solve(solver=cp.ECOS)  # ECOS or SCS

    return u.value, prob.status