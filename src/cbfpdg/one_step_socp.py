import cvxpy as cp
import numpy as np


def _diagnose_infeasibility(Lfh, Lgh, alpha_h, LfV, LgV, gamma_V, rho2):
    """
    Analytically check which constraints are individually infeasible within
    the trust region ||u|| <= rho2.

    For the CBF constraint  Lfh + Lgh @ u >= -alpha_h:
        max over ||u||<=rho2 of LHS = Lfh + ||Lgh|| * rho2
        Infeasible alone if this max < -alpha_h.

    For the CLF constraint  LfV + LgV @ u <= -gamma_V:
        min over ||u||<=rho2 of LHS = LfV - ||LgV|| * rho2
        Infeasible alone if this min > -gamma_V.
    """
    cbf_max = Lfh + np.linalg.norm(Lgh) * rho2
    clf_min = LfV - np.linalg.norm(LgV) * rho2

    cbf_alone = cbf_max < -alpha_h
    clf_alone  = clf_min > -gamma_V

    msgs = []
    if cbf_alone:
        msgs.append(
            f"  CBF: max(Lfh + Lgh@u) = {cbf_max:.3f} < -alpha_h = {-alpha_h:.3f}"
        )
    if clf_alone:
        msgs.append(
            f"  CLF: min(LfV + LgV@u) = {clf_min:.3f} > -gamma_V = {-gamma_V:.3f}"
        )
    if not cbf_alone and not clf_alone:
        msgs.append("  CBF and CLF are individually feasible but conflict with each other.")

    print("Infeasibility diagnosis:")
    for m in msgs:
        print(m)


def solve_socp(Lfh, Lgh, alpha_h,
               LfV, LgV, gamma_V,
               rho2,
               u_dim,
               clf_penalty=1):
    """
    Solve:
        min ||u|| + clf_penalty * delta
        s.t.
            Lfh + Lgh @ u >= -alpha_h
            LfV + LgV @ u <= -gamma_V + delta
            delta >= 0
            ||u|| <= rho2

    The CLF constraint is soft: if it is infeasible the slack delta absorbs
    the violation and the controller saturates against the trust-region bound.
    """

    # Decision variables
    u = cp.Variable(u_dim)
    delta = cp.Variable(nonneg=True)  # CLF slack

    # Objective
    objective = cp.Minimize(cp.norm(u, 2) + clf_penalty * delta)

    # Constraints
    constraints = [
        Lfh + Lgh @ u >= -alpha_h,
        LfV + LgV @ u <= -gamma_V + delta,
        cp.norm(u, 2) <= rho2
    ]

    # Problem
    prob = cp.Problem(objective, constraints)

    # Solve
    prob.solve(solver=cp.ECOS)  # ECOS or SCS

    if prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        # Only the CBF constraint can make the problem infeasible now
        _diagnose_infeasibility(Lfh, Lgh, alpha_h, LfV, LgV, gamma_V, rho2)

    if delta.value is not None and delta.value > 0:
        print(f"CLF slack delta = {delta.value:.4f} (CLF constraint relaxed)")

    return u.value, prob.status, delta.value