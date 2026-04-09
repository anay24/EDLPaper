import cvxpy as cp
import numpy as np
from scipy.optimize import NonlinearConstraint, shgo


def _diagnose_infeasibility(Lfh, Lgh, alpha_h, LfV, LgV, gamma_V, rho1):
    """
    Analytically check which constraints are individually infeasible within
    the trust region ||u|| <= rho1.

    For the CBF constraint  Lfh + Lgh @ u >= -alpha_h:
        max over ||u||<=rho1 of LHS = Lfh + ||Lgh|| * rho1
        Infeasible alone if this max < -alpha_h.

    For the CLF constraint  LfV + LgV @ u <= -gamma_V:
        min over ||u||<=rho1 of LHS = LfV - ||LgV|| * rho1
        Infeasible alone if this min > -gamma_V.
    """
    # P2 CHANGE: P3 uses rho1 instead of rho2
    cbf_max = Lfh + np.linalg.norm(Lgh) * rho1
    clf_min = LfV - np.linalg.norm(LgV) * rho1

    cbf_alone = cbf_max < -alpha_h
    clf_alone = clf_min > -gamma_V

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


def p3(Lfh, Lgh, alpha_h,
       LfV, LgV, gamma_V,
       rho1,
       u_dim):
    """
    Solve:
        max ||u||
        s.t.
            Lfh + Lgh @ u >= -alpha_h
            LfV + LgV @ u <= -gamma_V
            ||u|| <= rho1

    The CLF constraint is hard: if the direct CVXPY formulation is not DCP-valid
    the SciPy fallback enforces same constraints inside  rho1 bound.
    """

    # Decision variables
    u = cp.Variable(u_dim)
    # P2 CHANGE: P3 does not use CLF slack, the CLF constraint stays hard

    # Objective
    objective = cp.Maximize(cp.norm(u, 2))
    # P2 CHANGE: P3 maximizes ||u||

    # Constraints
    constraints = [
        Lfh + Lgh @ u >= -alpha_h,
        LfV + LgV @ u <= -gamma_V,
        cp.norm(u, 2) <= rho1
    ]
    # P2 CHANGE: P3 keeps CLF constraint hard, uses rho1 instead of rho2

    # Problem
    prob = cp.Problem(objective, constraints)

    # P2 CHANGE: P3 uses rho1,  reject negative norm bound
    if rho1 < 0:
        raise ValueError("rho1 must be non-negative.")

    # Solve
    try:
        prob.solve(solver=cp.ECOS)  # ECOS or SCS
    except (cp.error.DCPError, cp.SolverError):
        # P2 CHANGE: CVXPY cannot solve max ||u|| directly under DCP rules, so P3 falls back to SciPy
        Lfh = np.atleast_1d(np.asarray(Lfh, dtype=float))
        Lgh = np.asarray(Lgh, dtype=float)
        alpha_h = np.atleast_1d(np.asarray(alpha_h, dtype=float))
        LfV = np.atleast_1d(np.asarray(LfV, dtype=float))
        LgV = np.asarray(LgV, dtype=float)
        gamma_V = np.atleast_1d(np.asarray(gamma_V, dtype=float))

        if Lgh.ndim == 0:
            Lgh = Lgh.reshape(1, 1)
        elif Lgh.ndim == 1:
            Lgh = Lgh.reshape(1, -1)

        if LgV.ndim == 0:
            LgV = LgV.reshape(1, 1)
        elif LgV.ndim == 1:
            LgV = LgV.reshape(1, -1)

        # P2 CHANGE: P3 uses global nonlinear solve to maximize ||u|| while enforcing same affine constraints and  rho1 norm cap
        cbf_constraint = NonlinearConstraint(
            lambda u_value: Lfh + Lgh @ u_value,
            lb=-alpha_h,
            ub=np.full(alpha_h.shape, np.inf)
        )
        clf_constraint = NonlinearConstraint(
            lambda u_value: LfV + LgV @ u_value,
            lb=np.full(gamma_V.shape, -np.inf),
            ub=-gamma_V
        )
        norm_constraint = NonlinearConstraint(
            lambda u_value: np.array([np.linalg.norm(u_value, 2)]),
            lb=np.array([0.0]),
            ub=np.array([float(rho1)])
        )

        result = shgo(
            func=lambda u_value: -float(np.linalg.norm(u_value, 2)),
            bounds=[(-float(rho1), float(rho1))] * u_dim,
            constraints=[cbf_constraint, clf_constraint, norm_constraint],
            n=max(64, 32 * u_dim),
            iters=4 if u_dim <= 3 else 2,
            sampling_method="sobol"
        )

        if result.x is None:
            _diagnose_infeasibility(
                float(np.asarray(Lfh).reshape(-1)[0]),
                np.asarray(Lgh).reshape(-1),
                float(np.asarray(alpha_h).reshape(-1)[0]),
                float(np.asarray(LfV).reshape(-1)[0]),
                np.asarray(LgV).reshape(-1),
                float(np.asarray(gamma_V).reshape(-1)[0]),
                rho1
            )
            return None, "infeasible"

        u_star = np.asarray(result.x, dtype=float).reshape(-1)
        cbf_ok = np.all(Lfh + Lgh @ u_star >= -alpha_h - 1e-7)
        clf_ok = np.all(LfV + LgV @ u_star <= -gamma_V + 1e-7)
        norm_ok = np.linalg.norm(u_star, 2) <= rho1 + 1e-7

        if not result.success or not (cbf_ok and clf_ok and norm_ok):
            _diagnose_infeasibility(
                float(np.asarray(Lfh).reshape(-1)[0]),
                np.asarray(Lgh).reshape(-1),
                float(np.asarray(alpha_h).reshape(-1)[0]),
                float(np.asarray(LfV).reshape(-1)[0]),
                np.asarray(LgV).reshape(-1),
                float(np.asarray(gamma_V).reshape(-1)[0]),
                rho1
            )
            return None, "infeasible"

        return u_star, "optimal"

    if prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        _diagnose_infeasibility(Lfh, Lgh, alpha_h, LfV, LgV, gamma_V, rho1)

    return u.value, prob.status
