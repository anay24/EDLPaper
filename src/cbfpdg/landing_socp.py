"""
One-step SOCP for powered-descent landing combining:

  - Glide-slope HOCBF: keeps the vehicle inside the approach cone
  - Velocity-zeroing CLF: drives velocity to zero for a soft landing

Dynamics (double integrator in R^3):
    ṙ = v
    v̇ = u + a_grav        (u = thrust acceleration, a_grav = [0, 0, -g])

State:   x = [r; v] ∈ R^6,   r, v ∈ R^3
Control: u ∈ R^3

HOCBF formulation (relative degree 2):
    h   = r_z - k ‖r_xy‖
    ψ₁  = Lf h + α₁ h = (v_z - k r̂·v_xy) + α₁ (r_z - k ‖r_xy‖)

    Constraint: Lf ψ₁ + Lg ψ₁ · u ≥ -α₂ ψ₁

For the double integrator (u enters v̇, so ∂ψ₁/∂v = ∇_r h):

    Lg ψ₁ = ∇_r h = [-k r̂_x, -k r̂_y, 1]          (nonzero → control authority)

    Lf ψ₁ = -k ‖P_⊥ v_xy‖² / ‖r_xy‖  +  α₁ Lf h  +  a_grav[2]

    where P_⊥ v_xy = v_xy - r̂ (r̂·v_xy)  is the component of v_xy ⊥ to r̂.

CLF formulation:
    V(x) = 0.5 ‖v‖²
    Lf V = v · a_grav = -g v_z
    Lg V = v
    Constraint: Lf V + Lg V · u ≤ -γ V
"""

import numpy as np

from cbfpdg.cbf import GlideSlopeCBF
from cbfpdg.clfs import ControlLyapunovFunction
from cbfpdg.one_step_socp import solve_socp


def _hocbf_terms(
    cbf: GlideSlopeCBF,
    x: np.ndarray,
    a_grav: np.ndarray,
    alpha1: float,
    alpha2: float,
) -> tuple[float, np.ndarray, float]:
    """
    Compute HOCBF constraint terms for the glide-slope CBF.

    Returns:
        Lf_psi1:  scalar  — drift part of the HOCBF constraint LHS
        Lg_psi1:  (3,)    — control coefficient  (= ∇_r h)
        alpha_hocbf: scalar — RHS scaling  (= α₂ · ψ₁)
    """
    i = cbf.pos_start
    r   = x[i     : i + 3]
    v   = x[i + 3 : i + 6]   # velocity immediately follows position in state
    rxy = r[:2]
    vxy = v[:2]
    vz  = v[2]
    rxy_norm = np.linalg.norm(rxy)

    h_val = cbf.h(x)

    if rxy_norm > 1e-10:
        r_hat = rxy / rxy_norm

        Lfh = vz - cbf.k * (r_hat @ vxy)
        psi1 = Lfh + alpha1 * h_val

        # Lg ψ₁ = ∇_r h  (control enters v-dynamics with identity)
        Lg_psi1 = np.array([-cbf.k * r_hat[0], -cbf.k * r_hat[1], 1.0])

        # Lf ψ₁ = -k ‖P_⊥ v_xy‖² / ‖r_xy‖  +  α₁ Lf h  +  a_grav[2]
        perp_vxy   = vxy - r_hat * (r_hat @ vxy)   # v_xy ⊥ r̂
        Lf_psi1    = (
            -cbf.k * float(perp_vxy @ perp_vxy) / rxy_norm
            + alpha1 * Lfh
            + a_grav[2]
        )
    else:
        # On the vertical axis: horizontal subgradient is zero.
        psi1    = vz + alpha1 * h_val
        Lg_psi1 = np.array([0.0, 0.0, 1.0])
        Lf_psi1 = alpha1 * vz + a_grav[2]

    return Lf_psi1, Lg_psi1, float(alpha2 * psi1)


def solve_landing_socp(
    x: np.ndarray,
    cbf: GlideSlopeCBF,
    a_grav: np.ndarray,
    clf: ControlLyapunovFunction,
    alpha1: float = 1.0,
    alpha2: float = 1.0,
    gamma: float = 0.5,
    rho: float = 20.0,
    pos_dim: int = 3,
) -> tuple[np.ndarray | None, str]:
    """
    Solve the one-step landing SOCP with HOCBF glide-slope safety.

    Minimises ‖u‖ subject to:
      (1) HOCBF cone constraint:    Lf ψ₁ + Lg ψ₁ · u ≥ -α₂ ψ₁
      (2) CLF velocity convergence: Lf V  + Lg V  · u ≤ -γ V
      (3) Trust-region bound:       ‖u‖ ≤ ρ

    Args:
        x:      Current state [r; v] ∈ R^6, r = x[:3], v = x[3:].
        cbf:    GlideSlopeCBF instance.
        a_grav: Gravity vector, e.g. np.array([0., 0., -9.81]).
        clf:    ControlLyapunovFunction for V(x) = 0.5 ‖v‖².
        alpha1: HOCBF first-layer class-K gain  (α₁ > 0).
        alpha2: HOCBF second-layer class-K gain (α₂ > 0).
        gamma:  CLF decay rate; enforces V̇ ≤ -γ V  (γ > 0).
        rho:    Maximum thrust acceleration magnitude.
        pos_dim: Dimension of position sub-vector (default 3).

    Returns:
        u_opt:  Optimal thrust acceleration ∈ R^3, or None if infeasible.
        status: Solver status string.
    """
    # HOCBF terms  (Lgh ≠ 0, so the constraint actively steers u)
    Lfh, Lgh, alpha_h = _hocbf_terms(cbf, x, a_grav, alpha1, alpha2)

    # CLF terms  (V = 0.5 ‖v‖²)
    f_double_int = lambda x: np.concatenate([x[pos_dim:], a_grav])
    g_double_int = lambda x: np.vstack([np.zeros((pos_dim, pos_dim)), np.eye(pos_dim)])

    LfV, LgV = clf.lie_derivative(f_double_int, g_double_int, x)
    gamma_V  = gamma * clf.V(x)

    return solve_socp(
        Lfh=Lfh,
        Lgh=Lgh,
        alpha_h=alpha_h,
        LfV=LfV,
        LgV=LgV,
        gamma_V=gamma_V,
        rho2=rho,
        u_dim=pos_dim,
    )
