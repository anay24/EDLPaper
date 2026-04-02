"""
Simulation of a lander descending under the landing SOCP controller.

System: double integrator with uniform gravity
    ṙ = v
    v̇ = u + a_grav        (u = thrust acceleration, a_grav = [0, 0, -g])

State:   x = [r; v] ∈ R^6
Control: u ∈ R^3 (thrust acceleration, bounded by rho)

Barrier function:
    GlideSlopeCBF with HOCBF (relative degree 2).
    ψ₁ = Lf h + α₁ h,  constraint: Lf ψ₁ + Lg ψ₁ · u ≥ -α₂ ψ₁
    Lg ψ₁ = ∇_r h ≠ 0, so the SOCP has real control authority.

Lyapunov function:
    V(x) = 0.5 * ‖v‖²   (drives velocity to zero for soft landing)
    Uses QuadraticCLF with block-zero P = diag(0,0,0, 0.5,0.5,0.5).

Run with:
    uv run python simulate_landing.py
"""

import numpy as np
import matplotlib.pyplot as plt

from cbfpdg.cbf import GlideSlopeCBF
from cbfpdg.clfs import QuadraticCLF
from cbfpdg.landing_socp import solve_landing_socp


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

G = 9.81              # gravitational acceleration (m/s^2)
A_GRAV = np.array([0.0, 0.0, -G])

THETA  = np.deg2rad(40)              # glide-slope half-angle from horizontal
R0     = np.array([15.0, 3.0, 30.0]) # initial position  (m)
V0     = np.array([-1.0, -0.6, -10.0]) # initial velocity  (m/s)

ALPHA1 = 1.0   # HOCBF first-layer gain
ALPHA2 = 1.0   # HOCBF second-layer gain
GAMMA  = 0.5   # CLF decay rate  (V̇ ≤ -γ V)
RHO    = 15.0  # max thrust magnitude (m/s²)

DT     = 0.05  # timestep (s)
T_MAX  = 20.0  # max simulation time (s)


# ---------------------------------------------------------------------------
# Barrier and Lyapunov functions
# ---------------------------------------------------------------------------

cbf = GlideSlopeCBF(theta=THETA, pos_start=0)

# V(x) = 0.5 * ‖v‖²  via block-zero P = diag(0,0,0, 0.5,0.5,0.5)
P = np.block([
    [np.zeros((3, 3)), np.zeros((3, 3))],
    [np.zeros((3, 3)), 0.5 * np.eye(3)],
])
clf = QuadraticCLF(P=P)


# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([R0, V0])
    n_steps = int(T_MAX / DT)

    t_hist = [0.0]
    x_hist = [x.copy()]
    u_hist = []
    h_hist = [cbf.h(x)]

    for step in range(n_steps):
        r = x[:3]
        if r[2] <= 0.0:
            print(f"Landed at t = {step * DT:.2f} s,  r = {r}")
            break

        h_val = cbf.h(x)
        if h_val < -0.05:
            print(f"WARNING: CBF violated at t = {step * DT:.2f} s,  h = {h_val:.3f}")

        u, status = solve_landing_socp(
            x=x, cbf=cbf, a_grav=A_GRAV, clf=clf,
            alpha1=ALPHA1, alpha2=ALPHA2,
            gamma=GAMMA, rho=RHO,
        )

        if u is None:
            print(f"Solver failed at t = {step * DT:.2f} s,  status = {status}")
            break

        # Euler integration of double integrator
        v_new = x[3:] + DT * (u + A_GRAV)
        r_new = x[:3] + DT * x[3:]
        x = np.concatenate([r_new, v_new])

        t_hist.append((step + 1) * DT)
        x_hist.append(x.copy())
        u_hist.append(u.copy())
        h_hist.append(cbf.h(x))

    return (
        np.array(t_hist),
        np.array(x_hist),
        np.array(u_hist) if u_hist else np.zeros((0, 3)),
        np.array(h_hist),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot(t, x_hist, u_hist, h_hist) -> None:
    r_hist = x_hist[:, :3]
    v_hist = x_hist[:, 3:]

    fig = plt.figure(figsize=(13, 9))
    fig.suptitle(
        f"Landing SOCP  |  θ = {np.rad2deg(THETA):.0f}°  "
        f"ρ = {RHO} m/s²  γ = {GAMMA}  g = {G} m/s²",
        fontsize=12,
    )

    # ── 3-D trajectory ───────────────────────────────────────────────────────
    ax3d = fig.add_subplot(221, projection="3d")

    # Cone surface
    phi = np.linspace(0, 2 * np.pi, 60)
    z_vals = np.linspace(0, R0[2] * 1.1, 30)
    PHI, Z = np.meshgrid(phi, z_vals)
    cone_r = Z / cbf.k    # at altitude z, cone radius = z / k
    ax3d.plot_surface(
        cone_r * np.cos(PHI), cone_r * np.sin(PHI), Z,
        alpha=0.10, color="orange", linewidth=0,
    )

    ax3d.plot(r_hist[:, 0], r_hist[:, 1], r_hist[:, 2], "b-", lw=1.5, label="Trajectory")
    ax3d.scatter(*r_hist[0],  color="green", s=60, zorder=5, label="Start")
    ax3d.scatter(*r_hist[-1], color="red",   s=60, zorder=5, label="End")
    ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
    ax3d.set_title("3-D Trajectory")
    ax3d.legend(fontsize=8)

    # ── Side view (horizontal distance vs altitude) ───────────────────────────
    ax2 = fig.add_subplot(222)
    rxy_norm = np.linalg.norm(r_hist[:, :2], axis=1)
    ax2.plot(rxy_norm, r_hist[:, 2], "b-o", ms=2, label="Trajectory")

    z_line = np.linspace(0, R0[2] * 1.1, 200)
    ax2.plot(z_line / cbf.k, z_line, "r--", lw=1.5,
             label=f"Cone boundary  (θ = {np.rad2deg(THETA):.0f}°)")
    ax2.fill_betweenx(z_line, 0, z_line / cbf.k,
                      alpha=0.08, color="orange", label="Safe region")
    ax2.scatter(rxy_norm[0],  r_hist[0, 2],  color="green", s=60, zorder=5, label="Start")
    ax2.scatter(rxy_norm[-1], r_hist[-1, 2], color="red",   s=60, zorder=5, label="End")
    ax2.set_xlabel("Horizontal distance  ‖r_xy‖  (m)")
    ax2.set_ylabel("Altitude  r_z  (m)")
    ax2.set_title("Side View  (cone cross-section)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.4)

    # ── Control magnitude & velocity magnitude ────────────────────────────────
    ax3 = fig.add_subplot(223)
    t_ctrl = t[: len(u_hist)]
    if len(u_hist):
        ax3.plot(t_ctrl, np.linalg.norm(u_hist, axis=1), "k-", lw=1.5, label="‖u‖  (thrust)")
        ax3.axhline(RHO, color="r", ls="--", lw=1, label=f"Bound ρ = {RHO} m/s²")
    ax3.plot(t, np.linalg.norm(v_hist, axis=1), "b--", lw=1.2, label="‖v‖  (speed)")
    ax3.set_xlabel("Time  (s)")
    ax3.set_ylabel("Magnitude  (m/s  or  m/s²)")
    ax3.set_title("Control Magnitude & Speed")
    ax3.legend(fontsize=8)
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.4)

    # ── CBF value ─────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(224)
    h_arr = np.array(h_hist)
    ax4.plot(t, h_arr, "g-", lw=1.5, label=r"$h(x) = r_z - k\,\|r_{xy}\|$")
    ax4.axhline(0, color="r", ls="--", lw=1.5, label="Safety boundary  h = 0")
    ax4.fill_between(t, 0, h_arr,
                     where=h_arr >= 0, alpha=0.15, color="green", label="Safe")
    ax4.fill_between(t, 0, h_arr,
                     where=h_arr < 0,  alpha=0.30, color="red",   label="Unsafe")
    ax4.set_xlabel("Time  (s)")
    ax4.set_ylabel("h(x)")
    ax4.set_title("CBF Value  (must stay ≥ 0)")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.4)

    plt.tight_layout()
    out = "landing_trajectory.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")
    plt.show()


if __name__ == "__main__":
    t, x_hist, u_hist, h_hist = run()
    plot(t, x_hist, u_hist, h_hist)
