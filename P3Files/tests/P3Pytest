import numpy as np

from edl_project.p3 import p3


def test_p3_pass_1():
    # This test checks that P3 can find a feasible point on the rho1 boundary.
    u, status = p3(
        Lfh=0.0,
        Lgh=np.array([1.0]),
        alpha_h=0.0,
        LfV=0.0,
        LgV=np.array([1.0]),
        gamma_V=-2.0,
        rho1=1.0,
        u_dim=1,
    )

    assert status == "optimal"
    assert u is not None
    assert np.linalg.norm(u) > 0.9


def test_p3_pass_2():
    # This test checks that P3 works for another simple feasible case.
    u, status = p3(
        Lfh=0.0,
        Lgh=np.array([1.0]),
        alpha_h=-0.5,
        LfV=0.0,
        LgV=np.array([1.0]),
        gamma_V=-3.0,
        rho1=2.0,
        u_dim=1,
    )

    assert status == "optimal"
    assert u is not None
    assert np.linalg.norm(u) > 1.5


def test_p3_infeasible_1():
    # This test checks that P3 returns infeasible when the constraints conflict.
    u, status = p3(
        Lfh=0.0,
        Lgh=np.array([1.0]),
        alpha_h=-2.0,
        LfV=0.0,
        LgV=np.array([1.0]),
        gamma_V=-1.0,
        rho1=3.0,
        u_dim=1,
    )

    assert status == "infeasible"
    assert u is None


def test_p3_infeasible_2():
    # This test checks that P3 returns infeasible when rho1 is too small.
    u, status = p3(
        Lfh=0.0,
        Lgh=np.array([1.0]),
        alpha_h=-2.0,
        LfV=0.0,
        LgV=np.array([1.0]),
        gamma_V=-3.0,
        rho1=1.0,
        u_dim=1,
    )

    assert status == "infeasible"
    assert u is None
