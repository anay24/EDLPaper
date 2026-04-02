from abc import ABC, abstractmethod

import numpy as np


class ControlBarrierFunction(ABC):
    """
    Abstract base class for Control Barrier Functions (CBFs).

    A CBF h(x) defines a safe set S = {x : h(x) >= 0}. For an affine
    control system ẋ = f(x) + g(x)u, the CBF condition requires that there
    exists a control u satisfying:

        Lf h(x) + Lg h(x) u >= -alpha(h(x))

    where Lf, Lg are Lie derivatives along drift f and control matrix g,
    and alpha is a class-K function.

    Subclasses must implement h() and grad_h(). Override alpha() to use a
    nonlinear class-K function.
    """

    @abstractmethod
    def h(self, x: np.ndarray) -> float:
        """Evaluate the barrier function. Returns >= 0 inside the safe set."""
        ...

    @abstractmethod
    def grad_h(self, x: np.ndarray) -> np.ndarray:
        """Gradient of h w.r.t. the full state x. Shape: (n,)."""
        ...

    def alpha(self, h_val: float) -> float:
        """Class-K function for the CBF decay condition (default: identity)."""
        return h_val

    def lie_derivative(
        self,
        f: callable,
        g: callable,
        x: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """
        Compute Lf h and Lg h at state x.

        Args:
            f: Drift dynamics f(x) -> R^n.
            g: Control matrix g(x) -> R^(n x m).
            x: Current state, shape (n,).

        Returns:
            Lfh: scalar Lie derivative along drift.
            Lgh: Lie derivative along control directions, shape (m,).
        """
        grad = self.grad_h(x)
        Lfh = float(grad @ f(x))
        Lgh = grad @ g(x)
        return Lfh, Lgh


class GlideSlopeCBF(ControlBarrierFunction):
    """
    CBF enforcing a glide-slope (cone) constraint for powered descent.

    The cone surface is defined by:

        z = k * sqrt(x^2 + y^2),    k = tan(theta)

    where theta is the cone half-angle measured from the horizontal ground
    plane. The safe set (interior of the cone) is:

        S = {x : r_z - k * ||r_xy|| >= 0}

    so the barrier function is:

        h(x) = r_z - k * ||r_xy||

    A larger theta (steeper glide slope, e.g. pi/3) gives a larger k and a
    tighter, more vertical cone. A smaller theta (shallower, e.g. pi/6) gives
    a wider cone with more horizontal freedom.

    The state x must contain position as x[pos_start : pos_start+3]
    ordered [r_x, r_y, r_z].
    """

    def __init__(self, theta: float, pos_start: int = 0):
        """
        Args:
            theta: Glide-slope angle from the horizontal (radians), in (0, pi/2).
                   Example: pi/6 (30 deg) -> k ≈ 0.577 (wide cone);
                            pi/3 (60 deg) -> k ≈ 1.732 (narrow, steep cone).
            pos_start: Index of the first position component (r_x) in the state.
        """
        if not (0.0 < theta < np.pi / 2.0):
            raise ValueError("theta must be in (0, pi/2)")
        self.theta = theta
        self.k = np.tan(theta)   # cone slope: z = k * sqrt(x^2 + y^2)
        self.pos_start = pos_start

    def h(self, x: np.ndarray) -> float:
        """
        h(x) = r_z - k * ||r_xy||.

        Positive inside the safe cone, zero on the cone surface, negative outside.
        """
        r = x[self.pos_start : self.pos_start + 3]
        rxy_norm = np.linalg.norm(r[:2])
        return float(r[2] - self.k * rxy_norm)

    def grad_h(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of h w.r.t. the full state x.

            d h / d r_xy = -k * r_xy / ||r_xy||   (zero on the vertical axis)
            d h / d r_z  = 1
            d h / d v    = 0  (if state includes velocity)
        """
        r = x[self.pos_start : self.pos_start + 3]
        rxy = r[:2]
        rxy_norm = np.linalg.norm(rxy)

        grad = np.zeros(len(x))
        i = self.pos_start
        if rxy_norm > 1e-10:
            grad[i : i + 2] = -self.k * rxy / rxy_norm
        # On the vertical axis (rxy = 0), horizontal subgradient is 0.
        grad[i + 2] = 1.0
        return grad
