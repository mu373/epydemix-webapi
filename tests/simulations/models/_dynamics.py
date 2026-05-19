"""Shared helpers for dynamics tests.

Reference ODE integrators and trajectory extractors used to assert
epidemiological behavior (peak, final size, monotonicity over R0, ...).
"""

from __future__ import annotations

import numpy as np


def rk4_sir(
    beta: float,
    gamma: float,
    N: float,
    S0: float,
    I0: float,
    R0: float,
    T: int,
    h: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """RK4-integrate homogeneous SIR for `T` days. Returns daily (S, I)."""

    def deriv(s, i):
        infection = beta * s * i / N
        return -infection, infection - gamma * i

    s, i, r = S0, I0, R0
    S = np.empty(T)
    I = np.empty(T)
    S[0], I[0] = s, i
    steps = int(round(1.0 / h))
    for day in range(1, T):
        for _ in range(steps):
            k1s, k1i = deriv(s, i)
            k2s, k2i = deriv(s + h * k1s / 2, i + h * k1i / 2)
            k3s, k3i = deriv(s + h * k2s / 2, i + h * k2i / 2)
            k4s, k4i = deriv(s + h * k3s, i + h * k3i)
            s += h * (k1s + 2 * k2s + 2 * k3s + k4s) / 6
            i += h * (k1i + 2 * k2i + 2 * k3i + k4i) / 6
            r += h * gamma * (i)  # not used; kept for symmetry
        S[day], I[day] = s, i
    return S, I


def median_series(data: dict, compartment: str, age_group: str = "A") -> np.ndarray:
    """Pull the 0.5-quantile time series for a compartment out of a response body."""
    return np.array(data["compartments"]["data"][compartment][age_group]["0.5"], dtype=float)
