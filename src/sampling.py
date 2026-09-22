"""Noise and signal ensembles for the Ising quantum sensor.

Two ingredients of eq (1) are random:

* the spatially correlated noise ``Xi = {xi_i} ~ N(0, Sigma)`` and
  ``Lambda = {lam_ij} ~ N(0, Sigma')``, and
* the signal ``dB``, drawn from the multi-mode ensemble of eq (7),

      dB_x = sum_{m=1}^{M} A_m cos(omega_m x + zeta_m),

  with ``A``, ``omega``, ``zeta`` normally distributed and ``x`` the site index.

The correlation model follows ``spatial_noise`` in ``Functions.jl``: an
exponential kernel ``C(r) = exp(-r / ell)`` factorized by a Cholesky
decomposition, so a white draw ``z`` becomes a correlated draw ``L z``. The
covariance is ``sigma^2 C`` with ``sigma = sigma1`` for the on-site noise and
``sigma = sigma2`` for the bond noise, giving each component marginal standard
deviation ``sigma``.

All randomness is driven by explicit ``jax.random`` keys, so a run is fully
determined by its CLI seed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import ising  # noqa: F401  (imported for its float64 configuration side effect)

DEFAULTS = {
    "sigma1": 0.05,
    "sigma2": 0.05,
    "ell": 1.0,
    "n_modes": 3,
    "amp_std": 0.25,
    "omega_std": 1.0,
    "phase_std": jnp.pi,
}


def correlation_cholesky(n: int, ell: float, sigma: float, jitter: float = 1e-10) -> jax.Array:
    """Lower Cholesky factor of ``sigma^2 exp(-|i - j| / ell)`` on ``n`` points."""
    if n == 0:
        return jnp.zeros((0, 0))
    r = jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :]).astype(float)
    cov = sigma**2 * jnp.exp(-r / ell) + jitter * jnp.eye(n)
    return jnp.linalg.cholesky(cov)


def sample_xi(key, N: int, sigma1: float, ell: float) -> jax.Array:
    """``(3, N)`` on-site noise: independent correlated draws for x, y and z."""
    L = correlation_cholesky(N, ell, sigma1)
    white = jax.random.normal(key, (3, N))
    return white @ L.T


def sample_lambda(key, N: int, sigma2: float, ell: float) -> jax.Array:
    """``(N - 1,)`` correlated bond noise on the open chain."""
    L = correlation_cholesky(N - 1, ell, sigma2)
    white = jax.random.normal(key, (N - 1,))
    return L @ white


def sample_signal(
    key,
    N: int,
    n_modes: int,
    amp_std: float,
    omega_std: float,
    phase_std: float,
) -> jax.Array:
    """One ``(N,)`` draw from the eq (7) signal ensemble."""
    k_amp, k_omega, k_phase = jax.random.split(key, 3)
    amp = amp_std * jax.random.normal(k_amp, (n_modes,))
    omega = omega_std * jax.random.normal(k_omega, (n_modes,))
    phase = phase_std * jax.random.normal(k_phase, (n_modes,))
    x = jnp.arange(N, dtype=float)
    return jnp.sum(amp[:, None] * jnp.cos(omega[:, None] * x[None, :] + phase[:, None]), axis=0)


def sample_signals(key, n_signals: int, N: int, **kwargs) -> jax.Array:
    """``(n_signals, N)`` batch of eq (7) signals."""
    keys = jax.random.split(key, n_signals)
    return jax.vmap(lambda k: sample_signal(k, N, **kwargs))(keys)


def sample_noise(key, n_draws: int, N: int, sigma1: float, sigma2: float, ell: float):
    """``(n_draws, 3, N)`` on-site noise and ``(n_draws, N - 1)`` bond noise."""
    k_xi, k_lam = jax.random.split(key)
    xi = jax.vmap(lambda k: sample_xi(k, N, sigma1, ell))(jax.random.split(k_xi, n_draws))
    lam = jax.vmap(lambda k: sample_lambda(k, N, sigma2, ell))(jax.random.split(k_lam, n_draws))
    return xi, lam
