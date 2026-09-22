"""Analytic sanity checks for the Ising quantum-sensing physics.

Run with ``pytest tests`` (or ``.venv/bin/pytest tests``).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import ising
import sampling

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
ID = np.eye(2, dtype=complex)


def local_operator(op, site, N):
    """Dense ``kron(I, ..., op, ..., I)`` reference, site 0 leftmost."""
    chain = [ID] * N
    chain[site] = op
    return functools.reduce(np.kron, chain)


def dense_hamiltonian(J, B, dB, xi, lam, N):
    """Independent dense-Kronecker construction of eq (1)."""
    H = np.zeros((2**N, 2**N), dtype=complex)
    for i in range(N - 1):
        H += (J[i] + lam[i]) * local_operator(Z, i, N) @ local_operator(Z, i + 1, N)
    for i in range(N):
        H += (B[i] + dB[i]) * local_operator(X, i, N)
        H += xi[0][i] * local_operator(X, i, N)
        H += xi[1][i] * local_operator(Y, i, N)
        H += xi[2][i] * local_operator(Z, i, N)
    return H


def random_inputs(N, seed=0):
    rng = np.random.default_rng(seed)
    return (
        jnp.asarray(rng.normal(0.5, 0.2, N - 1)),
        jnp.asarray(rng.normal(1.0, 0.2, N)),
        jnp.asarray(rng.normal(0.0, 0.3, N)),
        jnp.asarray(rng.normal(0.0, 0.1, (3, N))),
        jnp.asarray(rng.normal(0.0, 0.1, N - 1)),
    )


@pytest.mark.parametrize("N", [1, 2, 3, 4])
def test_hamiltonian_matches_dense_kronecker(N):
    """Index arithmetic must reproduce the textbook Kronecker construction."""
    J, B, dB, xi, lam = random_inputs(N)
    H = np.asarray(ising.build_hamiltonian(J, B, dB, xi, lam, N))
    ref = dense_hamiltonian(np.asarray(J), np.asarray(B), np.asarray(dB), np.asarray(xi),
                            np.asarray(lam), N)
    np.testing.assert_allclose(H, ref, atol=1e-12)


@pytest.mark.parametrize("N", [2, 3, 5])
def test_hamiltonian_is_hermitian(N):
    J, B, dB, xi, lam = random_inputs(N, seed=N)
    H = np.asarray(ising.build_hamiltonian(J, B, dB, xi, lam, N))
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


@pytest.mark.parametrize("method", ising.READOUT_METHODS)
@pytest.mark.parametrize("beta,B", [(1.0, 0.7), (0.3, 1.5), (2.5, -0.4), (5.0, 1.0)])
def test_single_spin_matches_minus_tanh(method, beta, B):
    """N=1 with H = B X has <X>_beta = -tanh(beta B) exactly.

    The thermal state weights the X = -1 eigenstate (energy -B) by e^{+beta B}
    and the X = +1 eigenstate by e^{-beta B}, so the magnetization
    anti-aligns with the field.
    """
    S = ising.readout(
        jnp.zeros(0), jnp.array([B]), jnp.zeros(1), jnp.zeros((3, 1)), jnp.zeros(0),
        1, beta, method,
    )
    assert S.shape == (1,)
    np.testing.assert_allclose(float(S[0]), -np.tanh(beta * B), atol=1e-12)


def test_infinite_temperature_readout_vanishes():
    """At beta = 0 every state is equally likely and Tr[X_j] = 0."""
    J, B, dB, xi, lam = random_inputs(4, seed=7)
    S = ising.readout(J, B, dB, xi, lam, 4, 0.0)
    np.testing.assert_allclose(np.asarray(S), 0.0, atol=1e-12)


def test_large_beta_converges_to_ground_state():
    """As beta -> infinity the thermal average becomes <gs|X_j|gs>."""
    N = 4
    J, B, dB, xi, lam = random_inputs(N, seed=3)
    H = ising.build_hamiltonian(J, B, dB, xi, lam, N)
    _, V = jnp.linalg.eigh(H)
    gs = np.asarray(V[:, 0])
    expected = np.array(
        [np.real(gs.conj() @ (local_operator(X, j, N) @ gs)) for j in range(N)]
    )
    S = np.asarray(ising.thermal_readout(H, 200.0, N))
    np.testing.assert_allclose(S, expected, atol=1e-8)


@pytest.mark.parametrize("N", [2, 4])
def test_readout_methods_agree(N):
    """eigh and expm are two routes to the same eq (3) expectation."""
    J, B, dB, xi, lam = random_inputs(N, seed=11)
    H = ising.build_hamiltonian(J, B, dB, xi, lam, N)
    a = np.asarray(ising.thermal_readout(H, 1.3, N, "eigh"))
    b = np.asarray(ising.thermal_readout(H, 1.3, N, "expm"))
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_readout_is_bounded():
    """X_j has spectrum {-1, +1}, so any thermal average lies in [-1, 1]."""
    J, B, dB, xi, lam = random_inputs(5, seed=5)
    S = np.asarray(ising.readout(J, B, dB, xi, lam, 5, 1.0))
    assert np.all(np.abs(S) <= 1.0 + 1e-12)


@pytest.mark.parametrize("method", ising.READOUT_METHODS)
def test_gradients_through_diagonalization_are_finite(method):
    """Autodiff must reach Phi = {J, B} through the ED, per eq (5)."""
    N = 4
    J, B, dB, xi, lam = random_inputs(N, seed=13)
    target = jnp.asarray(np.linspace(-0.5, 0.5, N))

    def loss(J_, B_):
        S = ising.readout(J_, B_, dB, xi, lam, N, 1.0, method)
        return jnp.sum((S - target) ** 2)

    gJ, gB = jax.grad(loss, argnums=(0, 1))(J, B)
    assert np.all(np.isfinite(np.asarray(gJ)))
    assert np.all(np.isfinite(np.asarray(gB)))
    assert np.linalg.norm(np.asarray(gB)) > 0


def test_gradient_matches_finite_differences():
    """Spot-check the ED gradient against a central difference."""
    N = 3
    J, B, dB, xi, lam = random_inputs(N, seed=17)

    def loss(B_):
        return jnp.sum(ising.readout(J, B_, dB, xi, lam, N, 1.0) ** 2)

    grad = np.asarray(jax.grad(loss)(B))
    eps = 1e-6
    fd = np.zeros(N)
    for i in range(N):
        plus = B.at[i].add(eps)
        minus = B.at[i].add(-eps)
        fd[i] = (float(loss(plus)) - float(loss(minus))) / (2 * eps)
    np.testing.assert_allclose(grad, fd, atol=1e-6)


def test_signal_matches_equation_7():
    """sample_signal must evaluate sum_m A_m cos(omega_m x + zeta_m)."""
    N, M = 6, 3
    key = jax.random.key(42)
    k_amp, k_omega, k_phase = jax.random.split(key, 3)
    amp = 0.25 * np.asarray(jax.random.normal(k_amp, (M,)))
    omega = 1.0 * np.asarray(jax.random.normal(k_omega, (M,)))
    phase = float(np.pi) * np.asarray(jax.random.normal(k_phase, (M,)))
    x = np.arange(N)
    expected = sum(amp[m] * np.cos(omega[m] * x + phase[m]) for m in range(M))

    got = np.asarray(sampling.sample_signal(key, N, M, 0.25, 1.0, float(np.pi)))
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_spatial_noise_has_requested_covariance():
    """L L^T must reproduce sigma^2 exp(-r/ell)."""
    n, ell, sigma = 6, 1.5, 0.3
    L = np.asarray(sampling.correlation_cholesky(n, ell, sigma))
    r = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    np.testing.assert_allclose(L @ L.T, sigma**2 * np.exp(-r / ell), atol=1e-9)


def test_noise_shapes_and_scale():
    xi, lam = sampling.sample_noise(jax.random.key(1), 4000, 5, 0.2, 0.1, 1.0)
    assert xi.shape == (4000, 3, 5)
    assert lam.shape == (4000, 4)
    assert abs(float(jnp.std(xi)) - 0.2) < 0.02
    assert abs(float(jnp.std(lam)) - 0.1) < 0.01
