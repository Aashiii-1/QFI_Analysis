"""Differentiable exact diagonalization for the Ising quantum sensor.

Implements equations (1)-(3) of ``Ising_Model_Quantum_Sensing.pdf``:

    (1)  H = sum_<ij> (J_ij + lam_ij) Z_i Z_j
             + sum_i (B_i + dB_i) X_i
             + sum_i (xi^x_i X_i + xi^y_i Y_i + xi^z_i Z_i)

    (2)  S_j = <X_j>_beta

    (3)  <O>_beta = Tr[O exp(-beta H)] / Tr[exp(-beta H)]

The sign convention is the all-plus one of the paper, which differs from the
``TFIM`` in ``Functions.jl`` (that one carries explicit minus signs).

Pauli operators are applied by index arithmetic on the 2^N computational basis
instead of dense Kronecker products, so no 3N dense matrices are ever built.
Site ``j`` is bit ``N - 1 - j`` of the basis index, which reproduces the
ordering of ``kron(op_0, op_1, ..., op_{N-1})``.

Everything here is a pure function of ``(J, B, dB, xi, lam)`` built from
``jnp`` primitives, so ``jax.grad`` flows all the way from the readout back to
the sensor parameters Phi = {J, B}.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

# Eigen-decomposition of a near-degenerate Hamiltonian and its reverse-mode
# derivative are badly conditioned in single precision, so the whole pipeline
# runs in float64. This has to happen before any array is created.
jax.config.update("jax_enable_x64", True)

READOUT_METHODS = ("eigh", "expm")


@functools.partial(jax.jit, static_argnums=0)
def flip_indices(N: int) -> jax.Array:
    """``(N, 2^N)`` table with ``flip[j, a] = a ^ (1 << (N - 1 - j))``.

    Row ``j`` is the permutation of basis indices induced by ``X_j`` (and, up
    to a phase, by ``Y_j``).
    """
    a = jnp.arange(2**N)
    masks = 1 << jnp.arange(N - 1, -1, -1)
    return a[None, :] ^ masks[:, None]


@functools.partial(jax.jit, static_argnums=0)
def z_diagonals(N: int) -> jax.Array:
    """``(N, 2^N)`` table with ``z[j, a] = +1/-1``, the diagonal of ``Z_j``."""
    a = jnp.arange(2**N)
    shifts = jnp.arange(N - 1, -1, -1)
    bits = (a[None, :] >> shifts[:, None]) & 1
    return 1 - 2 * bits


def build_hamiltonian(J, B, dB, xi, lam, N: int) -> jax.Array:
    """Assemble the dense ``(2^N, 2^N)`` Hermitian Hamiltonian of eq (1).

    Args:
        J: ``(N - 1,)`` nearest-neighbour couplings of the open chain.
        B: ``(N,)`` static transverse field.
        dB: ``(N,)`` spatially resolved signal, eq (7).
        xi: ``(3, N)`` noise field ``Xi``; rows are the x, y, z components of
            ``xi_i . S_i``.
        lam: ``(N - 1,)`` bond noise ``Lambda``.
        N: number of sites (static).
    """
    dim = 2**N
    z = z_diagonals(N)
    flip = flip_indices(N)
    xi_x, xi_y, xi_z = xi[0], xi[1], xi[2]

    diag = jnp.sum(xi_z[:, None] * z, axis=0)
    if N > 1:
        diag = diag + jnp.sum((J + lam)[:, None] * z[:-1] * z[1:], axis=0)
    H = jnp.diag(diag + 0j)

    cols = jnp.arange(dim)
    for j in range(N):
        # X_j |a> = |flip(a)>, and Y_j |a> = i z_j(a) |flip(a)>.
        coeff = (B[j] + dB[j] + xi_x[j]) + 1j * xi_y[j] * z[j]
        H = H.at[flip[j], cols].add(coeff)
    return H


def _readout_eigh(H, beta: float, N: int) -> jax.Array:
    E, V = jnp.linalg.eigh(H)
    # softmax(-beta E) is exp(-beta E_n) / Z without ever forming exp of a
    # large positive number, so beta can be taken large safely.
    w = jax.nn.softmax(-beta * E)
    flip = flip_indices(N)
    # <n|X_j|n> = sum_a conj(V[a, n]) V[flip[j, a], n]
    xj = jnp.real(jnp.einsum("an,jan->jn", jnp.conj(V), V[flip]))
    return xj @ w


def _readout_expm(H, beta: float, N: int) -> jax.Array:
    dim = 2**N
    # Gershgorin: every eigenvalue is >= -max_i sum_j |H_ij|, so shifting by
    # that bound keeps exp(-beta H') <= 1. The shift cancels in the ratio
    # below, hence stop_gradient.
    shift = jax.lax.stop_gradient(jnp.max(jnp.sum(jnp.abs(H), axis=1)))
    rho = jax.scipy.linalg.expm(-beta * (H + shift * jnp.eye(dim)))
    Z = jnp.real(jnp.trace(rho))
    flip = flip_indices(N)
    cols = jnp.arange(dim)
    # Tr[X_j rho] = sum_a rho[flip(a), a]
    return jnp.real(jnp.sum(rho[flip, cols[None, :]], axis=1)) / Z


def thermal_readout(H, beta: float, N: int, method: str = "eigh") -> jax.Array:
    """Raw readout ``S_j = <X_j>_beta``, eqs (2)-(3), as an ``(N,)`` array.

    ``method="eigh"`` diagonalizes ``H`` and Boltzmann-weights the eigenbasis
    expectation values; it is the cheaper path and the default. ``method="expm"``
    forms ``exp(-beta H)`` directly, which costs more but has a reverse-mode
    derivative free of the ``1 / (E_i - E_j)`` terms that ``eigh`` backprop
    produces at (near-)degeneracies.
    """
    if method == "eigh":
        return _readout_eigh(H, beta, N)
    if method == "expm":
        return _readout_expm(H, beta, N)
    raise ValueError(f"unknown readout method {method!r}, expected one of {READOUT_METHODS}")


def readout(J, B, dB, xi, lam, N: int, beta: float, method: str = "eigh") -> jax.Array:
    """Build eq (1) and measure eq (2) in one differentiable step."""
    return thermal_readout(build_hamiltonian(J, B, dB, xi, lam, N), beta, N, method)
