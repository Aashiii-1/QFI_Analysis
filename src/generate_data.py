"""Generate the quantum-sensing dataset by exact diagonalization.

Loops over ``N_B`` signals drawn from eq (7) and, for each of them, ``N_n``
independent noise realizations ``(Xi, Lambda)``. For every record it builds the
Hamiltonian of eq (1) at the *initial* sensor parameters ``Phi_0 = {J_0, B_0}``
and computes the raw readout ``S`` of eqs (2)-(3).

Because ``S`` depends on ``Phi``, which training is free to change, the stored
``S`` is only the ``Phi_0`` snapshot. What makes the dataset reusable is that
the sampled ``dB``, ``Xi`` and ``Lambda`` are written out as well, so
``ml_train.py`` can replay the exact same realizations while re-simulating the
physics at the current ``Phi``.

Output is a whitespace-delimited text file with a commented header describing
the run and the column layout.
"""

from __future__ import annotations

import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import sampling
from ising import readout

def column_layout(N: int) -> list[tuple[str, int]]:
    """``(name, width)`` pairs describing one dataset row."""
    return [
        ("signal_id", 1),
        ("noise_id", 1),
        ("J", N - 1),
        ("B", N),
        ("dB", N),
        ("S", N),
        ("xi", 3 * N),
        ("lam", N - 1),
    ]


def column_slices(N: int) -> dict[str, slice]:
    slices, start = {}, 0
    for name, width in column_layout(N):
        slices[name] = slice(start, start + width)
        start += width
    return slices


def generate(args) -> tuple[np.ndarray, dict]:
    N, n_signals, n_noise = args.N, args.N_B, args.N_n
    key = jax.random.key(args.seed)
    k_signal, k_noise = jax.random.split(key)

    J0 = jnp.full((N - 1,), args.J0)
    B0 = jnp.full((N,), args.B0)

    signals = sampling.sample_signals(
        k_signal,
        n_signals,
        N,
        n_modes=args.n_modes,
        amp_std=args.amp_std,
        omega_std=args.omega_std,
        phase_std=args.phase_std,
    )
    # One independent (Xi, Lambda) draw per (signal, noise) pair, so the batch
    # is an unbiased Monte-Carlo sample of the joint expectation in eq (5).
    n_records = n_signals * n_noise
    xi, lam = sampling.sample_noise(k_noise, n_records, N, args.sigma1, args.sigma2, args.ell)

    signal_id = np.repeat(np.arange(n_signals), n_noise)
    noise_id = np.tile(np.arange(n_noise), n_signals)
    dB = jnp.repeat(signals, n_noise, axis=0)

    measure = jax.jit(
        jax.vmap(lambda d, x, l: readout(J0, B0, d, x, l, N, args.beta, args.readout_method))
    )
    S = measure(dB, xi, lam)

    rows = np.concatenate(
        [
            signal_id[:, None],
            noise_id[:, None],
            np.tile(np.asarray(J0), (n_records, 1)),
            np.tile(np.asarray(B0), (n_records, 1)),
            np.asarray(dB),
            np.asarray(S),
            np.asarray(xi).reshape(n_records, 3 * N),
            np.asarray(lam),
        ],
        axis=1,
    )

    meta = {
        "N": N,
        "N_B": n_signals,
        "N_n": n_noise,
        "beta": args.beta,
        "sigma1": args.sigma1,
        "sigma2": args.sigma2,
        "ell": args.ell,
        "n_modes": args.n_modes,
        "amp_std": args.amp_std,
        "omega_std": args.omega_std,
        "phase_std": args.phase_std,
        "readout_method": args.readout_method,
        "seed": args.seed,
        "J0": " ".join(f"{v:.10e}" for v in np.asarray(J0)),
        "B0": " ".join(f"{v:.10e}" for v in np.asarray(B0)),
    }
    return rows, meta


def write_dataset(path: str, rows: np.ndarray, meta: dict) -> None:
    layout = column_layout(meta["N"])
    names = []
    for name, width in layout:
        names.extend([name] if width == 1 else [f"{name}[{i}]" for i in range(width)])
    header_lines = ["QLSENSE dataset: Ising quantum sensing, eqs (1)-(3) and (7)."]
    header_lines += [f"{k} = {v}" for k, v in meta.items()]
    header_lines.append("columns: " + " ".join(names))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fmt = ["%d", "%d"] + ["%.10e"] * (rows.shape[1] - 2)
    np.savetxt(path, rows, fmt=fmt, header="\n".join(header_lines))


def read_dataset(path: str) -> tuple[np.ndarray, dict]:
    """Inverse of :func:`write_dataset`: returns the rows and header metadata."""
    meta: dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if " = " in body:
                key, value = body.split(" = ", 1)
                meta[key.strip()] = value.strip()
    rows = np.loadtxt(path, ndmin=2)
    return rows, meta


def build_parser() -> argparse.ArgumentParser:
    d = sampling.DEFAULTS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--N", type=int, default=6, help="number of spins (Hilbert space 2^N)")
    p.add_argument("--N_B", type=int, default=64, help="number of signals drawn from eq (7)")
    p.add_argument("--N_n", type=int, default=8, help="noise realizations per signal")
    p.add_argument("--beta", type=float, default=1.0, help="inverse temperature in eq (3)")
    p.add_argument("--sigma1", type=float, default=d["sigma1"], help="on-site noise std (Xi)")
    p.add_argument("--sigma2", type=float, default=d["sigma2"], help="bond noise std (Lambda)")
    p.add_argument("--ell", type=float, default=d["ell"], help="noise correlation length")
    p.add_argument("--n-modes", dest="n_modes", type=int, default=d["n_modes"], help="M in eq (7)")
    p.add_argument("--amp-std", dest="amp_std", type=float, default=d["amp_std"])
    p.add_argument("--omega-std", dest="omega_std", type=float, default=d["omega_std"])
    p.add_argument("--phase-std", dest="phase_std", type=float, default=float(d["phase_std"]))
    p.add_argument("--J0", type=float, default=0.5, help="initial uniform coupling")
    p.add_argument("--B0", type=float, default=1.0, help="initial uniform transverse field")
    p.add_argument("--readout-method", dest="readout_method", default="eigh", choices=("eigh", "expm"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="output path (default data/qls_N{N}.txt)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = args.out or f"data/qls_N{args.N}.txt"
    rows, meta = generate(args)
    write_dataset(out, rows, meta)
    print(f"wrote {rows.shape[0]} records x {rows.shape[1]} columns to {out}")
    S = rows[:, column_slices(args.N)["S"]]
    print(f"readout S: mean {S.mean():+.4f}, std {S.std():.4f}, range [{S.min():+.4f}, {S.max():+.4f}]")


if __name__ == "__main__":
    main()
