# Quantum Sensing via the Ising Model — JAX pipeline

A Python/JAX implementation of equations (1)–(7) of *Quantum Sensing via the Ising
Model* (R. T. Grimm). It lives alongside the Julia code in this repository
(`Functions.jl`, `tests.jl`), which is untouched.

The idea: a transverse-field Ising chain is used as a sensor. A spatially
resolved signal `δB` perturbs the transverse field, the chain is read out as the
thermal magnetization `S_j = ⟨X_j⟩_β`, and a neural filter `f_Θ` tries to invert
the readout back to `δB`. The sensor parameters `Φ = {J_ij, B_i}` are trained
*together with* the filter by differentiating through the exact diagonalization,
so the hardware and the post-processing are co-designed.

## Equation-to-code map

| Paper | Code |
| --- | --- |
| (1) `H = Σ⟨ij⟩(J+λ)Z_iZ_j + Σ_i(B+δB)X_i + Σ_i ξ_i·S_i` | `ising.build_hamiltonian` |
| (2) `S_j = ⟨X_j⟩_β` | `ising.thermal_readout` |
| (3) `⟨O⟩_β = Tr[O e^{-βH}]/Z` | `ising.thermal_readout` (`softmax(-βE)` weights, or `expm`) |
| (4) `S̃(Θ) = f_Θ(S)` | `ml_train.SensorFilter.mlp` (`eqx.nn.MLP`) |
| (5) `argmin_{Θ,Φ} E‖S̃ − δB‖²` | `ml_train.make_loss` + Optax Adam |
| (6) `Φ = {J_ij, B_i}` | `ml_train.SensorFilter.J`, `.B` |
| (7) `δB_x = Σ_m A_m cos(ω_m x + ζ_m)` | `sampling.sample_signal` |

Note that the Hamiltonian uses the paper's all-plus sign convention, which
differs from the explicit minus signs in the Julia `TFIM`.

## Layout

```
requirements.txt      jax, jaxlib, equinox, optax, numpy, pytest
run_qls.sh            two-stage driver (generate data, then train)
src/ising.py          Pauli operators, eq (1) Hamiltonian, eq (2)-(3) readout
src/sampling.py       correlated Ξ, Λ and the eq (7) signal ensemble
src/generate_data.py  exact diagonalization sweep -> data/qls_N{N}.txt
src/ml_train.py       Equinox MLP + trainable Φ, eq (5) objective
tests/test_ising.py   analytic checks
```

## Running

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

./run_qls.sh                       # defaults: N=6, N_B=64, N_n=8, beta=1.0
N=4 N_B=32 EPOCHS=40 ./run_qls.sh  # smaller/faster
.venv/bin/python -m pytest tests   # analytic sanity checks
```

Overridable environment variables: `N`, `N_N`, `N_B`, `BETA`, `SIGMA1`, `SIGMA2`,
`ELL`, `M_MODES`, `AMP_STD`, `J0`, `B0`, `EPOCHS`, `BATCH_SIZE`, `LR_THETA`,
`LR_PHI`, `WIDTH`, `DEPTH`, `READOUT`, `VAL_FRAC`, `LOG_EVERY`, `SEED`,
`DATA_DIR`, `RESULTS_DIR`, `PYTHON`.

## Implementation notes

**No dense Kronecker products.** Operators act by index arithmetic on the `2^N`
computational basis: site `j` is bit `N-1-j`, `X_j` and `Y_j` are the bit-flip
permutation `a ^ (1 << (N-1-j))`, and `Z_j` is the diagonal `1 - 2·bit_j(a)`.
`tests/test_ising.py` checks this against an explicit `kron` construction.

**The dataset stores noise, not just readouts.** `S` is a function of `Φ`, which
training changes, so a stored `S` would go stale after the first gradient step.
`generate_data.py` therefore records the sampled `δB`, `Ξ` and `Λ` as well, and
`ml_train.py` replays exactly those realizations while re-simulating the physics
at the current `Φ`.

**Two readout routes.** `eigh` diagonalizes `H` and Boltzmann-weights the
eigenbasis expectation values with `jax.nn.softmax(-βE)`, which is stable for
large `β`. Its reverse-mode derivative contains `1/(E_i − E_j)` terms, which are
ill-conditioned at (near-)degeneracies; nonzero `Ξ`, `Λ` generically lift those.
`READOUT=expm` instead forms `e^{-βH}` directly (Gershgorin-shifted to avoid
overflow), which is slower but has a well-conditioned derivative. The two agree
to `1e-10` in the tests. Everything runs in float64.

**Validation split is by signal.** Whole signals are held out, so the reported
validation loss measures generalization to unseen members of the eq (7) ensemble
rather than to unseen noise draws of a familiar signal.

## Output

`results/summary.json` plus per-run loss histories, optimized `Φ`, and serialized
models. The headline number is the final validation loss with `Φ` frozen at its
initial value versus `Φ` co-optimized — the gain attributable to engineering the
sensor rather than only the filter.
