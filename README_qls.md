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
submit_qls.sbatch     Slurm batch job for CU Boulder Research Computing
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

Note that `run_qls.sh` takes **no command-line flags**. Anything you pass on the
command line is ignored, so configure it through the environment.

## Running on CU Boulder Research Computing (Slurm)

`submit_qls.sbatch` runs the pipeline as a batch job. Build the virtualenv once
on a **login node** — compute nodes generally have no outbound network, so the
job will not pip install for you — and then submit:

```bash
module load anaconda                  # check `module avail python` for the real name
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests      # confirm the analytic checks pass

mkdir -p logs                         # Slurm will not create the output dir
sbatch submit_qls.sbatch
```

Defaults are `N=8`, `EPOCHS=1000`, writing to `data/n8_run/` and
`results/n8_run/`, with `logs/qls_n8_<jobid>.{out,err}`. Override at submit time
without editing the script:

```bash
sbatch --export=ALL,N=6,EPOCHS=300,BATCH_SIZE=32 submit_qls.sbatch
sbatch --export=ALL,VENV=/projects/$USER/qls-venv submit_qls.sbatch
```

The script requests `--partition=acpu --qos=cpu-normal`, 8 CPUs, 32 GB and 24
hours. **Verify the partition and QoS against your own allocation** before the
first real run — `sinfo -s` lists reachable partitions, `sacctmgr show qos` and
`sacctmgr show assoc user=$USER` list the QoS names and limits you are entitled
to. It also caps `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`
at `$SLURM_CPUS_PER_TASK` so the BLAS backend does not oversubscribe the cores
Slurm granted, and pins `JAX_PLATFORMS=cpu` so JAX does not probe for a GPU on a
CPU partition. The `.out` file opens with the fully resolved configuration
(host, `nproc`, thread limit, every pipeline parameter) and closes with the
total wall time, including on failure.

### Memory scaling: why N=8 fits and N=12 does not

Cost is set by the Hilbert space dimension `2^N`. Exact diagonalization
materializes dense `2^N × 2^N` complex128 matrices (16 bytes per element), and
`_readout_eigh` additionally gathers `N` bit-flip-permuted copies of the
eigenvector matrix so that all `⟨n|X_j|n⟩` can be evaluated in one einsum. Both
stages are vmapped, so the live footprint is roughly

```
bytes  ≈  16  ·  (N + 2)  ·  4^N  ·  records_in_flight
          ^^      ^^^^^      ^^^     ^^^^^^^^^^^^^^^^^
     complex128   H, V, and        N_B·N_n when generating,
                  N copies of V    BATCH_SIZE when training
```

and reverse-mode autodiff keeps residuals alive across the backward pass, adding
a further factor of roughly two to three during training.

The `records_in_flight` term matters more than it looks: **data generation is
the memory peak, not training**, because `generate_data.py` vmaps over the
entire `N_B × N_n` dataset in one call while training only ever holds
`BATCH_SIZE` records.

| N | `2^N` | one eigenvector matrix | batch of 64, eigenvectors only | training, batch 64 | generation, 512 records |
| --- | --- | --- | --- | --- | --- |
| 6 | 64 | 64 KB | 4 MB | 32 MB | 256 MB |
| 8 | 256 | 1 MB | 64 MB | 640 MB | 5 GB |
| 10 | 1024 | 16 MB | 1 GB | 12 GB | 96 GB |
| 12 | 4096 | 256 MB | 16 GB | 224 GB | 1.8 TB |
| 14 | 16384 | 4 GB | 256 GB | 4 TB | 32 TB |

**N=8 fits** in the requested 32 GB. Measured peak RSS for a full
`N=8, N_B=64, N_n=8, BATCH_SIZE=64` run is **5.3 GB**, against 5.0 GB predicted
for the generation stage — close enough to trust the formula for sizing other
jobs.

**N=12 will not fit.** The eigenvector matrices alone are 16 GB for a batch of
64 before any autodiff residuals, the full training footprint is over 200 GB,
and generating the dataset at the default 512 records would need terabytes.
Raising `--mem` cannot rescue this; the exponent wins. Past N≈10 the approach
itself has to change — project onto the low-lying spectrum with a Krylov/Lanczos
solver instead of full ED, or move to the MPS/TEBD route already sketched in the
main [README](README.md).

**N=10 is the awkward middle.** Training at `BATCH_SIZE=16` needs about 3 GB and
is fine, but generation at 512 records wants 96 GB and will be killed. Until
`generate_data.py` chunks its vmap, get there by shrinking the dataset
(`N_B=16 N_N=4` → 64 records ≈ 12 GB) rather than by asking for more memory.

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
