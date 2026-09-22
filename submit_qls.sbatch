#!/bin/bash
# Slurm batch script for the Ising quantum-sensing pipeline on CU Boulder
# Research Computing (CURC).
#
# Usage, from the repository root on a login node:
#
#     mkdir -p logs                      # Slurm will NOT create this for you
#     sbatch submit_qls.sbatch
#
# Override any knob at submit time without editing this file:
#
#     sbatch --export=ALL,N=6,EPOCHS=300 submit_qls.sbatch
#
# Submit from the repository root: Slurm copies this script to a spool
# directory, so $0 is useless for locating the repo and we rely on
# $SLURM_SUBMIT_DIR instead.

#SBATCH --job-name=qls_n8
#SBATCH --output=logs/qls_n8_%j.out
#SBATCH --error=logs/qls_n8_%j.err
# Partition and QoS are the values from the project's draft directives. Verify
# them against your own allocation before the first real run:
#     sinfo -s                       # partitions you can actually reach
#     sacctmgr show qos format=Name,MaxWall,MaxTRES
#     sacctmgr show assoc user=$USER format=Account,Partition,QOS
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user=aaso5268@colorado.edu
#SBATCH --mail-type=FAIL,END

set -euo pipefail

START_TIME=$SECONDS
print_wall_time() {
    local status=$?
    local elapsed=$((SECONDS - START_TIME))
    printf '\ntotal wall time: %02d:%02d:%02d (%d s), exit status %d\n' \
        $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60)) "$elapsed" "$status"
}
trap print_wall_time EXIT

REPO_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$REPO_DIR"

# --------------------------------------------------------------------------
# Pipeline configuration.
#
# run_qls.sh is driven entirely by environment variables and accepts no
# command-line flags -- passing "--N 8 --epochs 1000" would be silently
# ignored and you would get the N=6 defaults instead. So export the variables.
# --------------------------------------------------------------------------
export N=${N:-8}                              # spins; Hilbert space is 2^N = 256
export EPOCHS=${EPOCHS:-1000}
export N_B=${N_B:-64}                         # signals drawn from eq (7)
export N_N=${N_N:-8}                          # noise realizations per signal
export BETA=${BETA:-1.0}                      # inverse temperature in eq (3)
export BATCH_SIZE=${BATCH_SIZE:-64}
export SEED=${SEED:-0}
export LOG_EVERY=${LOG_EVERY:-25}
export DATA_DIR=${DATA_DIR:-data/n${N}_run}
export RESULTS_DIR=${RESULTS_DIR:-results/n${N}_run}

# --------------------------------------------------------------------------
# Threads. JAX/XLA and the LAPACK backend each pick their own thread count
# unless told otherwise, which on a shared node means oversubscribing the
# cores Slurm actually gave us.
# --------------------------------------------------------------------------
CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}
export OMP_NUM_THREADS=$CPUS
export MKL_NUM_THREADS=$CPUS
export OPENBLAS_NUM_THREADS=$CPUS

# acpu is a CPU partition: without this JAX probes for a GPU backend, fails to
# find one, and emits a warning on every run.
export JAX_PLATFORMS=cpu

# --------------------------------------------------------------------------
# Environment bootstrap.
#
# Module names differ between CURC systems and change across maintenance
# windows. Check `module avail python` / `module avail anaconda` on a login
# node and adjust the load below to whatever your allocation actually has.
# --------------------------------------------------------------------------
if command -v module >/dev/null 2>&1; then
    module purge
    module load anaconda || module load python || {
        echo "WARNING: could not load a python module; falling back to the system python" >&2
    }
fi

# Compute nodes generally have no outbound network, so pip installing here
# would hang or fail. Build the venv once on a login node and just activate it.
VENV=${VENV:-"$REPO_DIR/.venv"}
if [[ ! -x "$VENV/bin/python" ]]; then
    cat >&2 <<EOF
ERROR: no usable virtualenv at $VENV

Compute nodes have no outbound network access, so this job will not build one
for you. Create it on a login node first:

    cd $REPO_DIR
    module load anaconda          # or whatever 'module avail python' reports
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest tests

Then resubmit with: sbatch submit_qls.sbatch
EOF
    exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHON="$VENV/bin/python"

# --------------------------------------------------------------------------
# Resolved configuration, so the .out file documents its own run.
# --------------------------------------------------------------------------
echo "==================== QLSENSE Slurm job ===================="
echo "job id / name   : ${SLURM_JOB_ID:-<none>} / ${SLURM_JOB_NAME:-<none>}"
echo "submitted by    : ${SLURM_JOB_USER:-$USER}"
echo "started         : $(date --iso-8601=seconds)"
echo "host            : $(hostname)"
echo "partition / qos : ${SLURM_JOB_PARTITION:-<none>} / ${SLURM_JOB_QOS:-<none>}"
echo "nproc (visible) : $(nproc)"
echo "cpus-per-task   : ${SLURM_CPUS_PER_TASK:-<unset>}  -> thread limit $CPUS"
echo "mem requested   : ${SLURM_MEM_PER_NODE:-<unset>} MB"
echo "repo dir        : $REPO_DIR"
echo "venv            : $VENV"
echo "python          : $(command -v python) ($(python --version 2>&1))"
echo "JAX_PLATFORMS   : $JAX_PLATFORMS"
echo "-----------------------------------------------------------"
echo "N (spins)       : $N   (Hilbert space 2^N = $((2 ** N)))"
echo "N_B / N_n       : $N_B / $N_N  -> $((N_B * N_N)) records"
echo "beta            : $BETA"
echo "epochs          : $EPOCHS"
echo "batch size      : $BATCH_SIZE"
echo "seed            : $SEED"
echo "data dir        : $DATA_DIR"
echo "results dir     : $RESULTS_DIR"
echo "==========================================================="
echo

# One task, one process: srun would add a launch layer for no benefit here.
./run_qls.sh
