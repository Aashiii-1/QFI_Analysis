#!/usr/bin/env bash
# Driver for the Ising quantum-sensing pipeline: generate a dataset by exact
# diagonalization, then jointly optimize the filter Theta and the sensor
# Phi = {J, B} through the physics.
#
# Every knob is an overridable environment variable, e.g.
#   N=4 N_B=32 EPOCHS=20 ./run_qls.sh
#
# N=6 keeps the Hilbert space at 64 states, so full ED across the whole batch
# stays cheap.
set -euo pipefail

cd "$(dirname "$0")"

N=${N:-6}                 # spins
N_N=${N_N:-8}             # noise realizations per signal
N_B=${N_B:-64}            # signals drawn from eq (7)
BETA=${BETA:-1.0}         # inverse temperature in eq (3)
SIGMA1=${SIGMA1:-0.05}    # on-site noise std (Xi)
SIGMA2=${SIGMA2:-0.05}    # bond noise std (Lambda)
ELL=${ELL:-1.0}           # noise correlation length
M_MODES=${M_MODES:-3}     # M in eq (7)
AMP_STD=${AMP_STD:-0.25}
J0=${J0:-0.5}             # initial uniform coupling
B0=${B0:-1.0}             # initial uniform transverse field
EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-64}
LR_THETA=${LR_THETA:-3e-3}
LR_PHI=${LR_PHI:-1e-2}
WIDTH=${WIDTH:-64}
DEPTH=${DEPTH:-2}
READOUT=${READOUT:-eigh}  # eigh | expm
VAL_FRAC=${VAL_FRAC:-0.25}
LOG_EVERY=${LOG_EVERY:-10}
SEED=${SEED:-0}
DATA_DIR=${DATA_DIR:-data}
RESULTS_DIR=${RESULTS_DIR:-results}

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi

mkdir -p "$DATA_DIR" "$RESULTS_DIR"
DATA="${DATA_DIR}/qls_N${N}.txt"

echo "=== stage 1/2: generating data (N=${N}, N_B=${N_B}, N_n=${N_N}, beta=${BETA}) ==="
"$PYTHON" src/generate_data.py \
  --N "$N" --N_B "$N_B" --N_n "$N_N" --beta "$BETA" \
  --sigma1 "$SIGMA1" --sigma2 "$SIGMA2" --ell "$ELL" \
  --n-modes "$M_MODES" --amp-std "$AMP_STD" \
  --J0 "$J0" --B0 "$B0" \
  --readout-method "$READOUT" --seed "$SEED" --out "$DATA"

echo
echo "=== stage 2/2: training (epochs=${EPOCHS}) ==="
"$PYTHON" src/ml_train.py \
  --data "$DATA" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --lr-theta "$LR_THETA" --lr-phi "$LR_PHI" \
  --width "$WIDTH" --depth "$DEPTH" \
  --val-frac "$VAL_FRAC" --log-every "$LOG_EVERY" \
  --readout-method "$READOUT" --seed "$SEED" \
  --results-dir "$RESULTS_DIR"
