"""Joint optimization of the filter Theta and the sensor Phi, eqs (4)-(6).

The trainable state is a single pytree holding

    Theta: an Equinox MLP  f_Theta : R^N -> R^N   (eq (4)),
    Phi:   the sensor parameters {J_ij}, {B_i}    (eq (6)),

and the objective is the Monte-Carlo estimate of eq (5),

    F(Theta, Phi) = E_{dB, Xi, Lambda} || f_Theta(S(Theta, Phi)) - dB ||_2^2 .

The crucial point is that ``S`` is *not* read from the dataset: it depends on
Phi, so every gradient step rebuilds the Hamiltonian of eq (1) and re-runs the
exact diagonalization of eqs (2)-(3) on the recorded ``(dB, Xi, Lambda)``
realizations. Gradients flow through the eigendecomposition into ``J`` and
``B``, which is what makes this sensor engineering rather than plain
supervised learning.

The script trains twice from the same initialization: once with Phi frozen at
the values used to generate the data (the baseline: filter-only optimization)
and once with Phi co-optimized. The gap between the two validation losses is
the measured gain from engineering the sensor.
"""

from __future__ import annotations

import argparse
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from generate_data import column_slices, read_dataset
from ising import readout


class SensorFilter(eqx.Module):
    """Theta (the MLP) and Phi = {J, B} living in one differentiable pytree."""

    mlp: eqx.nn.MLP
    J: jax.Array
    B: jax.Array

    def __init__(self, N: int, width: int, depth: int, J0, B0, *, key):
        self.mlp = eqx.nn.MLP(
            in_size=N,
            out_size=N,
            width_size=width,
            depth=depth,
            activation=jnp.tanh,
            key=key,
        )
        self.J = jnp.asarray(J0)
        self.B = jnp.asarray(B0)


def make_loss(N: int, beta: float, method: str):
    """Eq (5) on a batch of recorded ``(dB, Xi, Lambda)`` realizations."""

    def loss_fn(model: SensorFilter, dB, xi, lam):
        S = jax.vmap(lambda d, x, l: readout(model.J, model.B, d, x, l, N, beta, method))(
            dB, xi, lam
        )
        pred = jax.vmap(model.mlp)(S)
        return jnp.mean(jnp.sum((pred - dB) ** 2, axis=-1))

    return loss_fn


def make_optimizer(model: SensorFilter, lr_theta: float, lr_phi: float, freeze_phi: bool):
    """Adam on Theta, and either Adam or a hard freeze on Phi."""
    params = eqx.filter(model, eqx.is_inexact_array)
    labels = jax.tree_util.tree_map(lambda _: "theta", params)
    labels = eqx.tree_at(lambda t: (t.J, t.B), labels, replace=("phi", "phi"))
    transforms = {
        "theta": optax.adam(lr_theta),
        "phi": optax.set_to_zero() if freeze_phi else optax.adam(lr_phi),
    }
    return optax.multi_transform(transforms, labels)


def evaluate(loss_fn, model, data, batch_size: int) -> float:
    """Loss over a full split, accumulated in chunks to bound peak memory."""
    dB, xi, lam = data
    n = dB.shape[0]
    total = 0.0
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        total += float(loss_fn(model, dB[start:stop], xi[start:stop], lam[start:stop])) * (
            stop - start
        )
    return total / n


def train(model0, train_data, val_data, *, N, beta, method, args, freeze_phi, tag):
    loss_fn = eqx.filter_jit(make_loss(N, beta, method))
    optim = make_optimizer(model0, args.lr_theta, args.lr_phi, freeze_phi)

    @eqx.filter_jit
    def step(model, opt_state, dB, xi, lam):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, dB, xi, lam)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_inexact_array)
        )
        return eqx.apply_updates(model, updates), opt_state, loss

    model = model0
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    dB, xi, lam = train_data
    n = dB.shape[0]
    rng = np.random.default_rng(args.shuffle_seed)

    history = [(0, evaluate(loss_fn, model, train_data, args.batch_size),
                evaluate(loss_fn, model, val_data, args.batch_size))]
    print(f"[{tag}] epoch {0:4d}  train {history[0][1]:.6f}  val {history[0][2]:.6f}")

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(n)
        epoch_loss, seen = 0.0, 0
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            model, opt_state, loss = step(model, opt_state, dB[idx], xi[idx], lam[idx])
            epoch_loss += float(loss) * len(idx)
            seen += len(idx)
        val_loss = evaluate(loss_fn, model, val_data, args.batch_size)
        history.append((epoch, epoch_loss / seen, val_loss))
        if epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"[{tag}] epoch {epoch:4d}  train {epoch_loss / seen:.6f}  val {val_loss:.6f}")

    return model, history


def split_by_signal(rows: np.ndarray, sl, val_frac: float, seed: int):
    """Hold out whole signals, so validation signals are never seen in training."""
    signal_id = rows[:, sl["signal_id"]].astype(int).ravel()
    unique = np.unique(signal_id)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_val = max(1, int(round(val_frac * len(unique))))
    val_signals = set(shuffled[:n_val].tolist())
    is_val = np.array([s in val_signals for s in signal_id])
    return ~is_val, is_val


def pack(rows: np.ndarray, mask: np.ndarray, sl, N: int):
    sub = rows[mask]
    dB = jnp.asarray(sub[:, sl["dB"]])
    xi = jnp.asarray(sub[:, sl["xi"]]).reshape(-1, 3, N)
    lam = jnp.asarray(sub[:, sl["lam"]])
    return dB, xi, lam


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="dataset written by generate_data.py")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr-theta", dest="lr_theta", type=float, default=3e-3)
    p.add_argument("--lr-phi", dest="lr_phi", type=float, default=1e-2)
    p.add_argument("--width", type=int, default=64, help="MLP hidden width")
    p.add_argument("--depth", type=int, default=2, help="MLP hidden layers")
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0, help="MLP initialization seed")
    p.add_argument("--split-seed", dest="split_seed", type=int, default=0)
    p.add_argument("--shuffle-seed", dest="shuffle_seed", type=int, default=0)
    p.add_argument("--log-every", dest="log_every", type=int, default=10)
    p.add_argument("--readout-method", dest="readout_method", default=None,
                   choices=("eigh", "expm"), help="default: whatever the dataset used")
    p.add_argument("--beta", type=float, default=None, help="default: the dataset's beta")
    p.add_argument("--results-dir", dest="results_dir", default="results")
    p.add_argument("--skip-baseline", dest="skip_baseline", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    rows, meta = read_dataset(args.data)
    N = int(meta["N"])
    beta = args.beta if args.beta is not None else float(meta["beta"])
    method = args.readout_method or meta.get("readout_method", "eigh")
    J0 = np.fromstring(meta["J0"], sep=" ")
    B0 = np.fromstring(meta["B0"], sep=" ")

    sl = column_slices(N)
    train_mask, val_mask = split_by_signal(rows, sl, args.val_frac, args.split_seed)
    train_data = pack(rows, train_mask, sl, N)
    val_data = pack(rows, val_mask, sl, N)
    print(
        f"N={N} beta={beta} readout={method}  "
        f"train {train_data[0].shape[0]} records / val {val_data[0].shape[0]} records"
    )

    model0 = SensorFilter(N, args.width, args.depth, J0, B0, key=jax.random.key(args.seed))
    os.makedirs(args.results_dir, exist_ok=True)

    results = {}
    runs = [("joint", False)] if args.skip_baseline else [("frozen_phi", True), ("joint", False)]
    for tag, freeze_phi in runs:
        model, history = train(
            model0,
            train_data,
            val_data,
            N=N,
            beta=beta,
            method=method,
            args=args,
            freeze_phi=freeze_phi,
            tag=tag,
        )
        np.savetxt(
            os.path.join(args.results_dir, f"history_{tag}.txt"),
            np.asarray(history),
            fmt=["%d", "%.10e", "%.10e"],
            header="epoch train_loss val_loss",
        )
        eqx.tree_serialise_leaves(os.path.join(args.results_dir, f"model_{tag}.eqx"), model)
        np.savetxt(
            os.path.join(args.results_dir, f"phi_{tag}.txt"),
            np.concatenate([np.asarray(model.J), np.asarray(model.B)])[None, :],
            fmt="%.10e",
            header=f"J[0..{N - 2}] B[0..{N - 1}]",
        )
        results[tag] = {
            "initial_val_loss": history[0][2],
            "final_train_loss": history[-1][1],
            "final_val_loss": history[-1][2],
            "J": np.asarray(model.J).tolist(),
            "B": np.asarray(model.B).tolist(),
        }

    print("\n=== eq (5) validation loss ===")
    for tag in results:
        print(f"  {tag:<12s} final val loss = {results[tag]['final_val_loss']:.6f}")
    if "frozen_phi" in results and "joint" in results:
        base = results["frozen_phi"]["final_val_loss"]
        joint = results["joint"]["final_val_loss"]
        gain = 100.0 * (base - joint) / base
        results["relative_gain_percent"] = gain
        print(f"  gain from co-optimizing Phi = {gain:+.2f}%")
        print(f"  optimized J = {np.round(results['joint']['J'], 4).tolist()}")
        print(f"  optimized B = {np.round(results['joint']['B'], 4).tolist()}")

    summary_path = os.path.join(args.results_dir, "summary.json")
    with open(summary_path, "w") as handle:
        json.dump({"N": N, "beta": beta, "readout_method": method, **results}, handle, indent=2)
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
