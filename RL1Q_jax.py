HS = 18

import os
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random
import optax

import torch

BATCH_SIZE = 320
NUM_ITERATIONS = 2000
LEARNING_RATE = 5e-3

OMEGA_Q = 2 * np.pi * 4.81
OMEGA_P = OMEGA_Q
N_SEGMENTS = 10
PULSE_DURATION = 22.4
DT = PULSE_DURATION / N_SEGMENTS

MAX_AMP = 2 * np.pi * 0.1
MAX_PHI = 2.0 * np.pi

sigma_x = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
sigma_z = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
I2 = jnp.eye(2, dtype=jnp.complex128)
H0 = -0.5 * OMEGA_Q * sigma_z
H1 = sigma_x

ket0 = jnp.array([[1], [0]], dtype=jnp.complex128)
ket1 = jnp.array([[0], [1]], dtype=jnp.complex128)
input_kets = [
    ket0,
    ket1,
    (ket0 + ket1) / jnp.sqrt(2),
    (ket0 - ket1) / jnp.sqrt(2),
    (ket0 + 1j * ket1) / jnp.sqrt(2),
    (ket0 - 1j * ket1) / jnp.sqrt(2),
]
INPUT_RHOS = jnp.stack([k @ k.conj().T for k in input_kets])

c4 = np.cos(np.pi / 4)
s4 = np.sin(np.pi / 4)
U_target = jnp.array([
    [c4, -1j * s4],
    [-1j * s4, c4]
], dtype=jnp.complex128)
TARGET_RHOS = jnp.einsum("aj,ijl,bl->iab", U_target, INPUT_RHOS, U_target.conj())


def _propagator(amp, phi, t_start):
    dt_sub = DT / 10
    U_total = I2
    for k in range(10):
        t_mid = t_start + (k + 0.5) * dt_sub
        drive = amp * jnp.sin(phi + OMEGA_P * t_mid)
        H = H0 + drive * H1
        U_sub = jax.scipy.linalg.expm(-1j * H * dt_sub)
        U_total = U_sub @ U_total
    return U_total


def evolve_episode(amps, phis):
    rhos = INPUT_RHOS
    for s in range(N_SEGMENTS):
        U = _propagator(amps[s], phis[s], s * DT)
        rhos = jnp.einsum("aj,ijl,bl->iab", U, rhos, U.conj())
    return rhos


evolve_batch = jax.jit(jax.vmap(evolve_episode))


def compute_fid(rhos_final):
    return jnp.real(jnp.einsum("cjk,bckj->bc", TARGET_RHOS, rhos_final))


@jax.jit
def policy_forward(params, progress):
    w1, b1, w2, b2 = params
    state = progress[:, None]
    h = jnp.tanh(state @ w1.T + b1)
    raw = jax.nn.sigmoid(h @ w2.T + b2)
    mu_amp = raw[..., 0:1] * MAX_AMP
    mu_phi = raw[..., 1:2] * MAX_PHI
    return jnp.concatenate([mu_amp, mu_phi], axis=-1)


@jax.jit
def loss_and_sample(params, rng, std_val):
    progress = jnp.arange(N_SEGMENTS, dtype=jnp.float32) / N_SEGMENTS
    mu = policy_forward(params, progress)

    keys = random.split(rng, BATCH_SIZE)

    def _sample_one(key):
        k_amp, k_phi = random.split(key)
        noise_amp = random.normal(k_amp, mu[..., 0:1].shape, dtype=jnp.float32)
        noise_phi = random.normal(k_phi, mu[..., 1:2].shape, dtype=jnp.float32)
        sampled_amp = mu[..., 0:1] + noise_amp * std_val * MAX_AMP
        sampled_phi = mu[..., 1:2] + noise_phi * std_val * MAX_PHI
        actions = jnp.concatenate([sampled_amp, sampled_phi], axis=-1)
        lp_amp = -0.5 * ((actions[..., 0:1] - mu[..., 0:1]) / (std_val * MAX_AMP)) ** 2 \
                 - jnp.log(std_val * MAX_AMP)
        lp_phi = -0.5 * ((actions[..., 1:2] - mu[..., 1:2]) / (std_val * MAX_PHI)) ** 2 \
                 - jnp.log(std_val * MAX_PHI)
        log_prob = (lp_amp + lp_phi).sum()
        return actions, log_prob

    actions_batch, log_probs_batch = jax.vmap(_sample_one)(keys)
    return actions_batch, log_probs_batch


@jax.jit
def compute_loss(params, actions, log_probs, rewards):
    advantages = rewards - jnp.mean(rewards)
    loss = -jnp.mean(log_probs * advantages)
    return loss


value_and_grad_fn = jax.jit(jax.value_and_grad(compute_loss))


def save_checkpoint(path, iteration, policy_params, best_mean_fid):
    w1, b1, w2, b2 = policy_params
    sd = {
        "w1": torch.from_numpy(np.array(w1)),
        "b1": torch.from_numpy(np.array(b1)),
        "w2": torch.from_numpy(np.array(w2)),
        "b2": torch.from_numpy(np.array(b2)),
    }
    torch.save({
        "iteration": iteration,
        "model_state_dict": sd,
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "best_mean_fidelity": best_mean_fid,
    }, path)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    params = (
        jnp.array(sd["w1"].numpy()),
        jnp.array(sd["b1"].numpy()),
        jnp.array(sd["w2"].numpy()),
        jnp.array(sd["b2"].numpy()),
    )
    return params, ckpt["iteration"], ckpt.get("best_mean_fidelity", -1.0)


def init_params(key):
    k1, k2 = random.split(key)
    w1 = random.normal(k1, (HS, 1), dtype=jnp.float32) * jnp.sqrt(2.0)
    b1 = jnp.zeros(HS, dtype=jnp.float32)
    w2 = random.normal(k2, (2, HS), dtype=jnp.float32) * jnp.sqrt(2.0 / HS)
    b2 = jnp.zeros(2, dtype=jnp.float32)
    return (w1, b1, w2, b2)


def main():
    gpus = jax.devices("gpu")
    if gpus:
        print(f"[JAX Setup] Using GPU: {gpus[0]}")
    else:
        print("[JAX Setup] No GPU found, using CPU.")

    rng = random.PRNGKey(42)
    params = init_params(rng)

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=LEARNING_RATE),
    )
    opt_state = tx.init(params)

    checkpoint_dir = "./ckpt_1q_jax"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_policy.pt")

    start_iteration = 0
    best_mean_fid = -1.0

    existing_ckpts = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_iter_") and f.endswith(".pt")
    ]

    if existing_ckpts:
        latest_ckpt_file = sorted(
            existing_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0])
        )[-1]
        resume_path = os.path.join(checkpoint_dir, latest_ckpt_file)
        print(f"\n[Checkpoint Found] Loading state from '{resume_path}'...")
        params, start_iteration, best_mean_fid = load_checkpoint(resume_path)
        print(f"[Resumed] Resuming training from Iteration {start_iteration + 1} / {NUM_ITERATIONS}\n")
    else:
        print(f"\n[No Periodic Checkpoint Found] Starting training from scratch.\n")

    rng = random.PRNGKey(0)

    t0 = time.time()
    for it in range(start_iteration, NUM_ITERATIONS):
        std_val = max(0.01, 0.3 * (1.0 - it / NUM_ITERATIONS))

        rng, sub_rng = random.split(rng)
        actions, log_probs = loss_and_sample(params, sub_rng, std_val)

        amps = actions[..., 0]
        phis = actions[..., 1]
        final_states = evolve_batch(amps, phis)
        fids = compute_fid(final_states)
        rewards = jnp.mean(fids, axis=1)

        loss_val, grads = value_and_grad_fn(params, actions, log_probs, rewards)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        mean_fid = float(jnp.mean(rewards))
        max_fid = float(jnp.max(rewards))

        t1 = time.time()
        print(f"Iteration {it+1:4d}/{NUM_ITERATIONS} | "
              f"Mean Fid: {mean_fid:.4f} | "
              f"Max Fid: {max_fid:.4f} | "
              f"Best Mean: {max(best_mean_fid, mean_fid):.4f} | "
              f"Loss: {float(loss_val):.4f} | "
              f"T: {t1-t0:.2f} s")
        t0 = t1

        current_iter = it + 1

        if mean_fid > best_mean_fid:
            best_mean_fid = mean_fid
            save_checkpoint(best_checkpoint_path, current_iter, params, best_mean_fid)
            print(f"  └─> [Saved Best Checkpoint] Improved Mean Fidelity to {best_mean_fid:.4f}")

        if current_iter % 1000 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{current_iter}.pt")
            save_checkpoint(periodic_path, current_iter, params, best_mean_fid)
            print(f"  └─> [Periodic Checkpoint] Saved checkpoint at iteration {current_iter}")

    print("\n[SUCCESS] Training session completed!")
    print(f"[FINAL] Best Overall Mean Fidelity Achieved: {best_mean_fid:.4f}")


if __name__ == "__main__":
    main()
