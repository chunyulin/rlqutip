import os

# Prevent JAX from preallocating nearly all GPU memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import re
import time
import numpy as np
import jax
import jax.numpy as jnp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


HS = 18
torch_device = torch.device("cpu")

# Use double precision for stable long lab-frame phase accumulation.
jax.config.update("jax_enable_x64", True)


def require_jax_gpu():
    gpus = jax.devices("gpu")
    if not gpus:
        raise RuntimeError(
            "No JAX GPU was found. Install a CUDA-enabled JAX build and check "
            "that the NVIDIA driver is visible."
        )
    gpu = gpus[0]
    print(f"[JAX Setup] Quantum evolution device: {gpu}")
    return gpu


JAX_GPU = require_jax_gpu()

SX = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
SZ = jnp.array([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
X_GATE = SX


@jax.jit
def _evolve_one_segment(rho_batch, amp_batch, phi_batch, t_start, dt,
                        omega_q, omega_p, n_substeps):
    """Closed-system rho -> U rho U^dagger for one piecewise pulse segment."""
    h = dt / n_substeps
    hz = -0.5 * omega_q

    def substep(k, rho):
        # Midpoint rule for the time-dependent drive coefficient.
        t_mid = t_start + (k + 0.5) * h
        hx = amp_batch * jnp.sin(phi_batch + omega_p * t_mid)
        norm = jnp.sqrt(hx * hx + hz * hz)

        # exp[-i h (hx*sx + hz*sz)] in closed form.
        c = jnp.cos(norm * h)
        s_over_norm = jnp.sin(norm * h) / norm
        hmat = hx[:, None, None] * SX + hz * SZ
        eye = jnp.eye(2, dtype=jnp.complex128)[None, :, :]
        u = c[:, None, None] * eye - 1j * s_over_norm[:, None, None] * hmat
        return u @ rho @ jnp.swapaxes(jnp.conj(u), -1, -2)

    return jax.lax.fori_loop(0, n_substeps, substep, rho_batch)


@jax.jit
def _pure_target_fidelity(rho_batch, psi0_batch):
    psi_target = psi0_batch @ jnp.swapaxes(X_GATE, -1, -2)
    value = jnp.einsum("bi,bij,bj->b", jnp.conj(psi_target), rho_batch,
                       psi_target)
    return jnp.clip(jnp.real(value), 0.0, 1.0)


SIX_PAULI_STATES = np.array(
    [
        [1.0, 0.0],                              # +Z = |0>
        [0.0, 1.0],                              # -Z = |1>
        [1.0, 1.0],                              # +X = |+>
        [1.0, -1.0],                             # -X = |->
        [1.0, 1.0j],                             # +Y = |+i>
        [1.0, -1.0j],                            # -Y = |-i>
    ],
    dtype=np.complex128,
)
SIX_PAULI_STATES[2:] /= np.sqrt(2.0)


class BatchedSingleQubitClosedJAX:
    """Each episode applies one shared pulse to all six Pauli eigenstates."""

    def __init__(self, batch_size, n_segments=8, pulse_duration=20.0,
                 n_gate_reps=1, n_substeps=64):
        self.batch_size = batch_size
        self.n_segments = n_segments
        self.pulse_duration = pulse_duration
        self.dt = pulse_duration / n_segments
        self.n_gate_reps = n_gate_reps
        self.n_substeps = n_substeps
        self.n_test_states = 6
        self.omega_q = 2.0 * np.pi * 5.0
        self.omega_p = self.omega_q
        self.current_step = 0
        self.psi0 = None
        self.rho = None

    def reset(self):
        # Layout: [episode 0: six states, episode 1: six states, ...].
        psi0 = np.tile(SIX_PAULI_STATES, (self.batch_size, 1))
        self.psi0 = jax.device_put(jnp.asarray(psi0, dtype=jnp.complex128),
                                   JAX_GPU)
        self.rho = self.psi0[:, :, None] * jnp.conj(self.psi0[:, None, :])
        self.current_step = 0
        return self.observations()

    def observations(self):
        # The policy must not see the input state. It sees progress only.
        return np.full(
            (self.batch_size, 1),
            self.current_step / self.n_segments,
            dtype=np.float32,
        )

    def step(self, actions):
        actions_jax = jax.device_put(jnp.asarray(actions, dtype=jnp.float64),
                                     JAX_GPU)
        # One action per episode, repeated for its six simultaneously evolved
        # input states. Thus all six states see exactly the same pulse.
        actions_six = jnp.repeat(actions_jax, self.n_test_states, axis=0)
        t_start = self.current_step * self.dt
        self.rho = _evolve_one_segment(
            self.rho,
            actions_six[:, 0],
            actions_six[:, 1],
            t_start,
            self.dt,
            self.omega_q,
            self.omega_p,
            self.n_substeps,
        )
        self.current_step += 1
        done = self.current_step >= self.n_segments

        if done:
            fidelity_six = _pure_target_fidelity(self.rho, self.psi0).reshape(
                self.batch_size, self.n_test_states
            )
            # One scalar score per candidate pulse sequence (episode).
            base_fidelity = jnp.mean(fidelity_six, axis=1)
            reward = base_fidelity ** self.n_gate_reps
            base_fidelity = np.asarray(jax.device_get(base_fidelity))
            reward = np.asarray(jax.device_get(reward))
        else:
            base_fidelity = np.zeros(self.batch_size, dtype=np.float64)
            reward = np.zeros(self.batch_size, dtype=np.float64)

        return self.observations(), reward, done, base_fidelity


class ContinuousPyTorchPolicy(nn.Module):
    def __init__(self, state_dim=1, action_dim=2, hidden_size=HS):
        super().__init__()
        self.max_amp = 2.0 * np.pi * 0.1
        self.max_phi = 2.0 * np.pi
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Sigmoid(),
        )

    def forward(self, state):
        raw = self.fc(state)
        return torch.cat(
            (raw[..., 0:1] * self.max_amp, raw[..., 1:2] * self.max_phi), dim=-1
        )


def play_episodes_jax(env, policy, std_val):
    """Vectorized policy sampling plus batched JAX quantum evolution."""
    state = env.reset()
    states, actions, rewards = [], [], []

    with torch.no_grad():
        for _ in range(env.n_segments):
            state_t = torch.as_tensor(state, dtype=torch.float32,
                                      device=torch_device)
            mu = policy(state_t)
            std = torch.tensor(
                [std_val * policy.max_amp, std_val * policy.max_phi],
                dtype=torch.float32,
                device=torch_device,
            )
            action_t = Normal(mu, std).sample()
            action = action_t.cpu().numpy()
            next_state, reward, done, base_fidelity = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

    # [time, batch, ...] -> [batch, time, ...]
    states_t = torch.as_tensor(np.stack(states, axis=1), dtype=torch.float32,
                               device=torch_device)
    actions_t = torch.as_tensor(np.stack(actions, axis=1), dtype=torch.float32,
                                device=torch_device)
    mu = policy(states_t)
    std = torch.tensor(
        [std_val * policy.max_amp, std_val * policy.max_phi],
        dtype=torch.float32,
        device=torch_device,
    )
    log_probs = Normal(mu, std).log_prob(actions_t).sum(dim=-1)

    # Rewards are zero except on the final step, matching the original code.
    rewards_bt = np.stack(rewards, axis=1)
    return log_probs, rewards_bt, base_fidelity


def update_policy_reinforce(policy, optimizer, batch_log_probs, rewards, eps=1e-8):
    returns = np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1).copy()
    returns_t = torch.as_tensor(returns, dtype=torch.float32,
                                device=batch_log_probs.device)
    advantages = returns_t - returns_t.mean(dim=0, keepdim=True)
    total_return = returns_t[:, 0].sum() + eps
    loss = -(batch_log_probs * advantages).sum() / total_return

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def checkpoint_payload(iteration, policy, optimizer, scheduler, best_fidelity):
    return {
        "iteration": iteration,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_mean_fidelity": best_fidelity,
    }


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    batch_size = 128
    num_iterations = 2000
    learning_rate = 5e-3
    env = BatchedSingleQubitClosedJAX(
        batch_size=batch_size,
        n_segments=10,
        pulse_duration=22.0,
        n_gate_reps=1,
        n_substeps=64,
    )
    policy = ContinuousPyTorchPolicy(state_dim=1, action_dim=2).to(torch_device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=1e-4
    )

    # Old 9-input checkpoints are structurally incompatible with this genuine
    # state-independent gate-calibration policy.
    checkpoint_dir = "./ckpt_1q_jax_6states"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "best_policy.pt")
    start_iteration = 0
    best_mean_base_fid = -1.0

    pattern = re.compile(r"checkpoint_iter_(\d+)\.pt$")
    candidates = []
    for filename in os.listdir(checkpoint_dir):
        match = pattern.fullmatch(filename)
        if match:
            candidates.append((int(match.group(1)), filename))
    if candidates:
        _, filename = max(candidates)
        resume_path = os.path.join(checkpoint_dir, filename)
        checkpoint = torch.load(resume_path, map_location=torch_device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_iteration = checkpoint["iteration"]
        best_mean_base_fid = checkpoint.get("best_mean_fidelity", -1.0)
        print(f"[Resumed] {resume_path}; next iteration {start_iteration + 1}")
    else:
        print("[Checkpoint] Starting from scratch.")

    # Compile once before timing the training loop.
    print("[JAX] Compiling batched propagator ...")
    env.reset()
    env.step(np.zeros((batch_size, 2), dtype=np.float64))
    jax.block_until_ready(env.rho)
    print("[JAX] Compilation complete.")

    t0 = time.time()
    for it in range(start_iteration, num_iterations):
        std_val = max(0.01, 0.3 * (1.0 - it / num_iterations))
        log_probs, rewards, base_fids = play_episodes_jax(env, policy, std_val)
        loss = update_policy_reinforce(policy, optimizer, log_probs, rewards)
        scheduler.step()

        mean_fid = float(np.mean(base_fids))
        max_fid = float(np.max(base_fids))
        elapsed = time.time() - t0
        current_iter = it + 1
        print(
            f"Iteration {current_iter:4d}/{num_iterations} | "
            f"Mean Fid: {mean_fid:.6f} | Max Fid: {max_fid:.6f} | "
            f"Best Mean: {max(best_mean_base_fid, mean_fid):.6f} | "
            f"Loss: {loss:.6f} | T: {elapsed:.2f} s"
        )
        t0 = time.time()

        if mean_fid > best_mean_base_fid:
            best_mean_base_fid = mean_fid
            torch.save(
                checkpoint_payload(current_iter, policy, optimizer, scheduler,
                                   best_mean_base_fid),
                best_path,
            )
            print(f"  -> saved best checkpoint ({best_mean_base_fid:.6f})")

        if current_iter % 1000 == 0:
            periodic_path = os.path.join(
                checkpoint_dir, f"checkpoint_iter_{current_iter}.pt"
            )
            torch.save(
                checkpoint_payload(current_iter, policy, optimizer, scheduler,
                                   best_mean_base_fid),
                periodic_path,
            )
            print(f"  -> saved periodic checkpoint at {current_iter}")

    print(f"[SUCCESS] Best mean fidelity: {best_mean_base_fid:.6f}")