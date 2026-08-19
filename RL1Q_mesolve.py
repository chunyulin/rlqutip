HS = 18

import os
import time
import numpy as np
import qutip as qt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from joblib import Parallel, delayed

device = torch.device("cpu")

N_SEGMENTS = 10
PULSE_DURATION = 22.4
DT = PULSE_DURATION / N_SEGMENTS
OMEGA_Q = 2 * np.pi * 4.81
OMEGA_P = OMEGA_Q

BATCH_SIZE = 320
NUM_ITERATIONS = 2000
LEARNING_RATE = 5e-3


def get_pauli_eigenstates():
    ket0 = qt.basis(2, 0)
    ket1 = qt.basis(2, 1)
    ket_plus = (ket0 + ket1).unit()
    ket_minus = (ket0 - ket1).unit()
    ket_plus_i = (ket0 + 1j * ket1).unit()
    ket_minus_i = (ket0 - 1j * ket1).unit()
    return [ket0, ket1, ket_plus, ket_minus, ket_plus_i, ket_minus_i]


class SingleQubitGateControlEnv:
    def __init__(self, n_segments=N_SEGMENTS, pulse_duration=PULSE_DURATION):
        self.n_segments = n_segments
        self.pulse_duration = pulse_duration
        self.dt = pulse_duration / n_segments
        self.current_step = 0

        self.omega_q = OMEGA_Q
        self.omega_p = OMEGA_P

        self.H_int = -0.5 * self.omega_q * qt.sigmaz()
        self.H_drive_op = qt.sigmax()

        # Rx(pi/2) = exp(-i * pi/4 * sigma_x)  (sqrt of X gate)
        theta = np.pi / 2
        self.U_target = qt.Qobj([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ])
        self.input_states = get_pauli_eigenstates()
        self.target_states = [qt.ket2dm(self.U_target * psi) for psi in self.input_states]

        self.current_states = None

    def reset(self):
        self.current_step = 0
        self.current_states = [qt.ket2dm(psi) for psi in self.input_states]
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.current_step / self.n_segments], dtype=np.float32)

    def step(self, action):
        amp_k, phi_k = action[0], action[1]

        t_start = self.current_step * self.dt
        t_end = (self.current_step + 1) * self.dt
        tlist_segment = np.linspace(t_start, t_end, 10)

        def drive_coeff(t, **kwargs):
            return amp_k * np.sin(phi_k + self.omega_p * t)

        H_segment = [self.H_int, [self.H_drive_op, drive_coeff]]

        new_states = []
        for rho in self.current_states:
            result = qt.mesolve(H_segment, rho, tlist_segment,
                                options={'store_final_state': True})
            new_states.append(result.states[-1])
        self.current_states = new_states

        self.current_step += 1
        done = self.current_step >= self.n_segments

        reward = 0.0
        fidelities = []

        if done:
            for rho_target, rho_final in zip(self.target_states, self.current_states):
                fid = float(qt.fidelity(rho_final, rho_target) ** 2)
                fidelities.append(fid)
            reward = np.mean(fidelities)

        return self._get_obs(), reward, done, False, {"fidelities": fidelities}


class ContinuousPyTorchPolicy(nn.Module):
    def __init__(self, state_dim=1, action_dim=2, hidden_size=HS):
        super().__init__()
        self.max_amp = 2 * np.pi * 0.1
        self.max_phi = 2.0 * np.pi

        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim),
            nn.Sigmoid()
        )

    def forward(self, state):
        raw_out = self.fc(state)
        mu_amp = raw_out[..., 0:1] * self.max_amp
        mu_phi = raw_out[..., 1:2] * self.max_phi
        return torch.cat([mu_amp, mu_phi], dim=-1)

    def select_action(self, state, std_val):
        state_t = torch.FloatTensor(state).to(device)
        mu = self.forward(state_t)
        std = torch.tensor([std_val * self.max_amp, std_val * self.max_phi], device=device)
        dist = Normal(mu, std)
        action_tensor = dist.sample()
        return action_tensor.detach().numpy(), None


def _run_single_episode_worker(args):
    policy_state_dict, std_val, env_kwargs, task_id = args

    worker_seed = (int(time.time() * 1000) & 0xFFFF) ^ (os.getpid() << 8) ^ (task_id * 10007)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed % (2**32 - 1))

    local_env = SingleQubitGateControlEnv(**env_kwargs)
    local_policy = ContinuousPyTorchPolicy(state_dim=1, action_dim=2).to(device)
    local_policy.load_state_dict(policy_state_dict)

    state = local_env.reset()
    ep_states, ep_actions, ep_rewards = [], [], []

    while True:
        action, _ = local_policy.select_action(state, std_val)
        next_state, reward, done, truncated, info = local_env.step(action)
        ep_states.append(state)
        ep_actions.append(action)
        ep_rewards.append(reward)
        state = next_state
        if done:
            return (np.array(ep_states, dtype=np.float32),
                    np.array(ep_actions, dtype=np.float32),
                    ep_rewards, info)


def play_episodes_parallel(env_kwargs, policy, batch_size, std_val, max_workers):
    policy_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    tasks = [(policy_state_dict, std_val, env_kwargs, i) for i in range(batch_size)]
    results = Parallel(n_jobs=max_workers, backend="loky")(
        delayed(_run_single_episode_worker)(task) for task in tasks
    )

    batch_states_list = [res[0] for res in results]
    batch_actions_list = [res[1] for res in results]
    batch_raw_rewards = [res[2] for res in results]
    batch_final_info = [res[3] for res in results]

    states_tensor = torch.tensor(np.array(batch_states_list), dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(np.array(batch_actions_list), dtype=torch.float32, device=device)

    mu = policy(states_tensor)
    std = torch.tensor([std_val * policy.max_amp, std_val * policy.max_phi], device=device)
    dist = Normal(mu, std)
    batch_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)

    return batch_log_probs, batch_raw_rewards, batch_final_info


def update_policy_reinforce(policy, optimizer, batch_log_probs, batch_raw_rewards, eps=1e-8):
    batch_size = len(batch_raw_rewards)

    trajectory_returns_to_go = []
    for j in range(batch_size):
        rw_list = batch_raw_rewards[j]
        g = 0.0
        returns_j = []
        for r in reversed(rw_list):
            g = r + g
            returns_j.insert(0, g)
        trajectory_returns_to_go.append(returns_j)

    returns_tensor = torch.tensor(
        trajectory_returns_to_go, dtype=torch.float32, device=batch_log_probs.device
    )

    step_baselines = returns_tensor.mean(dim=0, keepdim=True)
    advantages = returns_tensor - step_baselines
    total_return_sum = torch.sum(returns_tensor[:, 0]) + eps

    loss = -torch.sum(batch_log_probs * advantages) / total_return_sum

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


if __name__ == '__main__':
    CORE = 10
    print(f"[CPU Setup] Using {CORE} worker threads for parallel sampling.")

    env_kwargs = {
        'n_segments': N_SEGMENTS,
        'pulse_duration': PULSE_DURATION,
    }

    policy = ContinuousPyTorchPolicy(state_dim=1, action_dim=2, hidden_size=HS).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_ITERATIONS, eta_min=1e-4)

    checkpoint_dir = "./ckpt_1q_mesolve"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_policy.pt")

    start_iteration = 0
    best_mean_fid = -1.0

    existing_ckpts = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_iter_") and f.endswith(".pt")
    ]

    if existing_ckpts:
        latest_ckpt_file = sorted(existing_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        resume_path = os.path.join(checkpoint_dir, latest_ckpt_file)
        print(f"\n[Checkpoint Found] Loading state from '{resume_path}'...")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iteration = checkpoint['iteration']
        best_mean_fid = checkpoint.get('best_mean_fidelity', -1.0)
        print(f"[Resumed] Resuming training from Iteration {start_iteration + 1} / {NUM_ITERATIONS}\n")
    else:
        print(f"\n[No Periodic Checkpoint Found] Starting training from scratch.\n")

    t0 = time.time()
    for it in range(start_iteration, NUM_ITERATIONS):
        std_val = max(0.01, 0.3 * (1.0 - it / NUM_ITERATIONS))

        batch_log_probs, batch_raw_rewards, batch_final_info = play_episodes_parallel(
            env_kwargs, policy, BATCH_SIZE, std_val, max_workers=CORE
        )

        loss_val = update_policy_reinforce(policy, optimizer, batch_log_probs, batch_raw_rewards)
        scheduler.step()

        batch_fids = [np.mean(info['fidelities']) for info in batch_final_info]
        mean_fid = np.mean(batch_fids)
        max_fid = np.max(batch_fids)

        t1 = time.time()
        print(f"Iteration {it+1:4d}/{NUM_ITERATIONS} | "
              f"Mean Fid: {mean_fid:.4f} | "
              f"Max Fid: {max_fid:.4f} | "
              f"Best Mean: {max(best_mean_fid, mean_fid):.4f} | "
              f"Loss: {loss_val:.4f} | "
              f"T: {t1-t0:.2f} s")
        t0 = t1

        current_iter = it + 1

        if mean_fid > best_mean_fid:
            best_mean_fid = mean_fid
            torch.save({
                'iteration': current_iter,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_fidelity': best_mean_fid,
            }, best_checkpoint_path)
            print(f"  └─> [Saved Best Checkpoint] Improved Mean Fidelity to {best_mean_fid:.4f}")

        if current_iter % 1000 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{current_iter}.pt")
            torch.save({
                'iteration': current_iter,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_fidelity': best_mean_fid,
            }, periodic_path)
            print(f"  └─> [Periodic Checkpoint] Saved checkpoint at iteration {current_iter}")

    print("\n[SUCCESS] Training session completed!")
    print(f"[FINAL] Best Overall Mean Fidelity Achieved: {best_mean_fid:.4f}")
