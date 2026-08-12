# ==============================================================================
# 📦 0. Imports & Setup
# ==============================================================================

HS=18

import os
import time
import argparse
import numpy as np
import qutip as qt
import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from joblib import Parallel, delayed

# Enforce CPU operation
device = torch.device("cpu")


# ==============================================================================
# 🚀 1. 1-Qubit Gate Calibration Environment
# ==============================================================================
class SingleQubitGateControlEnv(gym.Env):
    """
    Simulates physical single-qubit quantum driving using QuTiP mesolve.
    Goal: Calibrate pulse parameters [amplitude, phase] to implement an X_pi gate.
    """
    def __init__(self, n_segments=8, pulse_duration=20.0, n_gate_reps=1):
        super().__init__()
        self.n_segments = n_segments
        self.pulse_duration = pulse_duration
        self.dt = pulse_duration / n_segments
        self.n_gate_reps = n_gate_reps
        self.current_step = 0
        
        # Physical parameters
        self.omega_q = 2 * np.pi * 5.0   # Qubit frequency (5 GHz)
        self.omega_p = self.omega_q       # On-resonance drive
        
        # Hamiltonian operators: H = -0.5 * omega_q * sz + drive(t) * sx
        self.H_int = -0.5 * self.omega_q * qt.sigmaz()
        self.H_drive_op = qt.sigmax()
        
        # Action space: [amplitude (GHz), phase (rad)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32), 
            high=np.array([2 * np.pi * 0.1, 2 * np.pi], dtype=np.float32), 
            dtype=np.float32
        )
        
        # State space: 2x2 density matrix (4 real + 4 imag + 1 progress = 9 dims)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(9,), dtype=np.float32
        )
        
        # Target Unitary: Single-qubit Pauli-X gate
        self.U_target = qt.sigmax()
        
        self.initial_psi = None
        self.current_state = None

    def _sample_haar_random_state(self):
        """Uniform Haar-random 1-qubit state: a|0> + b|1>"""
        rand_complex = np.random.normal(size=2) + 1j * np.random.normal(size=2)
        rand_complex /= np.linalg.norm(rand_complex)
        return qt.Qobj(rand_complex, dims=[[2], [1]])

    def _get_obs(self):
        """Flattens 2x2 density matrix (real & imag) + progress into 9-dim vector."""
        matrix_data = self.current_state.full()
        progress = np.array([self.current_step / self.n_segments], dtype=np.float32)
        return np.concatenate([
            matrix_data.real.flatten(), 
            matrix_data.imag.flatten(),
            progress
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.current_step = 0
        self.initial_psi = self._sample_haar_random_state()
        self.current_state = qt.ket2dm(self.initial_psi)
        return self._get_obs(), {}

    def step(self, action):
        amp_k, phi_k = action[0], action[1]

        t_start = self.current_step * self.dt
        t_end = (self.current_step + 1) * self.dt
        tlist_segment = np.linspace(t_start, t_end, 10)
        
        def drive_coeff(t, **kwargs):
            return amp_k * np.sin(phi_k + self.omega_p * t)
            
        H_segment = [self.H_int, [self.H_drive_op, drive_coeff]]
        
        result = qt.mesolve(H_segment, self.current_state, tlist_segment, options={'store_final_state': True})
        self.current_state = result.states[-1]
        
        self.current_step += 1
        done = self.current_step >= self.n_segments
        
        reward = 0.0
        fidelity_single = 0.0
        
        if done:
            psi_target_1 = self.U_target * self.initial_psi
            rho_target_1 = qt.ket2dm(psi_target_1)
            
            # State fidelity F = |<psi_target|rho|psi_target>|
            fidelity_single = float(qt.fidelity(self.current_state, rho_target_1) ** 2)
            fidelity_amplified = fidelity_single ** self.n_gate_reps
            reward = fidelity_amplified
            
        return self._get_obs(), reward, done, False, {
            "fidelity": reward,
            "base_fidelity": fidelity_single
        }


# ==============================================================================
# 🐍 2. Policy Network
# ==============================================================================
class ContinuousPyTorchPolicy(nn.Module):
    def __init__(self, state_dim=9, action_dim=2, hidden_size=HS):
        super(ContinuousPyTorchPolicy, self).__init__()
        
        self.max_amp = 2 * np.pi * 0.1  # Max drive amplitude (0.1 GHz)
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
        action_numpy = action_tensor.detach().numpy()
        return action_numpy, None


# ==============================================================================
# 🐍 3. Parallel Worker Sampling
# ==============================================================================
def _run_single_episode_worker(args):
    policy_state_dict, std_val, env_kwargs, task_id = args

    worker_seed = (int(time.time() * 1000) & 0xFFFF) ^ (os.getpid() << 8) ^ (task_id * 10007)
    np.random.seed(worker_seed % (2**32 - 1))
    torch.manual_seed(worker_seed % (2**32 - 1))

    local_env = SingleQubitGateControlEnv(**env_kwargs)
    local_policy = ContinuousPyTorchPolicy(
        state_dim=local_env.observation_space.shape[0],
        action_dim=local_env.action_space.shape[0],
    ).to(device)
    local_policy.load_state_dict(policy_state_dict)

    state, _ = local_env.reset()
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
                    ep_rewards, 
                    info)


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


# ==============================================================================
# 🎯 4. REINFORCE Policy Gradient Update
# ==============================================================================
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
        trajectory_returns_to_go, 
        dtype=torch.float32, 
        device=batch_log_probs.device
    )

    step_baselines = returns_tensor.mean(dim=0, keepdim=True)
    advantages = returns_tensor - step_baselines
    total_return_sum = torch.sum(returns_tensor[:, 0]) + eps

    loss = - torch.sum(batch_log_probs * advantages) / total_return_sum
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

# ==============================================================================
# 🏋️‍♂️ 5. Main Execution & Checkpointing Loop
# ==============================================================================
if __name__ == '__main__':

    #parser = argparse.ArgumentParser(description="Run 1-Qubit Gate RL Training on CPU.")
    #parser.add_argument("--core", "-c", type=int, default=os.cpu_count())
    #args = parser.parse_args()
    
    CORE = 10 #max(1, args.core - 1) if args.core else 1
    print(f"[CPU Setup] Using {CORE} worker threads for parallel sampling.")
    
    env_kwargs = {
        'n_segments': 8,
        'pulse_duration': 20.0,
        'n_gate_reps': 1
    }

    BATCH_SIZE = 128
    num_iterations = 1000
    learning_rate = 5e-3

    # Instantiate model and optimization components
    policy = ContinuousPyTorchPolicy(state_dim=9, action_dim=2, hidden_size=HS).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations, eta_min=1e-4)

    # Directory setup for checkpoints
    checkpoint_dir = "./ckpt_1q"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_policy.pt")

    start_iteration = 0
    best_mean_base_fid = -1.0

    # --------------------------------------------------------------------------
    # Checkpoint Recovery Check (Finds highest periodic checkpoint or best_policy)
    # --------------------------------------------------------------------------
    existing_ckpts = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("cpit_") and f.endswith(".pt")
    ]
    
    if existing_ckpts:
        # Sort by iteration number and get the latest 100-step checkpoint
        latest_ckpt_file = sorted(existing_ckpts, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
        resume_path = os.path.join(checkpoint_dir, latest_ckpt_file)
        
        print(f"\n[Checkpoint Found] Loading state from '{resume_path}'...")
        checkpoint = torch.load(resume_path, map_location=device)
        
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_iteration = checkpoint['iteration']
        best_mean_base_fid = checkpoint.get('best_mean_fidelity', -1.0)
        
        print(f"[Resumed] Resuming training from Iteration {start_iteration + 1} / {num_iterations}\n")
    else:
        print(f"\n[No Periodic Checkpoint Found] Starting training from scratch.\n")

    # --------------------------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------------------------
    t0 = time.time()
    for it in range(start_iteration, num_iterations):
        std_val = max(0.01, 0.3 * (1.0 - it / num_iterations))

        # 1. Parallel Episode Sampling on CPU
        batch_log_probs, batch_raw_rewards, batch_final_info = play_episodes_parallel(
            env_kwargs, policy, BATCH_SIZE, std_val, max_workers=CORE
        )
        
        # 2. REINFORCE Optimization Step
        loss_val = update_policy_reinforce(policy, optimizer, batch_log_probs, batch_raw_rewards)
        scheduler.step() 
        
        # 3. Fidelity Metrics
        batch_base_fids = [info['base_fidelity'] for info in batch_final_info]
        mean_base_fid = np.mean(batch_base_fids)
        max_base_fid = np.max(batch_base_fids)
        
        t1 = time.time()
        print(f"Iteration {it+1:4d}/{num_iterations} | "
              f"Mean Fid: {mean_base_fid:.4f} | "
              f"Max Fid: {max_base_fid:.4f} | "
              f"Best Mean: {max(best_mean_base_fid, mean_base_fid):.4f} | "
              f"Loss: {loss_val:.4f} | "
              f"T: {t1-t0:.2f} s")
        t0 = t1

        current_iter = it + 1

        # ----------------------------------------------------------------------
        # Checkpointing Logic
        # ----------------------------------------------------------------------
        # A. Save "best" model weights whenever mean fidelity reaches a new peak
        if mean_base_fid > best_mean_base_fid:
            best_mean_base_fid = mean_base_fid
            torch.save({
                'iteration': current_iter,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_fidelity': best_mean_base_fid,
            }, best_checkpoint_path)
            print(f"  └─> [Saved Best Checkpoint] Improved Mean Fidelity to {best_mean_base_fid:.4f}")

        # B. Save periodic checkpoint strictly every 100 steps
        if current_iter % 100 == 0:
            periodic_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{current_iter}.pt")
            torch.save({
                'iteration': current_iter,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mean_fidelity': best_mean_base_fid,
            }, periodic_path)
            print(f"  └─> [Periodic Checkpoint] Saved checkpoint at iteration {current_iter}")

    print("\n[SUCCESS] Training session completed!")
    print(f"[FINAL] Best Overall Mean Fidelity Achieved: {best_mean_base_fid:.4f}")