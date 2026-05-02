"""
buffer.py
---------
Replay buffer for SAC (continuous & discrete) and DQN.

Stores two done flags (pytorch_sac convention):
  not_done         = 1 - (terminated OR truncated)
                     → masks the Q-target completely when the episode ends
  not_done_no_max  = 1 - terminated
                     → still bootstraps when the episode ends due to time
                       limit only (TimeLimit truncation). This is the flag
                       used in the Bellman update.

References:
  Yarats & Kostrikov (2021) pytorch_sac
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int,
                 max_size: int = int(1e6), discrete: bool = False):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0

        self.obs              = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs         = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.rewards          = np.zeros((max_size, 1),       dtype=np.float32)
        self.not_dones        = np.zeros((max_size, 1),       dtype=np.float32)
        self.not_dones_no_max = np.zeros((max_size, 1),       dtype=np.float32)

        # discrete buffers store scalar int actions; continuous store vectors
        if discrete:
            self.actions = np.zeros((max_size,), dtype=np.int64)
        else:
            self.actions = np.zeros((max_size, act_dim), dtype=np.float32)

    # ---------------------------------------------------------------------- #

    def add(self, obs: np.ndarray, next_obs: np.ndarray,
            action, reward: float,
            terminated: bool, truncated: bool) -> None:
        """
        Parameters
        ----------
        terminated : episode truly ended (crash / goal reached)
        truncated  : episode ended only because of the time limit
        """
        self.obs[self.ptr]              = obs
        self.next_obs[self.ptr]         = next_obs
        self.actions[self.ptr]          = action
        self.rewards[self.ptr]          = float(reward)
        # mask Q-target only on a true terminal; keep bootstrapping on truncation
        self.not_dones[self.ptr]        = float(not (terminated or truncated))
        self.not_dones_no_max[self.ptr] = float(not terminated)

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[idx],
            self.next_obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.not_dones[idx],
            self.not_dones_no_max[idx],
        )

    def __len__(self) -> int:
        return self.size
