"""
dqn.py
------
Vanilla DQN baseline for the discrete-action LunarLander (Q2.2.4).

Design:
  • 10 K random-action warmup (matches SAC).
  • Epsilon-greedy exploration with linear decay after warmup.
  • Hard target-network copy every target_update_freq gradient updates.
  • not_done_no_max in Bellman target → bootstrap on TimeLimit truncations.
  • set_train() / set_eval() for correct mode switching.
"""

import numpy as np
import torch
import torch.nn.functional as F

from networks import DQNNetwork
from buffer   import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN:
    """
    Parameters
    ----------
    target_update_freq : int
        Hard-copy the target network every this many *gradient updates*
        (not environment steps).
    eps_decay_steps : int
        Number of environment steps (after warmup) over which epsilon
        decays linearly from eps_start to eps_end.
    """

    def __init__(
        self,
        obs_dim:            int,
        n_actions:          int,
        hidden_dim:         int   = 256,
        hidden_depth:       int   = 2,
        lr:                 float = 1e-3,
        adam_betas:         tuple = (0.9, 0.999),
        gamma:              float = 0.99,
        batch_size:         int   = 256,
        buffer_size:        int   = int(1e6),
        eps_start:          float = 1.0,
        eps_end:            float = 0.05,
        eps_decay_steps:    int   = 100_000,
        target_update_freq: int   = 1_000,
        random_steps:       int   = 10_000,
    ):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.random_steps       = random_steps
        self.target_update_freq = target_update_freq
        self.eps_start          = eps_start
        self.eps_end            = eps_end
        self.eps_decay_steps    = eps_decay_steps

        self.total_steps  = 0
        self.update_steps = 0

        # ── Networks ─────────────────────────────────────────────────────────
        self.q_net     = DQNNetwork(
            obs_dim, n_actions, hidden_dim, hidden_depth).to(DEVICE)
        self.q_net_tgt = DQNNetwork(
            obs_dim, n_actions, hidden_dim, hidden_depth).to(DEVICE)
        self.q_net_tgt.load_state_dict(self.q_net.state_dict())
        for p in self.q_net_tgt.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=lr, betas=adam_betas)

        # act_dim=1 is a placeholder (buffer stores scalar integer actions)
        self.buffer = ReplayBuffer(
            obs_dim, act_dim=1, max_size=buffer_size, discrete=True)

    # ── public API ────────────────────────────────────────────────────────────

    def set_train(self) -> None:
        self.q_net.train()

    def set_eval(self) -> None:
        self.q_net.eval()

    def _epsilon(self) -> float:
        """Linear decay from eps_start → eps_end over eps_decay_steps
        environment steps that occur AFTER the random warmup phase."""
        if self.total_steps < self.random_steps:
            return 1.0
        progress = min(1.0, (self.total_steps - self.random_steps)
                       / self.eps_decay_steps)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def select_action(self, obs: np.ndarray,
                      evaluate: bool = False) -> int:
        if not evaluate and self.total_steps < self.random_steps:
            return np.random.randint(self.n_actions)
        eps = 0.0 if evaluate else self._epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=-1).item()

    def store(self, obs, next_obs, action: int, reward: float,
              terminated: bool, truncated: bool) -> None:
        self.buffer.add(obs, next_obs, action, reward, terminated, truncated)
        self.total_steps += 1

    def update(self) -> None:
        if (len(self.buffer) < self.batch_size
                or self.total_steps < self.random_steps):
            return

        (obs_np, next_obs_np, actions_np,
         rewards_np, _not_done, not_done_no_max_np) = self.buffer.sample(
            self.batch_size)

        obs             = torch.FloatTensor(obs_np).to(DEVICE)
        next_obs        = torch.FloatTensor(next_obs_np).to(DEVICE)
        actions         = torch.LongTensor(actions_np).to(DEVICE)
        rewards         = torch.FloatTensor(rewards_np).to(DEVICE)
        not_done_no_max = torch.FloatTensor(not_done_no_max_np).to(DEVICE)

        with torch.no_grad():
            q_next   = self.q_net_tgt(next_obs).max(dim=-1, keepdim=True)[0]
            q_target = rewards + self.gamma * not_done_no_max * q_next

        q_pred = self.q_net(obs).gather(1, actions.unsqueeze(1))
        loss   = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        # Hard copy every target_update_freq gradient updates
        if self.update_steps % self.target_update_freq == 0:
            self.q_net_tgt.load_state_dict(self.q_net.state_dict())
