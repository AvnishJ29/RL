"""
dqn.py
------
Double DQN (DDQN) baseline for the discrete-action LunarLander (Q2.2.4).

This implements DOUBLE DQN, not vanilla DQN:
  Vanilla DQN  : q_next = q_net_tgt(s').max()
                 → target net selects AND evaluates the next action.
                 → tends to overestimate Q-values.
  Double DQN   : best_a  = q_net(s').argmax()          (online net selects)
                 q_next  = q_net_tgt(s')[best_a]        (target net evaluates)
                 → decoupled selection/evaluation reduces overestimation bias.

Why DDQN instead of vanilla?
  LunarLander has large sparse rewards (+100 landing, +200 hover). Vanilla DQN
  overestimates these and learns a biased policy. DDQN is strictly better and
  is the standard choice in practice. The report should label this as DDQN.

Design:
  • 10 K random-action warmup (matches SAC-Discrete).
  • Epsilon-greedy exploration with linear decay after warmup.
  • Hard target-network copy every target_update_freq gradient updates.
  • not_done_no_max in Bellman target → bootstrap on TimeLimit truncations.
  • Huber loss (smooth_l1): more robust to large TD errors from sparse rewards
    than MSE. Equivalent to MSE for small errors, L1 for large ones.
  • Raw rewards — NO scaling — so the comparison with SAC-Discrete is fair.
  • set_train() / set_eval() for correct network mode switching.

Hyperparameters chosen to be comparable with SAC-Discrete:
  lr          : 3e-4  (same as SAC-Discrete; previously was 1e-3 which was
                       an unfair advantage for DQN)
  buffer_size : 1 M   (same)
  batch_size  : 256   (same)
  random_steps: 10 K  (same)
  gamma       : 0.99  (same)
  hidden_dim  : 256   (same)
"""

import numpy as np
import torch
import torch.nn.functional as F

from networks import DQNNetwork
from buffer   import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN:
    """
    Double DQN for discrete-action LunarLander-v3.

    Parameters
    ----------
    target_update_freq : int
        Hard-copy the target network every this many *gradient updates*.
    eps_decay_steps : int
        Environment steps (after warmup) over which ε decays linearly
        from eps_start → eps_end.
    """

    def __init__(
        self,
        obs_dim:            int,
        n_actions:          int,
        hidden_dim:         int   = 256,
        hidden_depth:       int   = 2,
        lr:                 float = 3e-4,       # FIX: was 1e-3, now matches SAC
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

        # ── Networks ──────────────────────────────────────────────────────────
        self.q_net     = DQNNetwork(
            obs_dim, n_actions, hidden_dim, hidden_depth).to(DEVICE)
        self.q_net_tgt = DQNNetwork(
            obs_dim, n_actions, hidden_dim, hidden_depth).to(DEVICE)
        self.q_net_tgt.load_state_dict(self.q_net.state_dict())
        for p in self.q_net_tgt.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=lr, betas=adam_betas)

        # act_dim=1 is a placeholder — buffer stores scalar int actions
        self.buffer = ReplayBuffer(
            obs_dim, act_dim=1, max_size=buffer_size, discrete=True)

    # ── public API ────────────────────────────────────────────────────────────

    def set_train(self):
        self.q_net.train()

    def set_eval(self):
        self.q_net.eval()

    def _epsilon(self):
        """
        Linear decay from eps_start → eps_end over eps_decay_steps steps
        that occur AFTER the random warmup phase.
        """
        if self.total_steps < self.random_steps:
            return 1.0
        progress = min(1.0, (self.total_steps - self.random_steps)
                       / self.eps_decay_steps)
        return self.eps_start + progress * (self.eps_end - self.eps_start)

    def select_action(self, obs, evaluate=False):
        # Pure random during warmup
        if not evaluate and self.total_steps < self.random_steps:
            return np.random.randint(self.n_actions)
        # ε = 0 at evaluation; linear-decayed ε during training
        eps = 0.0 if evaluate else self._epsilon()
        if np.random.rand() < eps:
            return np.random.randint(self.n_actions)
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.q_net(obs_t).argmax(dim=-1).item()

    def store(self, obs, next_obs, action, reward, terminated, truncated):
        self.buffer.add(obs, next_obs, action, reward, terminated, truncated)
        self.total_steps += 1

    def update(self):
        if (len(self.buffer) < self.batch_size
                or self.total_steps < self.random_steps):
            return

        (obs_np, next_obs_np, actions_np,
         rewards_np, _not_done, not_done_no_max_np) = self.buffer.sample(
            self.batch_size)

        obs             = torch.FloatTensor(obs_np).to(DEVICE)
        next_obs        = torch.FloatTensor(next_obs_np).to(DEVICE)
        actions         = torch.LongTensor(actions_np).to(DEVICE)
        # Raw rewards — no scaling — matches SAC-Discrete for a fair comparison
        rewards         = torch.FloatTensor(rewards_np).to(DEVICE)
        not_done_no_max = torch.FloatTensor(not_done_no_max_np).to(DEVICE)

        with torch.no_grad():
            # DOUBLE DQN:
            #   Step 1 — online net selects the greedy next action
            best_next_actions = self.q_net(next_obs).argmax(
                dim=-1, keepdim=True)                               # (B, 1)
            #   Step 2 — target net evaluates that action's Q-value
            q_next = self.q_net_tgt(next_obs).gather(
                1, best_next_actions)                               # (B, 1)
            q_target = rewards + self.gamma * not_done_no_max * q_next

        q_pred = self.q_net(obs).gather(1, actions.unsqueeze(1))   # (B, 1)

        # Huber loss: L1 for large errors, MSE for small — robust to sparse
        # LunarLander rewards (+100 landing, -100 crash, +200 hover)
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_steps += 1

        # Hard target-network copy every target_update_freq gradient steps
        if self.update_steps % self.target_update_freq == 0:
            self.q_net_tgt.load_state_dict(self.q_net.state_dict())

