"""
sac_continuous.py
-----------------
Soft Actor-Critic for CONTINUOUS actions.

Follows pytorch_sac (Yarats & Kostrikov, 2021) exactly:
  • DoubleQCritic: single module + single optimizer (one backward()).
  • actor_update_frequency = 2  (actor updated every 2 critic steps).
  • critic_target_update_frequency = 2  (target soft-updated every 2 steps).
  • SquashedNormal + reparameterisation trick.
  • not_done_no_max in Bellman target → bootstrap on TimeLimit truncations.
  • Auto-alpha: target_entropy = -|A|, gradient flows through exp(log_alpha).
  • Fixed-alpha mode: auto_alpha=False, init_alpha is used as a constant.
  • set_train() / set_eval() for correct network-mode switching at eval time.

Assignment requirements satisfied:
  • γ = 0.99
  • tanh (squashed Gaussian) policy
  • 10 K random-action warmup
  • Adam optimiser
  • Clipped double Q-learning
  • Reparameterisation trick
  • Only Q-function learned (no separate V-network)
  • Automated temperature tuning (auto_alpha=True)
"""

import numpy as np
import torch
import torch.nn.functional as F

from networks import GaussianActor, DoubleQCritic
from buffer   import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────

def _soft_update(target: torch.nn.Module,
                 source: torch.nn.Module,
                 tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


# ─────────────────────────────────────────────────────────────────────────────

class SACContinuous:
    """
    Continuous SAC with optional automated temperature tuning.

    Parameters
    ----------
    auto_alpha : bool
        True  → learn alpha via gradient descent on the dual objective.
        False → keep alpha fixed at init_alpha (used in Q2.2.3 fixed-α agent).
    init_alpha : float
        Initial (or fixed) value of the temperature parameter α.
    buffer_size : int
        Replay-buffer capacity.  Set to HOVER_BUFFER_SIZE (100 K) for the
        Q2.2.3 experiment so Phase-1 hover memories fade after the switch.
    """

    def __init__(
        self,
        obs_dim:                        int,
        act_dim:                        int,
        hidden_dim:                     int   = 256,
        hidden_depth:                   int   = 2,
        lr:                             float = 3e-4,
        adam_betas:                     tuple = (0.9, 0.999),
        gamma:                          float = 0.99,
        tau:                            float = 0.005,
        batch_size:                     int   = 256,
        buffer_size:                    int   = int(1e6),
        random_steps:                   int   = 10_000,
        actor_update_frequency:         int   = 2,
        critic_target_update_frequency: int   = 2,
        auto_alpha:                     bool  = True,
        init_alpha:                     float = 0.1,
    ):
        self.act_dim      = act_dim
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.random_steps = random_steps
        self.auto_alpha   = auto_alpha
        self.alpha        = init_alpha

        self.total_steps  = 0   # environment interactions so far
        self.update_steps = 0   # critic gradient updates so far

        self.actor_update_freq         = actor_update_frequency
        self.critic_target_update_freq = critic_target_update_frequency

        # ── Networks ─────────────────────────────────────────────────────────
        self.actor      = GaussianActor(obs_dim, act_dim,
                                        hidden_dim, hidden_depth).to(DEVICE)
        self.critic     = DoubleQCritic(obs_dim, act_dim,
                                        hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt = DoubleQCritic(obs_dim, act_dim,
                                        hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        # target network never receives gradient updates
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        # ── Optimisers ───────────────────────────────────────────────────────
        self.actor_opt  = torch.optim.Adam(
            self.actor.parameters(),  lr=lr, betas=adam_betas)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=adam_betas)

        # ── Temperature (α) ──────────────────────────────────────────────────
        if auto_alpha:
            # target entropy = -|A|  (pytorch_sac heuristic for continuous)
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(
                np.log(init_alpha), dtype=torch.float32,
                requires_grad=True, device=DEVICE,
            )
            self.alpha_opt = torch.optim.Adam(
                [self.log_alpha], lr=lr, betas=adam_betas)

        # ── Replay buffer ────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(obs_dim, act_dim, buffer_size, discrete=False)

    # ── public API ────────────────────────────────────────────────────────────

    def set_train(self) -> None:
        self.actor.train()
        self.critic.train()

    def set_eval(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def select_action(self, obs: np.ndarray,
                      evaluate: bool = False) -> np.ndarray:
        """
        Random uniform during warmup.
        Stochastic sample during training.
        Deterministic tanh(mu) during evaluation.
        """
        if not evaluate and self.total_steps < self.random_steps:
            return np.random.uniform(-1.0, 1.0,
                                     size=(self.act_dim,)).astype(np.float32)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if evaluate:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy().flatten()

    def store(self, obs, next_obs, action, reward,
              terminated: bool, truncated: bool) -> None:
        self.buffer.add(obs, next_obs, action, reward, terminated, truncated)
        self.total_steps += 1

    def update(self) -> None:
        """One gradient step. No-op during warmup or when buffer is too small."""
        if (len(self.buffer) < self.batch_size
                or self.total_steps < self.random_steps):
            return

        (obs_np, next_obs_np, actions_np,
         rewards_np, _not_done, not_done_no_max_np) = self.buffer.sample(
            self.batch_size)

        obs              = torch.FloatTensor(obs_np).to(DEVICE)
        next_obs         = torch.FloatTensor(next_obs_np).to(DEVICE)
        actions          = torch.FloatTensor(actions_np).to(DEVICE)
        rewards          = torch.FloatTensor(rewards_np).to(DEVICE)
        not_done_no_max  = torch.FloatTensor(not_done_no_max_np).to(DEVICE)

        # ── Critic update ────────────────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs)
            q1_next, q2_next = self.critic_tgt(next_obs, next_action)
            # Clipped double-Q: take the minimum to reduce overestimation
            q_next   = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            # Bootstrap on truncation (not_done_no_max = 1 − terminated)
            q_target = rewards + self.gamma * not_done_no_max * q_next

        q1, q2      = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.update_steps += 1

        # ── Actor + alpha update (every actor_update_freq critic steps) ──────
        if self.update_steps % self.actor_update_freq == 0:
            # fresh sample for actor gradient (NOT the critic batch)
            action_new, log_prob = self.actor.sample(obs)
            q1_pi, q2_pi = self.critic(obs, action_new)
            q_pi         = torch.min(q1_pi, q2_pi)

            # actor: maximise  E[Q(s,a) − α log π(a|s)]
            actor_loss = (self.alpha * log_prob - q_pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # alpha: minimise  E[−log π(a|s) − H_target] * α
            if self.auto_alpha:
                # gradient flows through exp(log_alpha)
                alpha_loss = (
                    self.log_alpha.exp()
                    * (-log_prob.detach() - self.target_entropy)
                ).mean()

                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()
                self.alpha = self.log_alpha.exp().item()

        # ── Soft target update ────────────────────────────────────────────────
        if self.update_steps % self.critic_target_update_freq == 0:
            _soft_update(self.critic_tgt, self.critic, self.tau)
