"""
sac_discrete.py
---------------
Soft Actor-Critic for DISCRETE actions.
Based on Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings".

Key design decisions:
  • CategoricalActor: softmax over logits → P(a|s).
  • DiscreteDoubleQNetwork: Q1 + Q2 → R^|A| in ONE module, ONE optimizer.
    → single backward() updates both heads; no retain_graph issues.
  • Soft Bellman target uses the EXACT expectation over P(a|s):
        V(s') = Σ_a π(a|s') [Q(s',a) − α log π(a|s')]
    No reparameterisation trick needed (discrete actions).
  • Actor loss: maximise  Σ_a π(a|s) [Q_min(s,a) − α log π(a|s)]

  FIXES applied vs previous version:
  ────────────────────────────────────
  FIX 1 — Reward scaling removed.
    Previous code divided rewards by 10 inside the agent. This made Q-values
    10× smaller than alpha's entropy term, so alpha's effective influence was
    10× too large. The policy stayed near-uniform for 200K+ steps as a result.
    Rewards are now used raw so Q-values and alpha are on the same scale.

  FIX 2 — target_entropy changed from 0.20*log|A| to 0.50*log|A|.
    0.20*log(4)=0.277 ≈ entropy of a near-greedy (97%) policy.
    Alpha would immediately collapse to ~0, turning off all exploration.
    0.50*log(4)=0.693 targets ~50% of maximum entropy: the policy can be
    decisive (e.g. 75–80% on the best action) while maintaining exploration.

  FIX 3 — actor_update_frequency changed from 2 to 1.
    For continuous SAC, freq=2 gives the critic a head start on the actor
    because the actor's Gaussian is complex. For discrete SAC the actor is
    a simple 4-class softmax — updating it every critic step is stable and
    roughly doubles the useful gradient steps on the actor.

  Comparison with DQN is now fair:
    Both agents receive raw rewards, same buffer size, same warmup, same γ.

Assignment requirements satisfied:
  • γ = 0.99
  • 10 K random-action warmup
  • Adam optimiser
  • Clipped double Q-learning (DiscreteDoubleQNetwork)
  • Automated temperature tuning (auto-alpha)
"""

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as pyd

from networks import CategoricalActor, DiscreteDoubleQNetwork
from buffer   import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)


class SACDiscrete:
    """
    Discrete SAC (Christodoulou 2019).

    Parameters
    ----------
    obs_dim   : observation dimension  (8 for LunarLander-v3)
    n_actions : number of actions      (4 for LunarLander-v3)
    """

    def __init__(
        self,
        obs_dim:                        int,
        n_actions:                      int,
        hidden_dim:                     int   = 256,
        hidden_depth:                   int   = 2,
        lr:                             float = 3e-4,
        adam_betas:                     tuple = (0.9, 0.999),
        gamma:                          float = 0.99,
        tau:                            float = 0.005,
        batch_size:                     int   = 256,
        buffer_size:                    int   = int(1e6),
        random_steps:                   int   = 10_000,
        actor_update_frequency:         int   = 1,   # FIX 3: was 2
        critic_target_update_frequency: int   = 2,
        init_alpha:                     float = 0.2,
    ):
        self.n_actions    = n_actions
        self.gamma        = gamma
        self.tau          = tau
        self.batch_size   = batch_size
        self.random_steps = random_steps

        self.total_steps  = 0
        self.update_steps = 0

        self.actor_update_freq         = actor_update_frequency
        self.critic_target_update_freq = critic_target_update_frequency

        # ── Networks ──────────────────────────────────────────────────────────
        self.actor      = CategoricalActor(
            obs_dim, n_actions, hidden_dim, hidden_depth).to(DEVICE)
        self.critic     = DiscreteDoubleQNetwork(
            obs_dim, n_actions, hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt = DiscreteDoubleQNetwork(
            obs_dim, n_actions, hidden_dim, hidden_depth).to(DEVICE)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        for p in self.critic_tgt.parameters():
            p.requires_grad = False

        # ── Optimisers ────────────────────────────────────────────────────────
        self.actor_opt  = torch.optim.Adam(
            self.actor.parameters(),  lr=lr, betas=adam_betas)
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=lr, betas=adam_betas)

        # ── Auto-alpha ────────────────────────────────────────────────────────
        # FIX 2: 0.50 * log|A| (was 0.20 * log|A|)
        #   0.50 * log(4) ≈ 0.693  → policy can be decisive but still explores
        #   0.20 * log(4) ≈ 0.277  → alpha collapses to ~0, turns off exploration
        #   0.98 * log(4) ≈ 1.359  → forces near-uniform, too slow to learn
        self.target_entropy = 0.5 * np.log(n_actions)   # ≈ 0.693 for 4 actions

        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32,
            requires_grad=True, device=DEVICE,
        )
        self.alpha_opt = torch.optim.Adam(
            [self.log_alpha], lr=lr, betas=adam_betas)
        self.alpha = self.log_alpha.exp().item()

        # ── Replay buffer ─────────────────────────────────────────────────────
        self.buffer = ReplayBuffer(
            obs_dim, act_dim=1, max_size=buffer_size, discrete=True)

    # ── public API ────────────────────────────────────────────────────────────

    def set_train(self):
        self.actor.train()
        self.critic.train()

    def set_eval(self):
        self.actor.eval()
        self.critic.eval()

    def select_action(self, obs, evaluate=False):
        """
        Warmup  : uniform random over all actions.
        Training: stochastic categorical sample from π(a|s).
        Eval    : greedy argmax of π(a|s).
        """
        if not evaluate and self.total_steps < self.random_steps:
            return np.random.randint(self.n_actions)

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = self.actor(obs_t)           # (1, A)
        if evaluate:
            return probs.argmax(dim=-1).item()
        return pyd.Categorical(probs).sample().item()

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
        actions         = torch.LongTensor(actions_np).to(DEVICE)    # (B,)
        # FIX 1: NO /10.0 here. Raw rewards keep Q-values correctly scaled
        # relative to alpha so the temperature is properly calibrated.
        rewards         = torch.FloatTensor(rewards_np).to(DEVICE)   # (B, 1)
        not_done_no_max = torch.FloatTensor(not_done_no_max_np).to(DEVICE)

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_probs     = self.actor(next_obs)                # (B, A)
            next_log_probs = torch.log(next_probs + 1e-8)       # (B, A)
            q1_next, q2_next = self.critic_tgt(next_obs)
            q_next = torch.min(q1_next, q2_next)                # (B, A)

            # Exact soft value — no sampling required for discrete actions
            # V(s') = Σ_a π(a|s') [Q(s',a) − α log π(a|s')]
            v_next = (next_probs * (q_next - self.alpha * next_log_probs)
                      ).sum(dim=-1, keepdim=True)               # (B, 1)

            q_target = rewards + self.gamma * not_done_no_max * v_next

        # Single backward() through both Q-heads — no retain_graph needed
        q1_pred, q2_pred = self.critic(obs)                     # (B, A) each
        q1_pred = q1_pred.gather(1, actions.unsqueeze(1))       # (B, 1)
        q2_pred = q2_pred.gather(1, actions.unsqueeze(1))       # (B, 1)

        critic_loss = (F.mse_loss(q1_pred, q_target)
                       + F.mse_loss(q2_pred, q_target))

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.update_steps += 1

        # ── Actor + alpha update (FIX 3: every critic step, not every 2) ──
        if self.update_steps % self.actor_update_freq == 0:

            probs     = self.actor(obs)                         # (B, A)
            log_probs = torch.log(probs + 1e-8)                 # (B, A)

            # Detach critic — gradient must NOT flow through Q for actor loss
            with torch.no_grad():
                q_pi = self.critic.q_min(obs)                   # (B, A)

            # Actor: minimise  Σ_a π(a|s) [α log π(a|s) − Q_min(s,a)]
            # Equivalently: maximise expected soft Q-value
            actor_loss = (probs * (self.alpha * log_probs - q_pi)
                          ).sum(dim=-1).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Alpha: L(log_α) = log_α * (H(π) − H_target)
            #   H(π) > H_target  →  loss > 0  →  log_α decreases  →  α ↓
            #   H(π) < H_target  →  loss < 0  →  log_α increases  →  α ↑
            with torch.no_grad():
                entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)

            alpha_loss = (self.log_alpha
                          * (entropy - self.target_entropy)).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()
            self.alpha = self.log_alpha.exp().item()

        # ── Soft target update ─────────────────────────────────────────────
        if self.update_steps % self.critic_target_update_freq == 0:
            _soft_update(self.critic_tgt, self.critic, self.tau)
