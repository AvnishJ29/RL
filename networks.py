"""
networks.py
-----------
All neural-network modules for SAC (continuous & discrete) and DQN.

Follows pytorch_sac (Yarats & Kostrikov, 2021) exactly:
  - GaussianActor      : trunk → chunk mu/log_std, tanh-rescaled log_std,
                         SquashedNormal for numerically-stable log_prob.
  - DoubleQCritic      : Q1 + Q2 in ONE module, ONE optimizer.
  - CategoricalActor   : softmax policy for discrete SAC.
  - DiscreteDoubleQNet : Q1 + Q2 over action dimension, ONE optimizer.
  - DQNNetwork         : simple Q-network for DQN baseline.
  - weight_init        : orthogonal init (pytorch_sac convention).

hidden_dim = 256 everywhere (assignment requirement).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def mlp(input_dim: int, hidden_dim: int, output_dim: int,
        hidden_depth: int) -> nn.Sequential:
    """
    Build a fully-connected MLP with ReLU activations.
    hidden_depth = number of hidden layers (≥ 1).
    """
    if hidden_depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers: list = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
    for _ in range(hidden_depth - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def weight_init(m: nn.Module) -> None:
    """Orthogonal weight init for Linear layers (pytorch_sac convention)."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# SquashedNormal  (pytorch_sac DiagGaussianActor)
# ─────────────────────────────────────────────────────────────────────────────

class TanhTransform(pyd.transforms.Transform):
    """
    Numerically stable tanh bijector.
    log|det J| = 2*(log2 − x − softplus(−2x))
    Avoids the instability of log(1 − tanh²(x)) when |x| is large.
    """
    domain    = pyd.constraints.real
    codomain  = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign      = +1

    def __init__(self, cache_size: int = 1):
        super().__init__(cache_size=cache_size)

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return x.tanh()

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        # numerically-stable atanh
        return 0.5 * (y.log1p() - (-y).log1p())

    def log_abs_det_jacobian(self, x: torch.Tensor,
                             y: torch.Tensor) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.TransformedDistribution):
    """Diagonal Gaussian squashed through tanh. Supports rsample() & log_prob()."""

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        self.loc   = loc
        self.scale = scale
        base = pyd.Normal(loc, scale)
        super().__init__(base, [TanhTransform()])

    @property
    def mean(self) -> torch.Tensor:
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


# ─────────────────────────────────────────────────────────────────────────────
# Continuous-action SAC networks
# ─────────────────────────────────────────────────────────────────────────────

LOG_STD_MIN = -5
LOG_STD_MAX =  2


class GaussianActor(nn.Module):
    """
    Squashed-Gaussian policy (pytorch_sac DiagGaussianActor).
    Single trunk → split into mu + log_std → SquashedNormal.
    log_std is rescaled via tanh so gradients are smooth everywhere.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.trunk = mlp(obs_dim, hidden_dim, 2 * act_dim, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> SquashedNormal:
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        # smooth tanh-clamping instead of hard clamp (no gradient discontinuity)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1.0)
        return SquashedNormal(mu, log_std.exp())

    def sample(self, obs: torch.Tensor):
        """Reparameterised sample + log-prob. Returns (action, log_prob)."""
        dist     = self(obs)
        action   = dist.rsample()                               # reparameterisation trick
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) # sum over action dims
        return action, log_prob

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluation: return tanh(mu) — the mode of the distribution."""
        return self(obs).mean


class DoubleQCritic(nn.Module):
    """
    Q1 and Q2 in a single module with a single optimizer (pytorch_sac style).
    One backward() updates both heads simultaneously → no retain_graph issues.
    critic_loss = MSE(Q1, target) + MSE(Q2, target).
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.Q1 = mlp(obs_dim + act_dim, hidden_dim, 1, hidden_depth)
        self.Q2 = mlp(obs_dim + act_dim, hidden_dim, 1, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor,
                action: torch.Tensor):
        sa = torch.cat([obs, action], dim=-1)
        return self.Q1(sa), self.Q2(sa)


# ─────────────────────────────────────────────────────────────────────────────
# Discrete-action SAC networks  (Christodoulou 2019)
# ─────────────────────────────────────────────────────────────────────────────

class CategoricalActor(nn.Module):
    """Softmax policy over discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, n_actions, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns action probabilities P(a|s)  shape: (B, A)."""
        return F.softmax(self.net(obs), dim=-1)


class DiscreteDoubleQNetwork(nn.Module):
    """
    Two Q-heads Q(s,·) → R^|A| in ONE module with ONE optimizer.
    This is the correct way to implement clipped double-Q for discrete SAC:
      • Q1 and Q2 share one backward() call  → no retain_graph errors.
      • q_min(obs) returns element-wise min for the soft value computation.
    """

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.Q1 = mlp(obs_dim, hidden_dim, n_actions, hidden_depth)
        self.Q2 = mlp(obs_dim, hidden_dim, n_actions, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor):
        """Returns (q1, q2) each of shape (B, A)."""
        return self.Q1(obs), self.Q2(obs)

    def q_min(self, obs: torch.Tensor) -> torch.Tensor:
        """Element-wise minimum of Q1 and Q2  shape: (B, A)."""
        q1, q2 = self(obs)
        return torch.min(q1, q2)


# ─────────────────────────────────────────────────────────────────────────────
# DQN baseline network
# ─────────────────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """Simple Q-network Q(s) → R^|A| for the DQN baseline (Q2.2.4)."""

    def __init__(self, obs_dim: int, n_actions: int,
                 hidden_dim: int = 256, hidden_depth: int = 2):
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, n_actions, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)
