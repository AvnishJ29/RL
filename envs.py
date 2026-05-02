"""
envs.py
-------
Environment factories + HoverLunarLander wrapper for Q2.2.3.

LunarLander-v3 observation layout (8-dim):
  obs[0] = x  (horizontal position,  0 = centre)
  obs[1] = y  (altitude,             0 = ground)
  obs[2] = vx (horizontal velocity)
  obs[3] = vy (vertical velocity)
  obs[4] = angle
  obs[5] = angular velocity
  obs[6] = left leg contact  (bool)
  obs[7] = right leg contact (bool)

HoverLunarLander (Q2.2.3):
  Injects a ONE-TIME bonus/penalty per episode when the lander enters:
      |x| < 0.1   AND   0.4 < y < 0.6
  The assignment states 0.4 < |y| < 0.6. Since y ≥ 0 during flight,
  |y| == y so the conditions are identical.

  Phase 1: hover_reward = +200  (train agent to hover)
  Phase 2: hover_reward = -100  (reward sign flips mid-training)

  set_hover_reward(value) switches phases WITHOUT recreating the env,
  so the replay buffer and network weights are preserved.
"""

import gymnasium as gym


class HoverLunarLander(gym.Wrapper):
    """LunarLander-v3 (continuous) + one-time hover-box reward."""

    def __init__(self, env: gym.Env, hover_reward: float = 200.0):
        super().__init__(env)
        self.hover_reward  = hover_reward
        self._hover_given  = False

    # called mid-training to flip +200 → -100
    def set_hover_reward(self, value: float) -> None:
        self.hover_reward = value

    def reset(self, **kwargs):
        self._hover_given = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y = float(obs[0]), float(obs[1])
        # give hover reward at most ONCE per episode
        if not self._hover_given and abs(x) < 0.1 and 0.4 < y < 0.6:
            reward += self.hover_reward
            self._hover_given = True
        return obs, reward, terminated, truncated, info


# ─────────────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────────────

def make_continuous_env(seed: int = 0) -> gym.Env:
    """Standard continuous LunarLander-v3 (Q2.2.1, Q2.2.2)."""
    env = gym.make("LunarLander-v3", continuous=True)
    env.reset(seed=seed)
    return env


def make_continuous_hover_env(seed: int = 0,
                               hover_reward: float = 200.0) -> HoverLunarLander:
    """Hover-variant continuous LunarLander-v3 (Q2.2.3)."""
    base = gym.make("LunarLander-v3", continuous=True)
    env  = HoverLunarLander(base, hover_reward=hover_reward)
    env.reset(seed=seed)
    return env


def make_discrete_env(seed: int = 0) -> gym.Env:
    """Discrete LunarLander-v3 (Q2.2.4)."""
    env = gym.make("LunarLander-v3", continuous=False)
    env.reset(seed=seed)
    return env
