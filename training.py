"""
training.py
-----------
Training and evaluation loops for all agents.

Key guarantees:
  1.  evaluate() switches to agent.set_eval() before rollout, restores
      set_train() after → correct behaviour even with BN/Dropout.
  2.  Logs are returned as JSON-serialisable dicts AND saved to disk
      after EVERY seed so progress is never lost.
  3.  terminated / truncated are stored separately so the buffer can
      compute not_done_no_max (bootstrap on TimeLimit truncations).
  4.  HOVER_BUFFER_SIZE = 100_000 for Q2.2.3 so Phase-1 hover memories
      fade out naturally in Phase 2 (a full-size buffer would block
      adaptation because old +200 transitions would dominate forever).

Log format (JSON)
-----------------
{
    "timesteps":    [10000, 20000, ...],   # eval checkpoints
    "mean_returns": [m1, m2, ...],         # mean over eval_episodes
    "std_returns":  [s1, s2, ...],         # std  over eval_episodes
    "seed":         <int>
}
"""

import json
import os
import numpy as np

# Finite ring-buffer size for Q2.2.3.
# Phase-1 hover memories (reward +200) fill 100 K slots.
# After the reward switch they are overwritten within ~100 K Phase-2 steps,
# letting the agent adapt to the -100 signal.
HOVER_BUFFER_SIZE = 100_000


# ─────────────────────────────────────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_log(log: dict, fpath: str) -> None:
    """Atomically write a JSON log file."""
    os.makedirs(os.path.dirname(fpath) if os.path.dirname(fpath) else ".",
                exist_ok=True)
    with open(fpath, "w") as f:
        json.dump(log, f, indent=2)


def evaluate(agent, env, n_episodes: int = 20) -> list:
    """
    Run n_episodes with the DETERMINISTIC policy.
    Returns a list of undiscounted episode returns.
    """
    agent.set_eval()
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        total  = 0.0
        while not done:
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done   = terminated or truncated
        returns.append(total)
    agent.set_train()
    return returns


# ─────────────────────────────────────────────────────────────────────────────
# Generic single-agent training loop  (Q2.2.1 and Q2.2.4)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    agent,
    env,
    eval_env,
    total_steps:   int  = 300_000,
    eval_freq:     int  = 10_000,
    eval_episodes: int  = 20,
    seed:          int  = 0,
    save_path:     str  = "",     # full path to JSON file; "" = no intermediate save
    verbose:       bool = True,
) -> dict:
    """
    Train agent for total_steps environment steps.

    Saves the JSON log to save_path (if provided) after each eval checkpoint
    so progress is never lost if the run is interrupted.

    Returns
    -------
    dict with keys "timesteps", "mean_returns", "std_returns", "seed".
    """
    log = {"timesteps": [], "mean_returns": [], "std_returns": [], "seed": seed}

    obs, _     = env.reset()
    ep_reward  = 0.0
    ep_count   = 0

    for t in range(1, total_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(obs, next_obs, action, reward, terminated, truncated)
        agent.update()

        obs        = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            if verbose and ep_count % 50 == 0:
                print(f"  step={t:>8,}  ep={ep_count:>4}  "
                      f"ep_return={ep_reward:>8.1f}")
            obs, _    = env.reset()
            ep_reward = 0.0

        if t % eval_freq == 0:
            rets   = evaluate(agent, eval_env, n_episodes=eval_episodes)
            mean_r = float(np.mean(rets))
            std_r  = float(np.std(rets))
            log["timesteps"].append(t)
            log["mean_returns"].append(mean_r)
            log["std_returns"].append(std_r)

            if verbose:
                print(f"[Eval t={t:>8,}]  "
                      f"mean={mean_r:>8.1f}  std={std_r:>6.1f}")

            # ── Save after every checkpoint so progress is preserved ──────
            if save_path:
                _save_log(log, save_path)

    # Final save
    if save_path:
        _save_log(log, save_path)

    return log


# ─────────────────────────────────────────────────────────────────────────────
# Phase helper (used by Q2.2.3 hover experiment)
# ─────────────────────────────────────────────────────────────────────────────

def _run_phase(
    agent,
    env,
    eval_env,
    n_steps:       int,
    t_offset:      int,
    eval_freq:     int,
    eval_episodes: int,
    save_path:     str  = "",
    existing_log:  dict = None,
    verbose:       bool = True,
) -> dict:
    """
    Run ONE phase of training (phase 1 or phase 2).
    t_offset is added to every logged timestep so the combined log has
    globally-correct x-axis values.

    Saves incrementally to save_path after each eval checkpoint.
    existing_log: if provided, new checkpoints are appended to it.
    """
    if existing_log is None:
        log = {"timesteps": [], "mean_returns": [], "std_returns": []}
    else:
        log = existing_log   # append mode (phase 2 appends to phase-1 log)

    obs, _ = env.reset()

    for t in range(1, n_steps + 1):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store(obs, next_obs, action, reward, terminated, truncated)
        agent.update()

        obs = next_obs
        if done:
            obs, _ = env.reset()

        if t % eval_freq == 0:
            rets   = evaluate(agent, eval_env, n_episodes=eval_episodes)
            mean_r = float(np.mean(rets))
            std_r  = float(np.std(rets))
            global_t = t_offset + t
            log["timesteps"].append(global_t)
            log["mean_returns"].append(mean_r)
            log["std_returns"].append(std_r)

            if verbose:
                print(f"  [Eval t={global_t:>8,}]  "
                      f"mean={mean_r:>8.1f}  std={std_r:>6.1f}")

            # Save after every checkpoint
            if save_path:
                _save_log(log, save_path)

    if save_path:
        _save_log(log, save_path)

    return log


# ─────────────────────────────────────────────────────────────────────────────
# Q2.2.3 two-phase hover training
# ─────────────────────────────────────────────────────────────────────────────

def train_hover(
    agent_fixed,
    agent_auto,
    env_fixed,
    env_auto,
    eval_env_fixed,
    eval_env_auto,
    phase1_steps:  int = 200_000,
    phase2_steps:  int = 200_000,
    eval_freq:     int = 10_000,
    eval_episodes: int = 20,
    seed:          int = 0,
    log_dir:       str = "logs",
    verbose:       bool = True,
):
    """
    Two-phase training for Q2.2.3.

    Phase 1: hover_reward = +200  (runs phase1_steps steps)
    Reward switch: all four envs updated to hover_reward = -100
    Phase 2: hover_reward = -100  (runs phase2_steps steps)

    Logs are saved incrementally after every eval checkpoint to:
      logs/returns_q3_fixed_seed{seed}.json
      logs/returns_q3_auto_seed{seed}.json

    Returns
    -------
    (log_fixed, log_auto)  — both JSON-serialisable dicts.
    """
    path_fixed = os.path.join(log_dir, f"returns_q3_fixed_seed{seed}.json")
    path_auto  = os.path.join(log_dir, f"returns_q3_auto_seed{seed}.json")

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print(f"  [seed={seed}] Phase 1: hover_reward = +200")

    log_fixed = {"timesteps": [], "mean_returns": [], "std_returns": [], "seed": seed}
    log_auto  = {"timesteps": [], "mean_returns": [], "std_returns": [], "seed": seed}

    log_fixed = _run_phase(
        agent_fixed, env_fixed, eval_env_fixed,
        n_steps=phase1_steps, t_offset=0,
        eval_freq=eval_freq, eval_episodes=eval_episodes,
        save_path=path_fixed, existing_log=log_fixed, verbose=verbose,
    )
    log_auto = _run_phase(
        agent_auto, env_auto, eval_env_auto,
        n_steps=phase1_steps, t_offset=0,
        eval_freq=eval_freq, eval_episodes=eval_episodes,
        save_path=path_auto, existing_log=log_auto, verbose=verbose,
    )

    # ── Reward switch ─────────────────────────────────────────────────────────
    print(f"  [seed={seed}] Switching hover_reward: +200 → -100")
    for e in [env_fixed, env_auto, eval_env_fixed, eval_env_auto]:
        e.set_hover_reward(-100)

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    print(f"  [seed={seed}] Phase 2: hover_reward = -100")

    log_fixed = _run_phase(
        agent_fixed, env_fixed, eval_env_fixed,
        n_steps=phase2_steps, t_offset=phase1_steps,
        eval_freq=eval_freq, eval_episodes=eval_episodes,
        save_path=path_fixed, existing_log=log_fixed, verbose=verbose,
    )
    log_auto = _run_phase(
        agent_auto, env_auto, eval_env_auto,
        n_steps=phase2_steps, t_offset=phase1_steps,
        eval_freq=eval_freq, eval_episodes=eval_episodes,
        save_path=path_auto, existing_log=log_auto, verbose=verbose,
    )

    return log_fixed, log_auto
