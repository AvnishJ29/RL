"""
main.py
-------
Entry point for DA6400 PA3 – Section 2.2 (Lunar Lander).

Usage
-----
  python main.py --exp q1              # Q2.2.1  Continuous SAC
  python main.py --exp q3              # Q2.2.3  Hover-reward variant
  python main.py --exp q4              # Q2.2.4  Discrete SAC vs DQN
  python main.py --exp all             # All three, sequentially

Logging
-------
  One JSON file is written per seed, per agent, saved to logs/:

    Q2.2.1:  logs/returns_q1_seed{N}.json
    Q2.2.3:  logs/returns_q3_fixed_seed{N}.json
             logs/returns_q3_auto_seed{N}.json
    Q2.2.4:  logs/returns_q4_sac_seed{N}.json
             logs/returns_q4_dqn_seed{N}.json

  Files are written INCREMENTALLY after every eval checkpoint (every 10 K
  steps) AND again at the end of each seed.  If a run is interrupted, all
  completed checkpoints are already saved.

Hyperparameters
---------------
  γ               = 0.99
  hidden_dim      = 256
  batch_size      = 256
  lr              = 3e-4  (actor / critic / alpha)
  buffer_size     = 1 M   (Q3 uses 100 K)
  random_steps    = 10 K
  total_steps     = 300 K  (Q1 / Q4)
  phase_steps     = 200 K  per phase (Q3)
  eval_freq       = 10 K
  eval_episodes   = 20  (offline, deterministic)
  seeds           = 15
"""

import argparse
import json
import os

import numpy as np
import torch

from envs           import make_continuous_env, make_continuous_hover_env, make_discrete_env
from sac_continuous import SACContinuous
from sac_discrete   import SACDiscrete
from dqn            import DQN
from training       import train, train_hover, HOVER_BUFFER_SIZE
from plot           import load_seed_logs, aggregate_logs, plot_curves, plot_hover_phases

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

SEEDS         = [76,84,87,90,97,86, 65, 64]
N_SEEDS       = 15
EVAL_FREQ     = 10_000
EVAL_EPISODES = 20
LOG_DIR       = "logs"
TOTAL_STEPS   = 600_000   # Q1 and Q4
PHASE_STEPS   = 300_000   # per phase in Q3

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs("plots",   exist_ok=True)
os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Seeding helper
# ─────────────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Q2.2.1  Continuous SAC
# ─────────────────────────────────────────────────────────────────────────────

def run_q1(seeds=SEEDS, total_steps=TOTAL_STEPS):
    print(f"\n{'='*60}")
    print(f"Q2.2.1  Continuous SAC on LunarLander-v3 (continuous)")
    print(f"  seeds={len(seeds)}  total_steps={total_steps:,}")
    print(f"{'='*60}\n")

    all_logs = []

    for idx, seed in enumerate(seeds):
        print(f"── Seed {seed}  ({idx+1}/{len(seeds)}) ──")
        _seed_everything(seed)

        env      = make_continuous_env(seed=seed)
        eval_env = make_continuous_env(seed=seed + 10_000)
        obs_dim  = env.observation_space.shape[0]   # 8
        act_dim  = env.action_space.shape[0]         # 2

        agent = SACContinuous(obs_dim, act_dim, hidden_dim=256, auto_alpha=True)

        save_path = os.path.join(LOG_DIR, f"returns_q1_seed{seed}.json")
        log = train(
            agent, env, eval_env,
            total_steps=total_steps,
            eval_freq=EVAL_FREQ,
            eval_episodes=EVAL_EPISODES,
            seed=seed,
            save_path=save_path,
        )
        env.close()
        eval_env.close()

        all_logs.append(log)
        print(f"  → saved {save_path}\n")

    # ── Aggregate and plot ────────────────────────────────────────────────────
    ts, mean, ci = aggregate_logs(all_logs)
    plot_curves(
        {"SAC (auto-α)": (ts, mean, ci)},
        title="Q2.2.1 – Continuous SAC on LunarLander-v3",
        fname="q1_continuous_sac",
    )
    print(f"Q1 complete. Final mean return = {mean[-1]:.1f} ± {ci[-1]:.1f}")
    return all_logs


# ─────────────────────────────────────────────────────────────────────────────
# Q2.2.3  Hover-reward variant
# ─────────────────────────────────────────────────────────────────────────────

def run_q3(seeds=SEEDS,
           phase1_steps=PHASE_STEPS,
           phase2_steps=PHASE_STEPS):
    total = phase1_steps + phase2_steps
    print(f"\n{'='*60}")
    print(f"Q2.2.3  Hover LunarLander  (fixed α=0.01 vs auto α)")
    print(f"  seeds={len(seeds)}  Phase1={phase1_steps:,}  Phase2={phase2_steps:,}")
    print(f"{'='*60}\n")

    logs_fixed_all = []
    logs_auto_all  = []

    for idx, seed in enumerate(seeds):
        print(f"── Seed {seed}  ({idx+1}/{len(seeds)}) ──")
        _seed_everything(seed)

        env_fixed      = make_continuous_hover_env(seed=seed,           hover_reward=200)
        env_auto       = make_continuous_hover_env(seed=seed,           hover_reward=200)
        eval_env_fixed = make_continuous_hover_env(seed=seed + 10_000,  hover_reward=200)
        eval_env_auto  = make_continuous_hover_env(seed=seed + 10_000,  hover_reward=200)

        obs_dim = env_fixed.observation_space.shape[0]
        act_dim = env_fixed.action_space.shape[0]

        # Finite buffer so Phase-1 hover memories fade after the switch
        agent_fixed = SACContinuous(
            obs_dim, act_dim, hidden_dim=256,
            auto_alpha=False, init_alpha=0.01,
            buffer_size=HOVER_BUFFER_SIZE,
        )
        agent_auto = SACContinuous(
            obs_dim, act_dim, hidden_dim=256,
            auto_alpha=True,
            buffer_size=HOVER_BUFFER_SIZE,
        )

        log_fixed, log_auto = train_hover(
            agent_fixed, agent_auto,
            env_fixed,   env_auto,
            eval_env_fixed, eval_env_auto,
            phase1_steps=phase1_steps,
            phase2_steps=phase2_steps,
            eval_freq=EVAL_FREQ,
            eval_episodes=EVAL_EPISODES,
            seed=seed,
            log_dir=LOG_DIR,
        )

        for e in [env_fixed, env_auto, eval_env_fixed, eval_env_auto]:
            e.close()

        logs_fixed_all.append(log_fixed)
        logs_auto_all.append(log_auto)
        print(f"  → saved Q3 logs for seed={seed}\n")

    # ── Aggregate and plot ────────────────────────────────────────────────────
    agg_fixed = aggregate_logs(logs_fixed_all)
    agg_auto  = aggregate_logs(logs_auto_all)
    plot_hover_phases(agg_fixed, agg_auto,
                      phase1_steps=phase1_steps,
                      fname="q3_hover")
    print("Q3 complete.")
    return logs_fixed_all, logs_auto_all


# ─────────────────────────────────────────────────────────────────────────────
# Q2.2.4  Discrete SAC vs DQN
# ─────────────────────────────────────────────────────────────────────────────

def run_q4(seeds=SEEDS, total_steps=TOTAL_STEPS):
    print(f"\n{'='*60}")
    print(f"Q2.2.4  Discrete SAC vs DQN on LunarLander-v3 (discrete)")
    print(f"  seeds={len(seeds)}  total_steps={total_steps:,}")
    print(f"{'='*60}\n")

    logs_sac_all = []
    logs_dqn_all = []

    for idx, seed in enumerate(seeds):
        print(f"── Seed {seed}  ({idx+1}/{len(seeds)}) ──")
        _seed_everything(seed)

        # Each agent gets its own independent env
        env_sac      = make_discrete_env(seed=seed)
        env_dqn      = make_discrete_env(seed=seed)
        eval_env_sac = make_discrete_env(seed=seed + 10_000)
        eval_env_dqn = make_discrete_env(seed=seed + 10_000)

        obs_dim   = env_sac.observation_space.shape[0]  # 8
        n_actions = env_sac.action_space.n               # 4

        agent_sac = SACDiscrete(obs_dim, n_actions, hidden_dim=256)
        agent_dqn = DQN(obs_dim, n_actions, hidden_dim=256)

        path_sac = os.path.join(LOG_DIR, f"returns_q4_sac_seed{seed}.json")
        path_dqn = os.path.join(LOG_DIR, f"returns_q4_dqn_seed{seed}.json")

        print(f"  Training Discrete SAC …")
        log_sac = train(
            agent_sac, env_sac, eval_env_sac,
            total_steps=total_steps,
            eval_freq=EVAL_FREQ,
            eval_episodes=EVAL_EPISODES,
            seed=seed,
            save_path=path_sac,
        )

        print(f"  Training DQN …")
        log_dqn = train(
            agent_dqn, env_dqn, eval_env_dqn,
            total_steps=total_steps,
            eval_freq=EVAL_FREQ,
            eval_episodes=EVAL_EPISODES,
            seed=seed,
            save_path=path_dqn,
        )

        for e in [env_sac, env_dqn, eval_env_sac, eval_env_dqn]:
            e.close()

        logs_sac_all.append(log_sac)
        logs_dqn_all.append(log_dqn)
        print(f"  → saved {path_sac}\n  → saved {path_dqn}\n")

    # ── Aggregate and plot ────────────────────────────────────────────────────
    ts_s, mean_s, ci_s = aggregate_logs(logs_sac_all)
    ts_d, mean_d, ci_d = aggregate_logs(logs_dqn_all)
    plot_curves(
        {
            "Discrete SAC": (ts_s, mean_s, ci_s),
            "DQN":          (ts_d, mean_d, ci_d),
        },
        title="Q2.2.4 – Discrete SAC vs DQN on LunarLander-v3",
        fname="q4_discrete_sac_vs_dqn",
    )
    print(f"Q4 complete.  SAC={mean_s[-1]:.1f}  DQN={mean_d[-1]:.1f}")
    return logs_sac_all, logs_dqn_all


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DA6400 PA3 – Lunar Lander experiments")
    parser.add_argument(
        "--exp", type=str, default="all",
        choices=["q1", "q3", "q4", "all"],
        help="Which sub-question to run")
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS,
        help="Number of seeds (1-15; default=15)")
    parser.add_argument(
        "--steps", type=int, default=TOTAL_STEPS,
        help="Training steps for Q1/Q4 (Q3 uses steps//2 per phase)")
    args = parser.parse_args()

    seeds_to_use = SEEDS[:max(1, min(args.seeds, N_SEEDS))]

    if args.exp in ("q1", "all"):
        run_q1(seeds=seeds_to_use, total_steps=args.steps)

    if args.exp in ("q3", "all"):
        half = args.steps // 2
        run_q3(seeds=seeds_to_use,
               phase1_steps=half, phase2_steps=half)

    if args.exp in ("q4", "all"):
        run_q4(seeds=seeds_to_use, total_steps=args.steps)
