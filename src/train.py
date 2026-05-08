"""
Training entry-point.

Logs:
  * per-episode return,
  * per-step appraisal vector (saved to runs/<run_name>/appraisals.npz),
  * loss / TD stats.

Usage:
    python -m src.train --run extended_8dim
    python -m src.train --run baseline_4dim
"""
from __future__ import annotations
import argparse, os, time, json
import numpy as np
import torch

from .config import baseline_config, extended_config
from .env import GridWorld, N_ACTIONS
from .agent import DQNAgent
from .replay_buffer import Transition


def get_config(run_name: str):
    if run_name == "baseline_4dim":
        return baseline_config()
    if run_name == "extended_8dim":
        return extended_config()
    raise ValueError(f"Unknown run name: {run_name}")


def evaluate(agent: DQNAgent, env: GridWorld, episodes: int) -> float:
    rets = []
    for _ in range(episodes):
        obs = env.reset()
        done, ret = False, 0.0
        eps_save = agent.cfg.dqn.eps_end
        # greedy
        while not done:
            with torch.no_grad():
                q = agent.online.q_mean(
                    torch.as_tensor(obs, device=agent.device).unsqueeze(0))
                a = int(q.argmax(dim=-1).item())
            r = env.step(a)
            obs = r.obs
            ret += r.reward
            done = r.done
        rets.append(ret)
    return float(np.mean(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["baseline_4dim", "extended_8dim"],
                    default="extended_8dim")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = get_config(args.run)
    if args.frames is not None:
        cfg.train.total_frames = args.frames
    if args.seed is not None:
        cfg.train.seed = args.seed
        cfg.env.seed = args.seed

    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    out_dir = os.path.join("runs", cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)

    env = GridWorld(grid_size=cfg.env.grid_size,
                    max_steps=cfg.env.max_steps,
                    n_lava=cfg.env.n_lava,
                    n_keys=cfg.env.n_keys,
                    seed=cfg.env.seed)
    agent = DQNAgent(cfg, env.obs_shape, N_ACTIONS)

    print(f"[{cfg.run_name}] dims = {cfg.appraisal.use_dims}")
    print(f"[{cfg.run_name}] frames = {cfg.train.total_frames}")

    obs = env.reset()
    sid = env.state_id()
    had_key_before = env.has_key
    ep_ret, ep_len, ep_count = 0.0, 0, 0
    returns, eval_log = [], []
    appraisal_log, label_log = [], []   # for analysis

    t_start = time.time()
    last_log = 0
    while agent.frames < cfg.train.total_frames:
        a = agent.act(obs)
        r = env.step(a)
        next_sid = env.state_id()

        # appraisal vector for this transition
        v = agent.step_appraisal(obs, sid, a, r.reward, r.obs,
                                 next_sid, r.done, r.info["t"])
        appraisal_log.append(v)
        # crude "event label" -- useful for emotion classification later.
        # 0 = neutral step, 1 = picked up key, 2 = reached goal, 3 = died on lava
        if r.done and r.reward >= 0.9:
            label = 2  # success: reached goal with key
        elif r.done and r.reward <= -0.9:
            label = 3  # failure: stepped into lava
        elif (not had_key_before) and env.has_key:
            label = 1  # picked up key this step
        else:
            label = 0  # neutral
        label_log.append(label)
        had_key_before = env.has_key

        agent.push_transition(Transition(
            obs=obs, action=a, reward=r.reward, next_obs=r.obs,
            done=r.done, sid=sid, next_sid=next_sid, t=r.info["t"]))

        obs, sid = r.obs, next_sid
        ep_ret += r.reward; ep_len += 1
        agent.frames += 1

        info = agent.learn()

        if r.done:
            returns.append(ep_ret)
            ep_count += 1
            obs = env.reset()
            sid = env.state_id()
            had_key_before = env.has_key
            ep_ret, ep_len = 0.0, 0

        if agent.frames - last_log >= cfg.train.log_every:
            recent = returns[-20:] if returns else [0.0]
            print(f"frame={agent.frames:6d} eps={agent.epsilon():.2f} "
                  f"ep={ep_count} R20={np.mean(recent):+.3f} "
                  f"loss={info.get('loss', float('nan')):.3f}")
            last_log = agent.frames

        if (agent.frames % cfg.train.eval_every) == 0:
            ev = evaluate(agent, GridWorld(grid_size=cfg.env.grid_size,
                                           max_steps=cfg.env.max_steps,
                                           n_lava=cfg.env.n_lava,
                                           n_keys=cfg.env.n_keys,
                                           seed=cfg.env.seed + 1000),
                          cfg.train.eval_episodes)
            eval_log.append((agent.frames, ev))
            print(f"  [eval] frame={agent.frames} mean_return={ev:+.3f}")

    duration = time.time() - t_start
    print(f"done in {duration:.1f}s — episodes={ep_count}")

    # Save artefacts
    np.savez(os.path.join(out_dir, "appraisals.npz"),
             appraisals=np.stack(appraisal_log),
             labels=np.array(label_log),
             dims=np.array(cfg.appraisal.use_dims))
    np.save(os.path.join(out_dir, "returns.npy"), np.array(returns))
    with open(os.path.join(out_dir, "eval.json"), "w") as f:
        json.dump(eval_log, f, indent=2)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "run_name": cfg.run_name,
            "dims": cfg.appraisal.use_dims,
            "frames": cfg.train.total_frames,
            "duration_sec": duration,
        }, f, indent=2)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
