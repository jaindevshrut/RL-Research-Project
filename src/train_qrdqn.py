"""
Training entry-point for Method 2 (QR-DQN).

Mirrors src/train.py exactly so the comparison between methods is
controlled: same env, same seed, same total frames, same appraisal
extractor, same logging artefacts.

Usage:
    python -m src.train_qrdqn --run qrdqn_8dim
"""
from __future__ import annotations
import argparse, os, time, json
import numpy as np
import torch

from .config import qrdqn_config
from .env import GridWorld, N_ACTIONS
from .agent_qrdqn import QRDQNAgent
from .replay_buffer import Transition


def evaluate(agent: QRDQNAgent, env: GridWorld, episodes: int) -> float:
    rets = []
    for _ in range(episodes):
        obs = env.reset()
        done, ret = False, 0.0
        while not done:
            with torch.no_grad():
                q = agent.online.q_values(
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
    ap.add_argument("--run", choices=["qrdqn_8dim"], default="qrdqn_8dim")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = qrdqn_config()
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
    agent = QRDQNAgent(cfg, env.obs_shape, N_ACTIONS)

    print(f"[{cfg.run_name}] dims = {cfg.appraisal.use_dims}")
    print(f"[{cfg.run_name}] frames = {cfg.train.total_frames}")
    print(f"[{cfg.run_name}] n_quantiles = {cfg.dqn.n_quantiles}")

    obs = env.reset()
    sid = env.state_id()
    had_key_before = env.has_key
    ep_ret, ep_len, ep_count = 0.0, 0, 0
    returns, eval_log = [], []
    appraisal_log, label_log = [], []

    t_start = time.time()
    last_log = 0
    while agent.frames < cfg.train.total_frames:
        a = agent.act(obs)
        r = env.step(a)
        next_sid = env.state_id()

        v = agent.step_appraisal(obs, sid, a, r.reward, r.obs,
                                 next_sid, r.done, r.info["t"])
        appraisal_log.append(v)

        if r.done and r.reward >= 0.9:
            label = 2
        elif r.done and r.reward <= -0.9:
            label = 3
        elif (not had_key_before) and env.has_key:
            label = 1
        else:
            label = 0
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
            "method": "QR-DQN (Dabney et al. 2018)",
            "n_quantiles": cfg.dqn.n_quantiles,
            "dims": cfg.appraisal.use_dims,
            "frames": cfg.train.total_frames,
            "duration_sec": duration,
        }, f, indent=2)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
