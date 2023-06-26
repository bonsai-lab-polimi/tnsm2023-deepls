import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from tqdm import tqdm

from envs.rwa_sa import RWAEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PPO testing arguments.")
    parser.add_argument("--network", type=str, default="nsfnet")
    parser.add_argument("--episode_length", type=int, default=100)
    parser.add_argument("--n_slots", type=int, default=80)
    parser.add_argument("--n_requests", type=int, default=800)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default="./results/rwa/")

    args = parser.parse_args()

    env_kwargs = {
        "network": args.network,
        "episode_length": args.episode_length,
        "n_slots": args.n_slots,
        "n_requests": args.n_requests,
        "eval": True,
    }

    env = make_vec_env(RWAEnv, n_envs=1, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)
    agent = PPO.load(args.model_dir + "model", device="cpu")
    best_blocking = []
    ksp_blocking = []
    times = []
    obs = env.reset()
    for _ in tqdm(range(args.n_episodes)):
        done = False
        tic = time.time()
        while not done:
            with torch.no_grad():
                action = agent.policy._predict(torch.tensor(obs)).long().numpy().reshape(1, -1)
            # action = np.array([np.argmax(np.squeeze(obs)[:, 0])])
            obs, rewards, dones, infos = env.step(action[0])
            done = dones[0]
        times.append(time.time() - tic)
        best_blocking.append(infos[0]["best_blocking"])
        ksp_blocking.append(infos[0]["ksp_blocking_rate"])

    if not os.path.exists(args.model_dir + "eval"):
        os.mkdir(args.model_dir + "eval")
    np.savetxt(args.model_dir + "eval/" + args.network + "test.txt", best_blocking, delimiter=",")
