import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.ppo import PPO
from tqdm import tqdm

from envs.icnp_sa import ICNP2021Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PPO testing arguments.")
    parser.add_argument("--model_dir", type=str, default="./results/icnp/")

    args = parser.parse_args()

    networks = ["NSFNet", "GBN", "GEANT2"]
    traffics = ["uniform", "gravity_1"]
    for network in networks:
        for traffic in traffics:
            if network == "NSFNet":
                episode_length = 100
            elif network == "GBN":
                episode_length = 150
            else:
                episode_length = 200
            env = make_vec_env(
                ICNP2021Env,
                n_envs=1,
                vec_env_cls=DummyVecEnv,
                seed=0,
                env_kwargs={"episode_length": episode_length, "env_type": network, "traffic_profile": traffic, "eval": True},
            )
            agent = PPO.load(args.model_dir + "model", device="cpu")
            n_episodes = 100
            best_max_load = []
            obs = env.reset()
            times = []
            for _ in tqdm(range(n_episodes)):
                done = False
                tic = time.time()
                while not done:
                    with torch.no_grad():
                        # action = [np.argmax(np.squeeze(obs)[:, 0])]
                        action = agent.policy._predict(torch.tensor(obs)).long().numpy().reshape(1, -1)
                    obs, rewards, dones, infos = env.step(action)
                    done = dones[0]
                times.append(time.time() - tic)
                best_max_load.append(infos[0]["max_load"])
            if not os.path.exists(args.model_dir + "eval"):
                os.mkdir(args.model_dir + "eval")
            np.savetxt(args.model_dir + "eval/" + f"{network}_{traffic}_test.txt", np.array(best_max_load), delimiter=",")
            np.savetxt(args.model_dir + "eval/" + f"times_{network}_{traffic}.txt", np.array(times), delimiter=",")
