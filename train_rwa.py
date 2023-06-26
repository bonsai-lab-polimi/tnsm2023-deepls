import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from algos.equivariant_policy import EquivariantActorCriticPolicy, IdentityFeatureExtractor
from envs.rwa_sa import RWAEnv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PPO training arguments.")
    parser.add_argument("--network", type=str, default="nsfnet")
    parser.add_argument("--episode_length", type=int, default=10)
    parser.add_argument("--n_slots", type=int, default=10)
    parser.add_argument("--n_requests", type=int, default=100)
    parser.add_argument("--n_cpu", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./results/rwa/")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    monitor_dir = args.log_dir
    env_kwargs = {
        "network": args.network,
        "episode_length": args.episode_length,
        "n_slots": args.n_slots,
        "n_requests": args.n_requests,
    }
    monitor_kwargs = {"info_keywords": ("blocking_rate", "ksp_blocking_rate", "best_blocking")}
    env = make_vec_env(
        RWAEnv,
        n_envs=args.n_cpu,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
        env_kwargs=env_kwargs,
        seed=args.seed,
    )
    policy_kwargs = dict(features_extractor_class=IdentityFeatureExtractor, features_extractor_kwargs=dict(features_dim=3))
    agent = PPO(EquivariantActorCriticPolicy, env, n_steps=64, verbose=1, policy_kwargs=policy_kwargs, device="cpu")
    agent.learn(30000)
    agent.save(monitor_dir + "model")
