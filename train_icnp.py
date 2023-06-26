import argparse

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.ppo import PPO

from algos.equivariant_policy import EquivariantActorCriticPolicy, IdentityFeatureExtractor
from envs.icnp_sa import ICNP2021Env

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PPO training arguments.")
    parser.add_argument("--env_type", type=str, default="nsfnet")
    parser.add_argument("--episode_length", type=int, default=10)
    parser.add_argument("--traffic_profile", type=str, default="uniform")
    parser.add_argument("--n_cpu", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./results/icnp/")
    args = parser.parse_args()

    monitor_dir = args.log_dir

    monitor_kwargs = {"info_keywords": ("starting_max_load", "max_load", "n_accepted")}
    env_kwargs = {"env_type": args.env_type, "episode_length": args.episode_length, "traffic_profile": args.traffic_profile}
    env = make_vec_env(
        ICNP2021Env,
        n_envs=args.n_cpu,
        vec_env_cls=SubprocVecEnv,
        monitor_dir=monitor_dir,
        monitor_kwargs=monitor_kwargs,
        env_kwargs=env_kwargs,
        seed=0,
    )
    policy_kwargs = dict(features_extractor_class=IdentityFeatureExtractor, features_extractor_kwargs=dict(features_dim=2))
    agent = PPO(EquivariantActorCriticPolicy, env, learning_rate=0.0003, n_steps=128, verbose=1, policy_kwargs=policy_kwargs)
    agent.learn(300000)
    agent.save(monitor_dir + "model")
