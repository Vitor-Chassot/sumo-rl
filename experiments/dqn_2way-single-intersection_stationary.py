import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
import wandb

from sumo_rl import SumoEnvironment
from sumo_rl.environment.wandb_callback import SumoWandbCallback

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


class EpisodeSeedWrapper(gym.Wrapper):
    """Usa uma semente do SUMO diferente (e reprodutível) a cada episódio.

    Sem isso, o sumo_seed fixo é reaplicado em todo reset e todos os
    episódios teriam exatamente as mesmas chegadas de veículos.
    """

    def __init__(self, env, base_seed: int):
        super().__init__(env)
        self.base_seed = base_seed
        self.episode = 0

    def reset(self, **kwargs):
        kwargs["seed"] = self.base_seed * 10_000 + self.episode
        self.episode += 1
        return self.env.reset(**kwargs)


if __name__ == "__main__":
    prs = argparse.ArgumentParser(description="DQN baseline (sem pedestres) com demanda estacionária.")
    prs.add_argument("--seed", type=int, default=0, help="Semente do SUMO e do DQN.")
    prs.add_argument("--gui", action="store_true", help="Abre o sumo-gui.")
    args = prs.parse_args()

    # ------------------------------------------------------------------ #
    # Hiperparâmetros — centralizados aqui para facilitar sweeps no wandb #
    # ------------------------------------------------------------------ #
    config = {
        "seed": args.seed,
        "delta_time": 5,
        "min_green": 5,
        "max_green": 60,
        "yellow_time": 3,  # valor que o netconvert calcula para vias de 13,9 m/s
        "num_seconds": 21_600,  # 6 h por episódio
        "num_episodes": 30,
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "train_freq": 1,
        "target_update_interval": 2000,
        "exploration_initial_eps": 0.05,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.01,
    }
    # Cada passo do agente avança delta_time segundos de simulação.
    config["total_timesteps"] = config["num_episodes"] * config["num_seconds"] // config["delta_time"]

    run = wandb.init(
        project="sumo-rl-tcc",
        name=f"dqn_2way_stationary_seed{args.seed}",
        config=config,
        save_code=True,
    )

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-stationary.rou.xml",
        out_csv_name=f"outputs/2way-single-intersection/dqn_stationary_seed{args.seed}",
        single_agent=True,
        use_gui=args.gui,
        num_seconds=config["num_seconds"],
        reward_fn="diff-waiting-time",
        min_green=config["min_green"],
        max_green=config["max_green"],
        enforce_max_green=True,
        delta_time=config["delta_time"],
        yellow_time=config["yellow_time"],
    )
    env = EpisodeSeedWrapper(env, base_seed=config["seed"])

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        train_freq=config["train_freq"],
        target_update_interval=config["target_update_interval"],
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        seed=config["seed"],
        verbose=1,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=SumoWandbCallback(
            log_system=True,
            log_per_agent=False,
            log_reward=True,
            log_episode=True,
        ),
    )

    # Salva o modelo e a config (usada pelo script de visualização para recriar o mesmo ambiente).
    model_path = f"outputs/2way-single-intersection/dqn_stationary_seed{args.seed}"
    model.save(model_path)
    with open(f"{model_path}.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Modelo salvo em {model_path}.zip")

    env.close()
    run.finish()
