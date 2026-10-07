import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3.dqn.dqn import DQN
import wandb

from sumo_rl import SumoEnvironment
from sumo_rl.environment.observations import PedestrianObservationFunction
from sumo_rl.environment.wandb_callback import SumoWandbCallback

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


if __name__ == "__main__":
    prs = argparse.ArgumentParser(description="DQN com pedestres e demanda estacionária.")
    prs.add_argument("--seed", type=int, default=0, help="Semente do SUMO e do DQN.")
    prs.add_argument("--gui", action="store_true", help="Abre o sumo-gui.")
    args = prs.parse_args()

    # ------------------------------------------------------------------ #
    # Hiperparâmetros — centralizados aqui para facilitar sweeps no wandb #
    # ------------------------------------------------------------------ #
    config = {
        "seed": args.seed,
        "net_file": "sumo_rl/nets/2way-single-intersection/single-intersection_pedestre.net.xml",
        "route_file": "sumo_rl/nets/2way-single-intersection/single-intersection-stationary_pedestre.rou.xml",
        "reward_fn": ["diff-waiting-time", "pedestrian-waiting-time"],
        "reward_weights": [0.8, 0.2],
        "observation": "pedestrian",
        "delta_time": 5,
        "min_green": 5,
        "max_green": 60,
        "yellow_time": 3,  # valor que o netconvert calcula para vias de 13,9 m/s
        "num_seconds": 21_600,  # 6 h por episódio
        "num_episodes": 10,
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
        name=f"dqn_2way_pedestres_seed{args.seed}",
        config=config,
        save_code=True,
    )

    env = SumoEnvironment(
        net_file=config["net_file"],
        route_file=config["route_file"],
        out_csv_name=f"outputs/2way-single-intersection/dqn_pedestre_seed{args.seed}",
        single_agent=True,
        use_gui=args.gui,
        num_seconds=config["num_seconds"],
        reward_fn=config["reward_fn"],
        reward_weights=config["reward_weights"],
        observation_class=PedestrianObservationFunction,
        min_green=config["min_green"],
        max_green=config["max_green"],
        enforce_max_green=True,
        delta_time=config["delta_time"],
        yellow_time=config["yellow_time"],
    )

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
    model_path = f"outputs/2way-single-intersection/dqn_pedestre_seed{args.seed}"
    model.save(model_path)
    with open(f"{model_path}.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Modelo salvo em {model_path}.zip")

    env.close()
    run.finish()
