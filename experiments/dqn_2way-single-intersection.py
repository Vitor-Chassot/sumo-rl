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
    # ------------------------------------------------------------------ #
    # Hiperparâmetros — centralizados aqui para facilitar sweeps no wandb #
    # ------------------------------------------------------------------ #
    config = {
        "delta_time": 5,
        "min_green": 5,
        "max_green": 60,
        "yellow_time": 2,
        "num_seconds": 100_000,
        "reward_weights": [0.8, 0.2],
        "learning_rate": 1e-3,
        "learning_starts": 200,
        "train_freq": 1,
        "target_update_interval": 500,
        "exploration_initial_eps": 0.05,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.01,
        "total_timesteps": 500_000,
    }

    run = wandb.init(
        project="sumo-rl-tcc",
        name="dqn_2way_pedestres",
        config=config,
        save_code=True,
    )

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection_pedestre.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh_pedestre.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=config["num_seconds"],
        reward_fn=["diff-waiting-time", "pedestrian-waiting-time"],
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
        learning_starts=config["learning_starts"],
        train_freq=config["train_freq"],
        target_update_interval=config["target_update_interval"],
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_fraction=config["exploration_fraction"],
        exploration_final_eps=config["exploration_final_eps"],
        verbose=1,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=SumoWandbCallback(
            log_system=True,
            log_per_agent=False,
            log_reward=True,
        ),
    )

    run.finish()
