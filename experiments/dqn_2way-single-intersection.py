import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN
from sumo_rl.environment.observations import PedestrianObservationFunction

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment


if __name__ == "__main__":
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection_pedestre.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh_pedestre.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=40000,
        reward_fn=["diff-waiting-time", "pedestrian-waiting-time"],
        reward_weights=[4.0, 1.0],
        observation_class=PedestrianObservationFunction,
        min_green=15,
        max_green=60,
        enforce_max_green=True,
        delta_time=15,
        yellow_time=3,
    )

    model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=3e-4,
        learning_starts=200,
        train_freq=1,
        target_update_interval=200,
        exploration_initial_eps=1.0,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        verbose=1,
    )
    model.learn(total_timesteps=100000)
