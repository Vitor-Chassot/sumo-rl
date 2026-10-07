import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from stable_baselines3.dqn.dqn import DQN

from sumo_rl import SumoEnvironment

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")


if __name__ == "__main__":
    prs = argparse.ArgumentParser(description="Visualiza no sumo-gui a política DQN treinada (demanda estacionária).")
    prs.add_argument("--model", default="outputs/2way-single-intersection/dqn_stationary_seed0.zip",
                     help="Modelo salvo pelo script de treino (.zip, com o .json da config ao lado).")
    prs.add_argument("--seconds", type=int, default=3600, help="Duração da simulação em segundos.")
    prs.add_argument("--seed", type=int, default=12345, help="Semente do SUMO (tráfego) para esta execução.")
    prs.add_argument("--delay", type=int, default=100, help="Atraso do sumo-gui por passo de simulação (ms).")
    prs.add_argument("--fixed", action="store_true", help="Roda o semáforo de tempo fixo da rede, para comparar.")
    prs.add_argument("--no-gui", action="store_true", help="Roda sem interface e só imprime as métricas.")
    args = prs.parse_args()

    model_path = Path(args.model)
    with open(model_path.with_suffix(".json")) as f:
        config = json.load(f)

    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-stationary.rou.xml",
        single_agent=True,
        use_gui=not args.no_gui,
        num_seconds=args.seconds,
        reward_fn="diff-waiting-time",
        min_green=config["min_green"],
        max_green=config["max_green"],
        enforce_max_green=True,
        delta_time=config["delta_time"],
        yellow_time=config["yellow_time"],
        fixed_ts=args.fixed,
        sumo_seed=args.seed,
        additional_sumo_cmd=f"--delay {args.delay}",
    )
    model = None if args.fixed else DQN.load(model_path, device="cpu")

    obs, info = env.reset()
    waiting, stopped = [], []
    done = False
    while not done:
        action = None if args.fixed else model.predict(obs, deterministic=True)[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        waiting.append(info["system_mean_waiting_time"])
        stopped.append(info["system_total_stopped"])
    env.close()

    policy = "tempo fixo" if args.fixed else model_path.name
    print(f"[{policy}] espera média: {np.mean(waiting):.1f} s | veículos parados (média): {np.mean(stopped):.1f}")
