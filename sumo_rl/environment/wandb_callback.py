"""Callback para logar métricas do SUMO-RL e do DQN no Weights & Biases."""

import wandb
from stable_baselines3.common.callbacks import BaseCallback


# Prefixos das chaves do info dict que devem ser logadas no wandb.
_SYSTEM_PREFIXES = (
    "system_",
    "agents_",
    "step",
)


class SumoWandbCallback(BaseCallback):
    """Loga métricas do SumoEnvironment e do modelo DQN no W&B a cada step.

    Combina duas fontes:
    - Info dict do SUMO (tempos de espera, pedestres, velocidade, etc.)
    - Logger interno do SB3 (loss, learning_rate, exploration_rate, n_updates)

    Ao fim de cada episódio, loga também a média de cada métrica "system_*"
    no episódio ("episode/*"), que é a curva de aprendizado entre episódios.

    Args:
        log_system (bool): Loga chaves "system_*" e "agents_*" do info dict.
        log_per_agent (bool): Loga métricas por agente (ex.: "t_stopped").
        log_reward (bool): Loga a recompensa escalar a cada step.
        log_episode (bool): Loga as médias por episódio ("episode/*").
        verbose (int): Nível de verbosidade herdado do BaseCallback.
    """

    def __init__(
        self,
        log_system: bool = True,
        log_per_agent: bool = False,
        log_reward: bool = True,
        log_episode: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_system = log_system
        self.log_per_agent = log_per_agent
        self.log_reward = log_reward
        self.log_episode = log_episode
        self._episode = 0
        self._reset_episode_stats()

    def _reset_episode_stats(self):
        self._episode_sums = {}
        self._episode_len = 0
        self._episode_return = 0.0

    def _on_step(self) -> bool:
        """Chamado pelo SB3 a cada step de ambiente."""
        log_dict = {}

        # --- Métricas do SUMO ---
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        for key, value in info.items():
            if self.log_system and key.startswith(_SYSTEM_PREFIXES):
                log_dict[f"sumo/{key}"] = value
            elif self.log_per_agent and not key.startswith(_SYSTEM_PREFIXES):
                log_dict[f"sumo/agent/{key}"] = value

        rewards = self.locals.get("rewards", [None])
        reward = rewards[0] if rewards else None
        if self.log_reward and reward is not None:
            log_dict["sumo/reward"] = float(reward)

        # --- Médias por episódio ---
        # O SB3 só calcula as estatísticas de episódio (rollout/ep_rew_mean)
        # depois deste callback e limpa o logger em seguida, então elas não
        # chegam ao name_to_value lido abaixo; por isso agregamos aqui.
        if self.log_episode:
            for key, value in info.items():
                if key.startswith("system_"):
                    self._episode_sums[key] = self._episode_sums.get(key, 0.0) + value
            self._episode_len += 1
            if reward is not None:
                self._episode_return += float(reward)

            dones = self.locals.get("dones", [False])
            if dones[0]:
                self._episode += 1
                log_dict["episode/number"] = self._episode
                log_dict["episode/return"] = self._episode_return
                for key, total in self._episode_sums.items():
                    log_dict[f"episode/{key.removeprefix('system_')}"] = total / self._episode_len
                self._reset_episode_stats()

        # --- Métricas internas do DQN (via logger do SB3) ---
        # O logger do SB3 acumula valores entre dumps; lemos diretamente
        # do name_to_value para não depender do sync_tensorboard do wandb.
        sb3_logger = self.model.logger
        if hasattr(sb3_logger, "name_to_value"):
            for key, value in sb3_logger.name_to_value.items():
                log_dict[f"train/{key}"] = value

        if log_dict:
            wandb.log(log_dict, step=self.num_timesteps)

        return True
