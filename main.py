import os
import logging
import hydra
import torch as th
import gymnasium as gym
import torch.nn as nn
import torch.nn.functional as F
import shutil

from datetime import datetime
from omegaconf import OmegaConf
from gymnasium.wrappers import RescaleAction
from agent import DQN
from hydra.core.hydra_config import HydraConfig

logging.basicConfig(format='%(asctime)s: %(message)s', datefmt=' %I:%M:%S %p', level=logging.INFO)

class Qfunc(nn.Module):
    def __init__(self, obs_dims, act_dims):
        super(Qfunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, act_dims))

    def forward(self, state):
        return self.net(state)

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg):
    logging.info(OmegaConf.to_yaml(cfg))

    hydra_cfg = HydraConfig.get()

    logging.info(hydra_cfg.run.dir)

    # backup main and agent file
    date_key = datetime.now().strftime('%Y-%m-%d')
    time_key = datetime.now().strftime('%H-%M-%S')
    outputs_dir = f'{hydra_cfg.run.dir}/backups/'

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    shutil.copyfile('main.py', outputs_dir + 'main.txt')
    shutil.copyfile('agent.py', outputs_dir + 'agent.txt')

    env = gym.make(cfg.env_cfg.env_name)

    # exit()

    act_dims = env.action_space.n    # .shape[0]
    obs_dims = env.observation_space.shape[0]

    agent = DQN(
        env=env,
        network=Qfunc(obs_dims, act_dims),
        warmup_len=1_000,
        gamma=cfg.env_cfg.gamma,
        logger_kwargs=dict(
            tensorboard=cfg.exp_cfg.use_tb,
            log_interval=cfg.exp_cfg.log_int,
            log_dir=hydra_cfg.run.dir,
            exp_name=cfg.exp_cfg.exp_name,
            episode_window=50
            )
    )

    for epochs in range(cfg.epochs):
        avg_r = agent.fit(cfg.epoch_steps)

if __name__ == '__main__':
    main()