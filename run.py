import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.environments import utils
from tf_agents.utils import common

from environment import ClassifierEnv, init_dm


if __name__ == "__main__":
    dm = init_dm()
    env = ClassifierEnv(dm)
    utils.validate_py_environment(env, episodes=5)

    # init q network (FNN with 2 layers)
    fc_layer_params = (64, 64,)

    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params)

