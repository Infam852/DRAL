import numpy as np

from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.trajectories import time_step as ts

from utils import load_mnist10, preprocess_data, split
from models import CNNModel, AuxModel
from datasets_manager import DatasetsManager
from settings import N_QUERIES, N_EVAL, NUM_CLASSES


def init_dm(self):
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = load_mnist10()
    X_TRAIN = X_TRAIN[:20000]   # !TODO just for tests
    Y_TRAIN = Y_TRAIN[:20000]
    x_train = preprocess_data(X_TRAIN, data_type='x')
    y_train = preprocess_data(Y_TRAIN, data_type='y')
    (x_eval, y_eval), (x_train, y_train) = split(x_train, y_train, N_EVAL)
    x_test = preprocess_data(X_TEST, data_type='x')
    y_test = preprocess_data(Y_TEST, data_type='y')
    return DatasetsManager(x_train, x_eval, y_eval,
                           x_test, y_test, NUM_CLASSES)


class ClassifierEnv(py_environment.PyEnvironment):
    def __init__(self, dm):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int8, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(   # !TODO check range of values
            shape=(1610,), dtype=np.float32, minimum=-0.1, maximum=1.0,
            name='observation')
        self._state = np.zeros((1610,), dtype=np.float32)  # observation, should be updated and passes to timestep
        self._episode_ended = False
        self._n_queries = 0
        self.total_reward = 0
        self._counter = 0
        self.dm = dm
        self.cnn_model = CNNModel(self.dm.get_x_shape(), NUM_CLASSES, epochs=5)
        self.aux_model = AuxModel(self.cnn_model._model, 4)  # Flatten layer

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((1610,))
        self._n_queries = 0
        self._counter = 0
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if action == 1:
            self._n_queries += 1
            reward = -0.1
        elif action == 0:
            # do not query
            pass
        else:
            raise ValueError('action should be 0 or 1.')

        image = self.dm.get_x_unl(self._counter)
        # prepare next observation (aka next state)
        final_output, intermediate_output = self.aux_model.predict(
                                    np.array([image]))
        self._state = np.append(intermediate_output, final_output)
        self._counter += 1
        # end of episode
        if self._n_queries >= N_QUERIES or self._counter >= len(self.dm.x_unl):
            reward = 1.0  # !TODO
            return ts.termination(np.array(self._state, dtype=np.int32),
                                  reward)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.float32),
                reward=reward, discount=1.0
            )

    def _calculate_reward(self):
        pass

    def _crossentropy(self):
        pass


if __name__ == '__main__':
    dm = init_dm()
    environment = ClassifierEnv(dm)
    utils.validate_py_environment(environment, episodes=5)
