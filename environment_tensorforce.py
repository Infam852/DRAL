import numpy as np

import tensorflow as tf
from tensorforce.agents import Agent
from tensorforce.environments import Environment

from utils import load_mnist10, preprocess_data, split
from models import CNNModel, AuxModel
from datasets_manager import DatasetsManager
from settings import MAX_QUERIES, N_EVAL, NUM_CLASSES


def init_dm():
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = load_mnist10()
    X_TRAIN = X_TRAIN[:12000]   # !TODO just for tests
    Y_TRAIN = Y_TRAIN[:12000]
    x_train = preprocess_data(X_TRAIN, data_type='x')
    y_train = preprocess_data(Y_TRAIN, data_type='y')
    (x_eval, y_eval), (x_train, y_train) = split(x_train, y_train, N_EVAL)
    x_test = preprocess_data(X_TEST, data_type='x')
    y_test = preprocess_data(Y_TEST, data_type='y')
    print("""
        Shapes:
        x_train: {}
        x_eval: {}
        y_eval: {}
        x_test: {}
        y_test: {}
    """.format(x_train.shape, x_eval.shape, y_eval.shape,
               x_test.shape, y_test.shape))
    return DatasetsManager(x_train, x_eval, y_eval, x_test, y_test), y_train


class CustomEnvironment(Environment):
    def __init__(self, dm, session, cnn):
        super().__init__()
        self._state = np.zeros((1610,), dtype=np.float32)  # observation
        self._terminal = False
        self._queries = 0   # number of queries in episode
        self._counter = 0   # tracks idex of x_unl
        self.dm = dm
        self.session = session
        self.aux_model = AuxModel(cnn_model._model, 4)  # Flatten layer
        self.query_indicies = []

    def states(self):
        return dict(type='float', shape=(1610,))

    def actions(self):
        return dict(type='int', shape=(1,), num_values=2)  # num_values?

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()

    def close(self):
        super().close()

    def reset(self):
        self._counter = 0
        self._queries = 0
        self.query_indicies = []
        self._state = self._get_state_vector(self._counter)
        self._terminal = False
        return self._state

    def execute(self, actions):
        # print(f'actions: {actions}')
        reward = 0
        if actions[0] == 1:  # query label
            print(f'Add {self._counter} idx')
            self._queries += 1
            self.query_indicies.append(self._counter)  # add previous image to query list
            reward -= 0.1
        elif actions[0] == 0:
            pass
        else:
            raise ValueError(f'Action {actions} is not 0 or 1')

        # prepare next observation (aka next state)
        self._counter += 1
        next_state = self._get_state_vector(self._counter)
        print(f'*****ns: {next_state.shape}')
        # print(f'Predictions: {next_state[-NUM_CLASSES:]}, sum: {sum(next_state[-NUM_CLASSES:])}')
        reward += self._calculate_reward(next_state[-NUM_CLASSES:])  # last 10 values are the softmax outputs
        # print(f'counter: {self._counter}, STATE:', next_state.dtype)

        terminal = True if self._counter >= len(self.dm.unl) or \
            self._queries >= MAX_QUERIES else False
        if terminal and self._counter >= len(self.dm.unl) and \
                self._queries < MAX_QUERIES:
            reward -= 10

        return next_state, terminal, reward

    def _get_state_vector(self, counter):
        """ State vector has shape (F+O,) where F indicates length of
        feature vector extracted from CNN Flatten Layer and O indicates
        length of softmax output layer"""
        try:
            with self.session.as_default():
                with self.session.graph.as_default():
                    image = self.dm.unl.get_x(counter)
                    final_output, intermediate_output = self.aux_model.predict(
                        np.array([image]))
                    print(f'fo: {final_output.shape}, io: {intermediate_output.shape}')
                    print(f'ns: {np.append(intermediate_output, final_output, axis=0).shape}')

                    return np.append(intermediate_output, final_output)
        except Exception as ex:
            print('Aux model prediction error', ex, ex.__traceback__.tb_lineno)

    def _calculate_reward(self, predictions):
        # calculate entropy of the prediction
        entropy = (self.entropy(predictions))
        reward = 1 / entropy  # !TODO
        print(f'Entropy: {entropy} Reward: {reward}')
        return reward

    def entropy(self, dist):
        return -sum([p*np.log(p) for p in dist if p > 0])


if __name__ == '__main__':
    config = tf.compat.v1.ConfigProto(
        device_count={'GPU': 1},
        intra_op_parallelism_threads=1,
        allow_soft_placement=True
    )

    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.6

    session = tf.compat.v1.Session(config=config)
    dm, y_oracle = init_dm()
    cnn_model = CNNModel(dm.unl.get_feature_shape(), NUM_CLASSES, epochs=5)
    env = Environment.create(
        environment=CustomEnvironment, max_episode_timesteps=100,
        dm=dm, session=session, cnn=cnn_model,
    )

    agent = Agent.create(
        agent='dqn',
        states=env.states(),
        actions=env.actions(),
        batch_size=1,
        learning_rate=1e-3,
        memory=10000,
        exploration=0.2,
    )

    state = env.reset()
    # Train for 200 episodes
    for _ in range(20):
        print(f'[EPISODE {_}] started...')
        state = env.reset()
        terminal = False
        while True:
            print(f'state: {state}')
            print(f'state: {state.shape}')
            actions = agent.act(states=state)
            state, terminal, reward = env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
            if terminal:
                idxs = env.query_indicies
                print(f'query indicies: {idxs}')
                dm.label_samples(idxs, y_oracle[idxs])
                y_oracle = np.delete(y_oracle, idxs, axis=0)
                print(dm.train)
                cnn_model.fit(*dm.train.get_xy())
                cnn_model.print_evaluation(*dm.test.get_xy())
                break
