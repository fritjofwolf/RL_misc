import gym
import numpy as np
from dqn import DQN
from sklearn.neural_network import MLPRegressor


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    mlp = MLPRegressor(hidden_layer_sizes = (10), alpha=0.01)
    dqn = DQN(mlp, env, 0.9, 0.5, 0.999, 1000, 30)
    dqn.train(6000)
    # print("Die Gewichte sind: ", mlp.coefs_)
    # print("Die Biase sind: ", mlp.intercepts_)
    dqn.evaluate_qfunction()


