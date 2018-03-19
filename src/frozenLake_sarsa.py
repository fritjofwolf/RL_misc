import gym
import numpy as np
import rl_methods as rlm


if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    q_function = np.zeros((16,4))
    q_function = rlm.train_qfunction_with_mc(q_function, env, 100000, 0.1, 0.9)
    rlm.evaluate_qfunction(q_function, env, 10000)