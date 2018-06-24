import gym
import numpy as np
import approximation_methods_nn as rlm
import policy_network_methods as pnm
from sklearn.neural_network import MLPRegressor

if __name__ == '__main__':
    # q_function = np.zeros((16,4))
    # q_function = rlm.train_qfunction_with_td0(q_function, env, 100000, 0.1, 0.9)
    # print(q_function)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)

    # q_function = np.zeros((16,4))
    # q_function = rlm.train_qfunction_with_td0(q_function, env, 100000, 0.01, 0.9)
    # print(q_function)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    
    # q_function = np.zeros((16,4))
    # q_function = rlm.train_qfunction_with_td0(q_function, env, 100000, 0.001, 0.9)
    # print(q_function)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # rlm.evaluate_qfunction(q_function, env, 1000)
    # env = gym.make('LunarLander-v2')
    env = gym.make('MountainCar-v0')  
    env = gym.make('CartPole-v0')  
    #env = gym.make('Acrobot-v1')
    #env = gym.make('Pendulum-v0')
    mlp = MLPRegressor(hidden_layer_sizes = (10), alpha=0.0001, max_iter=2000)
    mlp = rlm.train_linear_qfunction_with_td0(mlp, env, 3000, 0.01, 0.9)
    print("Die Gewichte sind: ", mlp.coefs_)
    print("Die Biase sind: ", mlp.intercepts_)
    rlm.evaluate_qfunction(mlp, env, 100)
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n
    # weights1 = np.random.randn(state_size, state_size)
    # weights2 = np.random.randn(action_size, state_size)
    # pnm.train_weights_local_search(weights1, weights2, env, 10000, 5)