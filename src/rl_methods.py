import numpy as np

def train_qfunction_with_mc(q_function, env):
    pass


def train_qfunction_with_tdl(q_function, env):
    pass


def train_qfunction_with_td0(q_function, env, iter, alpha, gamma):
    for _ in range(iter):
        state = env.reset()
        action = compute_new_action_eps_greedy(state, q_function)
        done = False
        while not done:
            env.render()
            new_state, reward, done, info = env.step(action)
            new_action = compute_new_action_eps_greedy(new_state, q_function)
            ret = (reward+gamma*q_function[new_state,new_action] - q_function[state, action])
            q_function[state, action] += alpha* ret
            state, action = new_state, new_action
    return q_function


def evaluate_qfunction(q_function, env):
    sum_return = 0
    for _ in range(1000):
        state = env.reset()
        action = compute_new_action_eps_greedy(state, q_function)
        done = False
        while not done:
            state, reward, done, info = env.step(action)
            action = compute_new_action_eps_greedy(state, q_function)
            sum_return += reward
    print('Average reward per episode is:', sum_return/1000)


def compute_new_action_eps_greedy(current_state, q_function):
    eps = 0.1
    action = 0
    if np.random.rand() < eps:
        action = np.random.randint(0,4)
    else:
        action = np.argmax(q_function[current_state,:])
    return action


def compute_new_action_random(current_state):
    return np.random.randint(0,4)