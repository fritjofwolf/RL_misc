import numpy as np



def train_qfunction_with_mc(q_function, env, iter, alpha, gamma):
    for _ in range(iter):
        sar_list = []
        state = env.reset()
        action = compute_new_action_eps_greedy(state, q_function)
        sar_list.append((state, action, 0))
        done = False
        while not done:
            state, reward, done, info = env.step(action)
            if reward == 0 and done:
                reward = -1
            action = compute_new_action_eps_greedy(state, q_function)
            sar_list.append((state, action, reward))
        G = 0
        for elem in sar_list[::-1]:
            q_function[elem[0], elem[1]] += alpha * (G - q_function[elem[0], elem[1]])
            G = elem[2] + gamma * G
    return q_function

def train_qfunction_with_tdl(q_function, env, iter, alpha, gamma, lamb):
    for _ in range(iter):
        e_traces = np.zeros((16,4))
        state = env.reset()
        action = compute_new_action_eps_greedy(state, q_function)
        done = False
        while not done:
            new_state, reward, done, info = env.step(action)
            if reward == 0 and done:
                reward = -1
            new_action = compute_new_action_eps_greedy(new_state, q_function)
            delta = (reward+gamma*q_function[new_state,new_action] - q_function[state, action])
            e_traces[state, action] += 1
            for i in range(16):
                for j in range(4):
                    q_function[i, j] += alpha * delta * e_traces[i,j]
                    e_traces[i,j] *= gamma*lamb
            state, action = new_state, new_action
    return q_function


def train_qfunction_with_td0(q_function, env, iter, alpha, gamma):
    for _ in range(iter):
        state = env.reset()
        action = compute_new_action_eps_greedy(state, q_function)
        done = False
        while not done:
            # env.render()
            new_state, reward, done, info = env.step(action)
            if reward == 0 and done:
                reward = -1
            new_action = compute_new_action_eps_greedy(new_state, q_function)
            ret = (reward+gamma*q_function[new_state,new_action] - q_function[state, action])
            q_function[state, action] += alpha* ret
            state, action = new_state, new_action
    return q_function


def train_qfunction_weights_td0(weights, env, iter, alpha, gamma):
    """
    Assume that the feature vector is one-hot encoding the states and actions
    """
    for _ in range(iter):
        state = env.reset()
        action = compute_new_action_eps_greedy(state, q_function)
        done = False
        while not done:
            new_state, reward, done, info = env.step(action)
            
    return weights

def evaluate_qfunction(q_function, env, iter):
    sum_return = 0
    for _ in range(iter):
        state = env.reset()
        action = compute_new_action_greedy(state, q_function)
        done = False
        while not done:
            state, reward, done, info = env.step(action)
            action = compute_new_action_greedy(state, q_function)
            sum_return += reward
    print('Average reward per episode is:', sum_return/iter)

def compute_new_action_greedy(current_state, q_function):
    action = np.argmax(q_function[current_state,:])
    return action

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


def compute_greedy_policy_from_qfunction(qfunction):
    pass