import numpy as np
import progressbar

state_space = 5
action_space = 2

def train_linear_qfunction_with_td0(weights, env, iter, alpha, gamma):
    bar = progressbar.ProgressBar()
    for _ in bar(range(iter)):
        state = env.reset()
        action = compute_new_action_eps_greedy(state, weights)
        done = False
        old_features = create_feature_vector_cartpole(state, action)
        old_qvalue = compute_qfunction(weights, old_features)
        while not done:
            # env.render()
            new_state, reward, done, info = env.step(action)
            # if reward == 0 and done:
            #     reward = -1
            new_action = compute_new_action_eps_greedy(new_state, weights)
            #print(new_action)
            new_features = create_feature_vector_cartpole(new_state, new_action)
            new_qvalue = compute_qfunction(weights, new_features)
            ret = (reward+gamma*new_qvalue - old_qvalue)
            weights += alpha * ret * old_features
            state, action = new_state, new_action
            old_qvalue = new_qvalue
            old_features = new_features
    return weights


def evaluate_qfunction(weights, env, iter):
    sum_return = 0
    for _ in range(iter):
        episode_reward = 0
        state = env.reset()
        action = compute_new_action_greedy(state, weights)
        done = False
        while not done:
            #env.render()
            state, reward, done, info = env.step(action)
            action = compute_new_action_greedy(state, weights)
            sum_return += reward
            episode_reward += reward
        print(episode_reward)
    print('Average reward per episode is:', sum_return/iter)


def compute_new_action_greedy(state, weights):
    action_values = np.zeros(action_space)
    for i in range(action_space):
        features = create_feature_vector_cartpole(state, i)
        action_values[i] = compute_qfunction(weights, features)
    action = np.argmax(action_values)
    return action


def compute_new_action_eps_greedy(state, weights):
    if np.random.rand() < 0.1:
        action = np.random.randint(action_space)
    else:
        action_values = np.zeros(action_space)
        for i in range(action_space):
            features = create_feature_vector_cartpole(state, i)
            action_values[i] = compute_qfunction(weights, features)
        action = np.argmax(action_values)
    return action


def compute_qfunction(weights, features):
    return np.inner(weights, features)

def create_feature_vector_cartpole(state, action):
    feature_vector = np.zeros(action_space*state_space)
    state = np.append(state, 1)
    if action == 0:
        feature_vector[:state_space] = state
    elif action == 1:
        feature_vector[state_space:] = state
    return feature_vector

def create_feature_vector_mountaincar(state, action):
    feature_vector = np.zeros(action_space*state_space)
    state = np.append(state, 1)
    if action == 0:
        feature_vector[:state_space] = state
    elif action == 1:
        feature_vector[state_space:2*state_space] = state
    elif action == 2:
        feature_vector[2*state_space:]
    return feature_vector