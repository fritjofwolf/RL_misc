import numpy as np
import progressbar

state_space = 4
action_space = 2

def train_weights_random_search(env):
    best_score = 0
    #bar = progressbar.ProgressBar()
    for i in range(1000):
        weights = np.random.randn(2,4)
        current_score = 0
        for _ in range(1):
            state = env.reset()
            action = compute_action(weights, state)
            done = False
            while not done:
                state, reward, done, info = env.step(action)
                action = compute_action(weights, state)
                current_score += reward
        current_score /= 1
        #print(current_score)
        if current_score == 200:
            print(i)
            return
    print("Best Score is: ", best_score)    

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

def compute_action(weights, state):
    activation_values = np.dot(weights, state)
    probs = softmax(activation_values)
    cum_sum_probs = np.cumsum(probs)
    random_value = np.random.rand()
    for idx, value in enumerate(cum_sum_probs):
        if random_value < value:
            return idx

def train_weights_local_search(weights, env, iter, iter2, depth=1):
    best_score = -1000000
    best_weights = weights
    #bar = progressbar.ProgressBar()
    for i in range(iter):
        current_weights = best_weights + np.random.standard_normal(best_weights.shape)
        current_score = 0
        for _ in range(iter2):
            state = env.reset()
            action = compute_action(current_weights, state)
            done = False
            while not done:
                state, reward, done, info = env.step(action)
                action = compute_action(current_weights, state)
                current_score += reward
        current_score /= iter2
        print(i, current_score)
        if current_score > best_score:
            best_score = current_score
            best_weights = current_weights
    print("Best Score is: ", best_score)
    return best_weights

def train_weights_gradient_method(weights, env, iter, depth=1):
    pass

def train_weights_ga(weights, env, iter, depth=1):
    pass

def create_feature_vector_cartpole(state, action):
    feature_vector = np.zeros(action_space*state_space)
    if action == 0:
        feature_vector[:state_space] = state
    elif action == 1:
        feature_vector[state_space:] = state
    return feature_vector

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