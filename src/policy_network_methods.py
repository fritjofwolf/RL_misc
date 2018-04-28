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

def rect_linear(x):
    x[x < 0] = 0
    return x

def compute_action(weights1, weights2, state):
    activation1 = np.dot(weights1, state)
    activation2 = rect_linear(activation1)
    activation3 = np.dot(weights2, activation2)
    probs = softmax(activation3)
    # cum_sum_probs = np.cumsum(probs)
    #print(np.argmax(probs))
    return np.argmax(probs)
    # random_value = np.random.rand()
    # for idx, value in enumerate(cum_sum_probs):
    #     if random_value < value:
    #         return idx

def train_weights_local_search(weights1, weights2, env, iter1, iter2, depth=1):
    best_score = -1000000
    best_weights1 = weights1
    best_weights2 = weights2
    #bar = progressbar.ProgressBar()
    alpha = 10
    for i in range(iter1):
        current_weights1 = best_weights1 + alpha*np.random.standard_normal(best_weights1.shape)
        current_weights2 = best_weights2 + alpha*np.random.standard_normal(best_weights2.shape)
        current_score = 0
        for _ in range(iter2):
            state = env.reset()
            action = compute_action(current_weights1, current_weights2, state)
            done = False
            while not done:
                state, reward, done, _ = env.step(action)
                #env.render()
                action = compute_action(current_weights1, current_weights2, state)
                #if action == 1:
                #    action = 2
                current_score += reward
        current_score /= iter2
        print(i, current_score)
        if current_score > best_score:
            best_score = current_score
            best_weights1 = current_weights1
            best_weights2 = current_weights2
        if best_score > -120:
            alpha = 0.1
    print("Best Score is: ", best_score)
    print("Best Weights are: ", best_weights1, best_weights2)
    return best_weights1, best_weights2

def train_weights_numerical_gradient_method(weights, env, iter, depth=1):
    # best_score = 0
    # #bar = progressbar.ProgressBar()
    # for i in range(1000):
    #     weights = np.random.randn(2,4)
    #     current_score = 0
    #     for _ in range(1):
    #         state = env.reset()
    #         action = compute_action(weights, state)
    #         done = False
    #         while not done:
    #             state, reward, done, info = env.step(action)
    #             action = compute_action(weights, state)
    #             current_score += reward
    #     current_score /= 1
    #     #print(current_score)
    #     if current_score == 200:
    #         print(i)
    #         return
    # print("Best Score is: ", best_score)
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