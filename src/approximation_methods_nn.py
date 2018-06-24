import numpy as np
import progressbar
from collections import deque
import random

state_space = 4
action_space= 2

def train_linear_qfunction_with_td0(mlp, env, iter, alpha, gamma):
    # bar = progressbar.ProgressBar()
    eps = 0.5
    mlp = do_initial_fit(mlp, env)
    replay_mem = deque(maxlen = 10000)
    for i in range(iter):
        if i == 200:
            alpha /= 10
        elif i == 1000:
            alpha /= 10
        eps *= 0.999
        #if i % 100 == 0 and i != 0 or i == 1:
            #print('Die Gewichte und Biase nach ' + str(i)+ ' iterationen sind:', mlp.coefs_, mlp.intercepts_)
        state = env.reset()
        action = compute_new_action_eps_greedy(state, mlp, eps)
        done = False
        old_qvalue = compute_qfunction(mlp, state)
        cnt = 0
        while not done:
            # env.render()
            new_state, reward, done, info = env.step(action)
            if done and cnt != 199:
                reward = -10
            new_action = compute_new_action_eps_greedy(new_state, mlp, eps)
            #print(new_action)
            new_qvalue = compute_qfunction(mlp, new_state)
            #print(compute_qfunction(mlp, new_state))
            target = compute_target(old_qvalue, new_qvalue, gamma, reward, action, new_action)
            replay_mem.append((state, target))
            x, y = get_training_set(replay_mem)
            mlp.partial_fit(x, y)
            state, action = new_state, new_action
            old_qvalue = new_qvalue
            cnt += 1
        print(i, cnt, eps)
    return mlp

def get_training_set(replay_mem):
    x = []
    y = []
    for i in range(30):
        (a,b) = replay_mem[random.randrange(len(replay_mem))]
        x.append(a)
        y.append(b)
    return x, y
        

def compute_target(old_qvalue, new_qvalue, gamma, reward, action, new_action):
    ret = (reward+gamma*new_qvalue[new_action] - old_qvalue[action])
    target = old_qvalue.copy()
    target[action] = ret
    return target

def do_initial_fit(mlp, env):
    state = env.reset()
    mlp.partial_fit([state], [[0 for i in range(action_space)]])
    return mlp

def evaluate_qfunction(mlp, env, iter):
    sum_return = 0
    for _ in range(iter):
        episode_reward = 0
        state = env.reset()
        action = compute_new_action_greedy(state, mlp)
        done = False
        while not done:
            #env.render()
            state, reward, done, info = env.step(action)
            action = compute_new_action_greedy(state, mlp)
            sum_return += reward
            episode_reward += reward
        #print(episode_reward)
    print('Average reward per episode is:', sum_return/iter)


def compute_new_action_greedy(state, mlp):
    action_values = mlp.predict([state])
    action = np.argmax(action_values)
    return action


def compute_new_action_eps_greedy(state, mlp, eps):
    if np.random.rand() < eps:
        action = np.random.randint(action_space)
    else:
        action_values = mlp.predict([state])
        action = np.argmax(action_values)
    return action


def compute_qfunction(mlp, state):
    return mlp.predict([state])[0]

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