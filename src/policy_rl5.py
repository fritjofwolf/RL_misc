"""
Find optimal weights with policy gradient based Monte-Carlo
estimates using REINFORCE algorithm
"""
import gym
import numpy as np

env = gym.make('CartPole-v0')

def play_episode(weights):
    states = []
    actions = []
    rewards = []
    global env
    state = env.reset()
    done = False
    while not done:
        states.append(state)
        action_left = np.array(list(state)+[1,0])
        action_right = np.array(list(state)+[0,1])
        action_probs = np.zeros(2)
        action_probs[0] = np.dot(action_left, weights)
        action_probs[1] = np.dot(action_right, weights)
        probs = np.zeros(2)
        probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
        print(action_probs,np.exp(action_probs), np.sum(np.exp(action_probs)), probs)
        if probs[1] < np.random.rand():
            state, reward, done, info = env.step(1)
            actions.append(1)
        else:
            state, reward, done, info = env.step(0)
            actions.append(0)
        rewards.append(reward)
    return states, actions, rewards

def compute_weights_update(weights, states, actions, rewards):
    returns = np.cumsum(rewards)[::-1]
    alpha = 0.1
    for i in range(len(states)):
        action_left = np.array(list(states[i])+[1,0])
        action_right = np.array(list(states[i])+[0,1])
        action_probs = np.zeros(2)
        action_probs[0] = np.dot(action_left, weights)
        action_probs[1] = np.dot(action_right, weights)
        probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
        expected_value = probs[0] * action_left + probs[1] * action_right
        if actions[i] == 0:
            gradient = action_left - expected_value
        else:
            gradient = action_right - expected_value
        weights += alpha*gradient*returns[i]
    return weights

best_weights = np.zeros(6)
best_score = 0
weights = np.random.randn(6)
for i in range(10):
    states, actions, rewards = play_episode(weights)
    weights = compute_weights_update(weights[:], states, actions, rewards)
    score = sum(rewards)
    if score > best_score:
        best_score = score
        best_weights = weights
    print(i, score, best_score)
    if best_score > 195:
        print('Number of iterations', i)
        print('Best weights are', best_weights)
        break