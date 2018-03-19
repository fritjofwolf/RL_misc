"""
Find optimal weights with policy gradient based on finite-differences
using a softmax approach with two outputs
"""
import gym
import numpy as np

env = gym.make('CartPole-v0')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def evaluateWeights(weights):
    global env
    sum = 0
    iterations = 10
    for _ in range(iterations):
        state = env.reset()
        done = False
        while not done:
            #print('sigmoid is', sigmoid(np.inner(weights, state)))
            action_probs = np.dot(state, weights)
            probs = np.exp(action_probs) / np.sum(np.exp(action_probs))
            # print(probs)
            if probs[0] > np.random.rand():
                state, reward, done, info = env.step(1)
            else:
                state, reward, done, info = env.step(0)
            sum += 1
    return sum / iterations

def compute_gradient(weights):
    eps = 1
    gradient = np.zeros((4,2))
    old_value = evaluateWeights(weights)
    for k in range(0,4):
        for j in range(0,2):
            tmp_weights = weights.copy()
            tmp_weights[k,j] += eps
            new_value = evaluateWeights(tmp_weights)
            gradient[k,j] = (new_value - old_value) / eps
    return gradient

best_weights = np.zeros(4)
best_score = 0
weights = np.zeros((4,2))
for i in range(10000):
    gradient = compute_gradient(weights)
    print(weights, gradient)
    weights += 0.1*gradient
    score = evaluateWeights(weights)
    if score > best_score:
        best_score = score
        best_weights = weights
    print(i, score, best_score)
    if best_score > 195:
        print('Number of iterations', i)
        print('Best weights are', best_weights)
        break