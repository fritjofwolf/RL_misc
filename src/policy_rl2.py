"""
Find optimal weights with stochastic local function optimization
"""
import gym
import numpy as np

env = gym.make('CartPole-v0')

def evaluateWeights(weights):
    global env
    sum = 0
    for i in range(10):
        state = env.reset()
        done = False
        while not done:
            if np.inner(weights, state) > 0:
                state, reward, done, info = env.step(1)
            else:
                state, reward, done, info = env.step(0)
            sum += 1
    return sum // 10


best_weights = np.zeros(4)
best_score = 0
for i in range(1000):
    weight_change = [0,0,0.1*np.random.randn(1), 0.1*np.random.randn(1)]
    # weight_change = 0.1*((np.random.rand(4)*2)-1)
    weights = best_weights + weight_change
    score = evaluateWeights(weights)
    if score > best_score:
        best_score = score
        best_weights = weights
    print(best_score)
    if best_score > 195:
        print('Number of iterations', i)
        print('Best weights are', best_weights)
        break