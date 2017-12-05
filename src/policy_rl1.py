"""
Find optimal weights by purely sampling random weights till the result is good enough
"""
import gym
import numpy as np

env = gym.make('CartPole-v0')

def evaluateWeights(weights):
    global env
    sum = 0
    for i in range(100):
        state = env.reset()
        done = False
        while not done:
            if np.inner(weights, state) > 0:
                state, reward, done, info = env.step(1)
            else:
                state, reward, done, info = env.step(0)
            sum += 1
    return sum // 100


best_weights = np.zeros(4)
best_score = 0
for i in range(1000):
    # weights = [0,0,np.random.randn(1), np.random.randn(1)]
    weights = (np.random.rand(4)*2)-1
    score = evaluateWeights(weights)
    if score > best_score:
        best_score = score
        best_weights = weights
    print(best_score)
    if best_score > 195:
        print('Number of iterations', i)
        print('Best weights are', best_weights)
        break