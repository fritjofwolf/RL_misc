import math
import gym
import numpy as np

env = gym.make('CartPole-v1')


def makeState(observations, n):
    state = [0,0]
    e = 0.5
    tmp = (min(e, max(-e, observations[2])) + e) / (2*e)
    d = 3
    tmp2 = (min(d, max(-d, observations[3])) + d) / (2*d)
    #print(tmp)
    state[0] = math.floor(tmp * n)
    state[1] = math.floor(tmp2 * n)
    #print(state)
    return state[0]*n+state[1]


alpha = 0.1
gamma = 0.9
n = 6
qtable = np.zeros((n*n,2))
oldState = None
eps = 0.1
for j in range(1,10000):
    observation = env.reset()
    state = makeState(observation, n)
    cnt = 0
    if j%10000==0:
        eps /= 2
    #print(qtable)
    for i in range(200):
        cnt += 1
        # episode
        oldState = state
        # compute new action
        if np.random.rand() < eps:
            action = np.random.randint(2)
        else:
            action = 0 if qtable[state,0] > qtable[state,1] else 1
        observation, reward, done, info = env.step(action)
        #print(observation)
        state = makeState(observation, n)
        if done:
            print(cnt)
            break
        target = reward + gamma * max(qtable[state,:])
        qtable[oldState,action] += alpha * (target-qtable[oldState,action])
print(qtable)