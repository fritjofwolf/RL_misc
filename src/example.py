import gym
import numpy as np

env = gym.make('CartPole-v0')

def computeAction(observation, weights):
    if np.random.rand() < 0.1:
        return np.random.randint(2)
    valueLeft = computeQValue(observation, weights, 0)
    valueRight = computeQValue(observation, weights, 1)
    #print(valueLeft, valueRight)
    if valueLeft < valueRight:
        return 1
    else:
        return 0

def computeQValue(observation, weights, tmp):
    #print(observation)
    if tmp == 0:
        features = observation
        features = np.append(features, 1)
        features = np.append(features, 1)
        features = np.append(features, 0)
        
    else:
        features = observation
        features = np.append(features, 1)
        features = np.append(features, 0)
        features = np.append(features, 1)
    return np.matmul(weights, features)
    

# weights = np.random.rand(6) * 2 - 1
weights = np.zeros(7)
alpha = 1
gamma = 0.9
oldState = None
for _ in range(10000):
    observation = env.reset()
    cnt = 0
    print(weights)
    for i in range(200):
        cnt += 1
        # episode
        oldState = observation
        action = computeAction(observation, weights)
        observation, reward, done, info = env.step(action)
        if done:
            print(cnt)
            break
        tmp1 = computeQValue(observation, weights, 0)
        tmp2 = computeQValue(observation, weights, 1)
        #print(tmp1, tmp2)
        if tmp1 < tmp2:
            nextAction = 1
        else:
            nextAction = 0
        target = reward + gamma * max(tmp1,tmp2)
        #print(nextAction)
        if action == 0:
            tmpOldState = oldState
            tmpOldState = np.append(tmpOldState, 1)
            tmpOldState = np.append(tmpOldState, 1)
            tmpOldState = np.append(tmpOldState, 0)
        else:
            tmpOldState = oldState
            tmpOldState = np.append(tmpOldState, 1)
            tmpOldState = np.append(tmpOldState, 0)
            tmpOldState = np.append(tmpOldState, 1)
        #print(tmpOldState)
        deltaW = alpha*(target - computeQValue(oldState, weights, action))*tmpOldState
        #print(deltaW, weights)
        weights = weights + deltaW
        