"""
Q-Learning with linear function approximation for a simple openai gym with continuous inputs
"""
import gym
import numpy as np

def compute_action_value(state, action):
    ext_state = construct_extended_state(state, action)
    return np.inner(ext_state, weights)

def construct_extended_state(state, action):
    if action == 0:
        ext_state = np.array(list(state)+[0,0,0,0]+[1])
    elif action == 1:
        ext_state = np.array([0,0,0,0]+list(state)+[1])
    return ext_state

env = gym.make('CartPole-v0')

weights = np.random.randn(9)
alpha = 0.1
gamma = 0.99
epsilon = 1
for i in range(1000):
    # if i % 1000 == 0:
    #     print(weights)
    #     epsilon /= 2
    state = env.reset()
    done = False
    action_value_left = compute_action_value(state, 0)
    action_value_right = compute_action_value(state, 1)
    cnt = 0
    while not done:
        old_state = state
        if np.random.rand() < epsilon:
            if np.random.rand() < 0.5:
                state, reward, done, info = env.step(1)
                old_action_value = action_value_right
                old_action = 1
            else:
                state, reward, done, info = env.step(0)
                old_action_value = action_value_left
                old_action = 0  
        else:
            if action_value_left < action_value_right:
                state, reward, done, info = env.step(1)
                old_action_value = action_value_right
                old_action = 1
            else:
                state, reward, done, info = env.step(0)
                old_action_value = action_value_left
                old_action = 0  
        # print(old_action)        
        action_value_left = compute_action_value(state, 0)
        action_value_right = compute_action_value(state, 1)
        target = reward + gamma* max(action_value_left, action_value_right)
        weight_update = alpha * (target - old_action_value) * construct_extended_state(old_state, old_action)
        # print(weights)
        weights -= weight_update
        cnt += 1

    print(cnt)
    if cnt > 195:
        print('Number of iterations', i)
        print('Best weights are', cnt)
        break
print(weights)