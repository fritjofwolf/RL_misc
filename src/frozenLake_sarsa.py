import gym
import numpy as np

env = gym.make('FrozenLake-v0')
cnt_visited = np.zeros((16,4))

def play_game():
    global env
    global cnt_visited
    global q_function
    for _ in range(10000):
        state = env.reset()
        action = compute_new_action(state, np.random.randint(4))
        done = False
        while not done:
            new_state, reward, done, info = env.step(action)
            new_action = compute_new_action(new_state, action)
            ret = (reward+0.9*q_function[new_state,new_action] - q_function[state, action])
            q_function[state, action] += 0.1* ret
            cnt_visited[state, action] += 1
            state = new_state
            action = new_action

def compute_new_action(current_state, current_action):
    global cnt_visited
    global q_function
    action = 0
    epsilon = 0.1
    if np.random.rand() < epsilon / cnt_visited[current_state, current_action]:
        action = np.random.randint(0,4)
    else:
        action = np.argmax(q_function[current_state,:])
    return action

q_function = np.zeros((16,4))
play_game()
print('Q-Function ist:')
print(q_function)
print('Cnt of States visited is:')
print(cnt_visited)