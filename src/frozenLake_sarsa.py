import gym
import numpy as np

env = gym.make('FrozenLake-v0')
cnt_visited = np.zeros((16,4))

def play_game(q_function, cnt_visited):
    global env
    sum_return = 0
    for _ in range(1000):
        state = env.reset()
        action = compute_new_action_eps_greedy(state, np.random.randint(4))
        done = False
        while not done:
            new_state, reward, done, info = env.step(action)
            if done and reward == 0:
               reward = -1
            new_action = compute_new_action_eps_greedy(new_state, action)
            ret = (reward+0.9*q_function[new_state,new_action] - q_function[state, action])
            q_function[state, action] += 0.1* ret
            cnt_visited[state, action] += 1
            state = new_state
            action = new_action
            if reward == 1:
                sum_return += reward
    return q_function, cnt_visited

def compute_new_action_eps_greedy(current_state, current_action):
    global cnt_visited
    global q_function
    action = 0
    epsilon = 0.1
#    if np.random.rand() < epsilon / (cnt_visited[current_state, current_action]+1):
    if np.random.rand() < epsilon:
        action = np.random.randint(0,4)
    else:
        action = np.argmax(q_function[current_state,:])
    return action

def compute_new_action_random(current_state, current_action):
    return np.random.randint(0,4)

def evaluate_qfunction(q_function):
    global env
    sum_return = 0
    for _ in range(1000):
        state = env.reset()
        action = compute_new_action_eps_greedy(state, np.random.randint(4))
        done = False
        while not done:
            state, reward, done, info = env.step(action)
            action = compute_new_action_eps_greedy(state, action)
            sum_return += reward
    print('Average reward per episode is:',sum_return/1000)


q_function = np.zeros((16,4))
cnt_visited = np.zeros((16,4))
q_function, cnt_visited = play_game(q_function, cnt_visited)
print('Q-Function ist:')
print(q_function)
print('Cnt of States visited is:')
print(cnt_visited)
evaluate_qfunction(q_function)