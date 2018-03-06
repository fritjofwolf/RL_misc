import gym
import numpy as np

env = gym.make('FrozenLake-v0')

def evaluate_states(states):
    global env
    for i in range(100000):
        #print('RESTART')
        state = env.reset()
        done = False
        while not done:
            #print(state)
            old_state = state
            state, reward, done, info = env.step(np.random.randint(0,4))
            ret = (reward + 0.9*states[state]-states[old_state])
            states[old_state] += 0.1*ret
    return states

states = np.zeros(16)
states = evaluate_states(states)
print(states)