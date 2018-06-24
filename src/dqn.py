import numpy as np
from collections import deque
import random

class DQN():

    def __init__(self, mlp, env, gamma, eps, eps_discount, replay_mem_size, batch_size):
        self.mlp = mlp
        self.env = env
        self.gamma = gamma
        self.eps = eps
        self. eps_discount = eps_discount
        self.replay_mem = deque(maxlen = replay_mem_size)
        self.batch_size = batch_size

    def train(self, iterations):
        self.do_initial_fit()
        for i in range(iterations):
            self.eps *= self.eps_discount
            state = self.env.reset()
            action = self.compute_new_action_eps_greedy(state)
            done = False
            old_qvalue = self.compute_qfunction(state)
            cnt = 0
            while not done:
                new_state, reward, done, info = self.env.step(action)
                if done and cnt > 100:
                    reward = 100-cnt
                new_action = self.compute_new_action_eps_greedy(new_state)
                new_qvalue = self.compute_qfunction(new_state)
                target = self.compute_target(old_qvalue, new_qvalue, reward, action, new_action)
                self.replay_mem.append((state, target))
                x, y = self.get_current_training_batch()
                self.mlp.partial_fit(x, y)
                state, action = new_state, new_action
                old_qvalue = new_qvalue
                cnt += 1
            print(i, cnt, self.eps)


    def do_initial_fit(self):
        state = self.env.reset()
        self.mlp.partial_fit([state], [[0]*self.env.action_space.n])
    
    def get_current_training_batch(self):
        x = []
        y = []
        for i in range(self.batch_size):
            (a,b) = self.replay_mem[random.randrange(len(self.replay_mem))]
            x.append(a)
            y.append(b)
        return x, y

    def compute_target(self, old_qvalue, new_qvalue, reward, action, new_action):
        ret = (reward+self.gamma*new_qvalue[new_action] - old_qvalue[action])
        target = old_qvalue.copy()
        target[action] = ret
        return target

    def compute_new_action_greedy(self, state):
        action_values = self.mlp.predict([state])
        action = np.argmax(action_values)
        return action


    def compute_new_action_eps_greedy(self, state):
        if np.random.rand() < self.eps:
            action = np.random.randint(self.env.action_space.n)
        else:
            action_values = self.mlp.predict([state])
            action = np.argmax(action_values)
        return action


    def compute_qfunction(self, state):
        return self.mlp.predict([state])[0]

    def evaluate_qfunction(self):
        sum_return = 0
        iterations = 100
        for _ in range(iterations):
            episode_reward = 0
            state = self.env.reset()
            action = self.compute_new_action_greedy(state)
            done = False
            while not done:
                #env.render()
                state, reward, done, info = self.env.step(action)
                action = self.compute_new_action_greedy(state)
                sum_return += reward
                episode_reward += reward
        print('Average reward per episode is:', sum_return/iterations)