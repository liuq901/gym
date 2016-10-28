import numpy as np

class QLearning:
    def __init__(self, observation_cnt, action_cnt):
        self.observation_cnt = observation_cnt
        self.action_cnt = action_cnt
        self.q_value = np.zeros((observation_cnt, action_cnt), dtype = float)

    def argmax(self, observation):
        best = 0
        for i in xrange(self.action_cnt):
            if self.q_value[observation][i] > self.q_value[observation][best]:
                best = i
        return best

    def greedy(self, observation, epsilon):
        if np.random.random_sample() <= epsilon:
            return np.random.randint(self.action_cnt)
        else:
            return self.argmax(observation)

    def train(self, observation, new_observation, action, reward, alpha = 1.0, gamma = 0.25):
        tmp = self.q_value[new_observation][self.argmax(new_observation)]
        self.q_value[observation][action] += alpha * (reward + gamma * tmp - self.q_value[observation][action])
