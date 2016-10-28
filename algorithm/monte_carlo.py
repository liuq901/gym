import numpy as np

class MonteCarlo:
    def __init__(self, observation_cnt, action_cnt):
        self.observation_cnt = observation_cnt
        self.action_cnt = action_cnt
        self.cnt = np.zeros((observation_cnt, action_cnt), dtype = int)
        self.q_value = np.zeros((observation_cnt, action_cnt), dtype = float)

    def greedy(self, observation, epsilon):
        if np.random.random_sample() <= epsilon:
            return np.random.randint(self.action_cnt)
        else:
            best = 0
            for i in xrange(self.action_cnt):
                if self.q_value[observation][i] > self.q_value[observation][best]:
                    best = i
            return best

    def train(self, observations, actions, rewards, gamma = 1):
        assert len(observations) == len(actions) == len(rewards)
        length = len(observations)
        returns = [0.0] * length
        returns[length - 1] = rewards[length - 1]
        for i in xrange(length - 2, -1, -1):
            returns[i] = returns[i + 1] * gamma + rewards[i]
        for i in xrange(0, length):
            observation = observations[i]
            action = actions[i]
            self.cnt[observation][action] += 1
            self.q_value[observation][action] += (returns[i] - self.q_value[observation][action]) / self.cnt[observation][action]
