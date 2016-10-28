import gym
import algorithm

def upload(path = 'record'):
    gym.upload(path, api_key = 'sk_62lx3mKQnuGRNtYcmsOjA')

def discrete_cnt(space):
    if type(space) == gym.spaces.discrete.Discrete:
        return space.n
    elif type(space) == gym.spaces.tuple_space.Tuple:
        res = 1
        for x in space.spaces:
            res *= discrete_cnt(x)
        return res
    else:
        raise TypeError('Only support discrete spaces')

def encode(space, action):
    if type(space) == gym.spaces.discrete.Discrete:
        assert action < space.n
        return action
    elif type(space) == gym.spaces.tuple_space.Tuple:
        res = []
        for x in space.spaces:
            length = discrete_cnt(x)
            res.append(action % length)
            action /= length
        return tuple(res)
    else:
        raise TypeError('Only support discrete spaces')

def monte_carlo(observation_cnt, action_cnt):
    return algorithm.MonteCarlo(observation_cnt, action_cnt)

def q_learning(observation_cnt, action_cnt):
    return algorithm.QLearning(observation_cnt, action_cnt)
