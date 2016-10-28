import gym
import lib

env = gym.make('Taxi-v1')
env.monitor.start('record', force = 'True')

observation_cnt = lib.discrete_cnt(env.observation_space)
action_cnt = lib.discrete_cnt(env.action_space)
trainer = lib.q_learning(observation_cnt, action_cnt)

epsilon = 0.5
decay = 0.99
limit = 1000
for k in xrange(1, limit + 1):
    observation = env.reset()

    while True:
        if k == limit:
            env.render()

        action = trainer.greedy(observation, epsilon)
        true_action = lib.encode(env.action_space, action)
        new_observation, reward, done, info = env.step(true_action)

        trainer.train(observation, new_observation, action, reward)
        observation = new_observation

        if done:
            break

    epsilon *= decay

env.monitor.close()
lib.upload()
