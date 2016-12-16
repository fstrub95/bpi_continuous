import gym
from collections import namedtuple
import numpy as np

Sample = namedtuple('Sample', ['state', 'next_state', 'action', 'reward'])

class GymWrapper(object):
    def __init__(self, gym_name):
        self.env = gym.make(gym_name)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]

    def compute_samples(self, sess=None, network=None, no_episodes=5, max_length=200):

        samples = []
        for i_episode in range(no_episodes):
            state = self.env.reset()
            for t in range(max_length):

                # Pick the action by using network policy (or random)
                if sess is None:
                    action = self.env.action_space.sample()
                else:
                    action = network.eval_next_action(sess, state)

                # Sample environment
                next_state, reward, done, info = self.env.step(action)

                # store samples
                one_sample = Sample(state=state, action=action, reward=reward, next_state=next_state)
                samples.append(one_sample)

                state = next_state

                if done:
                    break

        return samples

    def evaluate(self, sess, network, no_episodes=1, max_length=200, gamma=0.99, display=False):

        final_reward, t = 0, 0
        rewards = []

        for i_episode in range(no_episodes):
            state = self.env.reset()

            for t in range(max_length):

                action = network.eval_next_action(sess, state)
                #print(action)
                next_state, reward, done, info = self.env.step(action)

                if display:
                    self.env.render()

                final_reward = reward + gamma*final_reward
                state = next_state

                if done:
                    break

            if display:
                print("After " + str(t+1) +  " steps, final reward: " + str(final_reward))
                self.env.render(close=True)

            rewards.append([final_reward, t+1])

        rewards = np.array(rewards)

        return (rewards[:,0].mean(), rewards[:,1].mean()), rewards
