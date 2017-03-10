import gym
from collections import namedtuple
import numpy as np

Sample = namedtuple('Sample', ['state', 'next_state', 'action', 'reward'])


class Sampler(object):

    @classmethod
    def create_from_gym_name(cls, gym_name):
        env = gym.make(gym_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        return cls(env, state_size, action_size)

    @classmethod
    def create_from_perso_env(cls, env):
        return cls(env, env.state_size, env.action_size)

    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

    def compute_samples(self, sess=None, runner=None, policy=None, no_episodes=0, max_length=0, std=0.05):

        samples = []
        for i_episode in range(no_episodes):
            state = self.env.reset()
            for t in range(max_length):

                # Pick the action by using network policy (or random)
                if sess is None:
                    action = self.env.action_space.sample()
                else:
                    action = runner.execute(sess, policy, {"state":[state]})[0]
                    action += np.random.normal(action, [std]*len(action), len(action))

                # Sample environment
                next_state, reward, done, info = self.env.step(action)

                # store samples
                one_sample = Sample(state=state, action=action, reward=reward, next_state=next_state)
                samples.append(one_sample)

                state = next_state

                if done:
                    break

        return samples

    def evaluate(self, sess, runner, policy, no_episodes, max_length, gamma, display=False):

        final_reward, t = 0, 0
        rewards = []

        for i_episode in range(no_episodes):
            state = self.env.reset()

            for t in range(max_length):

                action = runner.execute(sess, policy, {"state":[state]})[0]
                next_state, reward, done, info = self.env.step(action)

                if display:
                    print(action)
                    self.env.render()

                final_reward = reward + gamma*final_reward
                state = next_state

                if done:
                    break

            if display:
                # print("After " + str(t+1) + " steps, final reward: " + str(final_reward))
                self.env.render(close=True)

            rewards.append([final_reward, t+1])

        rewards = np.array(rewards)

        return (rewards[:,0].mean(), rewards[:,1].mean()), rewards
