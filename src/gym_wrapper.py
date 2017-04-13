import gym
from gym import wrappers
from collections import namedtuple
import numpy as np

import itertools

Sample = namedtuple('Sample', ['state', 'next_state', 'action', 'reward'])

def create_cholesky_noise(samples):

    # linearize the action
    actions = []
    for trajectory in samples:
        for step in trajectory:
            actions.append(step.action)
    actions = np.array(actions).transpose()

    # compute cholesky matrix
    cov = np.cov(actions)
    cholesky = np.linalg.cholesky(cov)
    
    def cholesky_noise(action):
        epsilon = np.random.normal(size=len(action))
        return np.dot(cholesky, epsilon)
    
    return cholesky_noise


def create_normal_noise(std):
    def normal_noise(action):
        return np.random.normal(action, [std]*len(action), len(action))
    return normal_noise





class Sampler(object):
    @classmethod
    def create_from_gym_name(cls, gym_name, save_path="/tmp/gym-results"):
        env = gym.make(gym_name)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]

        #save_path = save_path + "/" + gym_name
        #env = wrappers.Monitor(env, save_path,force=True)

        return cls(env, state_size, action_size, save_path=save_path)

    @classmethod
    def create_from_perso_env(cls, env):
        return cls(env, env.state_size, env.action_size, "")

    def __init__(self, env, state_size, action_size, save_path):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.save_path = save_path

    def compute_samples(self, sess=None, runner=None, policy=None, no_episodes=0, max_length=0, noise_fct=create_normal_noise(0.05), flatten=True):

        samples = []
        for i_episode in range(no_episodes):
            state = self.env.reset()

            one_trajectory = []
            for t in range(max_length):

                # Pick the action by using network policy (or random)
                if sess is None:
                    action = policy.evaluate(state)
                else:
                    action = runner.execute(sess, policy, {"state":[state]})[0]
                    action += noise_fct(action)

                # Sample environment
                next_state, reward, done, info = self.env.step(action)

                # store samples
                one_sample = Sample(state=state, action=action, reward=reward, next_state=next_state)
                one_trajectory.append(one_sample)

                state = next_state

                if done:
                    break
            samples.append(one_trajectory)

        if flatten:
            samples = list(itertools.chain(*samples))

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
                    # print(action)
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

        return (rewards[:,0].mean(), rewards[:,0].std(), rewards[:,1].mean()), rewards

    #def commit(self):
        #if len(self.save_path):
            #gym.upload('self.save_path', api_key='sk_EPE5bvpoQPSdM8I42EgHVw')