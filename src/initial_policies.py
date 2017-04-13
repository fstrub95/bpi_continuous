import numpy as np

class initial_pendulum_policy(object):

    def evaluate(self, state):
        return [np.random.uniform(-2, 2)]

class initial_moutain_car_policy(object):

    def __init__(self, mean=0.2, std=0.2):
        self.mean = mean
        self.std = std

    def evaluate(self, state):

        position = state[0]
        velocity = state[1]

        if velocity > 0:
            action = np.random.normal()*self.std+self.mean
        else:
            action =  np.random.normal()*self.std-self.mean

        return [action]