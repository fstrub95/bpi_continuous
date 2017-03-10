import numpy as np

try:
    from itertools import izip
except ImportError:  #python3.x
    izip = zip

from collections import namedtuple


MiniBatch = namedtuple('MiniBatch', ['state', 'next_state', 'action', 'reward'])

class Dataset(object):
    def __init__(self, samples):

        self.samples = samples

        self.state_size  = len(samples[0].state)
        self.action_size = len(samples[0].action)
        self.no_samples = len(self.samples)

        self.state=np.zeros((self.no_samples, self.state_size))
        self.next_state=np.zeros((self.no_samples, self.state_size))
        self.action=np.zeros((self.no_samples, self.action_size))
        self.reward=np.zeros(self.no_samples)

        for i, sample in enumerate(self.samples):
            self.state[i] = sample.state
            self.next_state[i] = sample.next_state
            self.action[i] = sample.action
            self.reward[i] = sample.reward


        self.epoch_completed = 0
        self.index_epoch_completed = 0


    def next_batch(self, size_batch, shuffle = True):

        # return a minibatch of size sizeBatch
        start = self.index_epoch_completed
        self.index_epoch_completed = self.index_epoch_completed + size_batch

        #when all the samples are used, restart and shuffle
        if self.index_epoch_completed > self.no_samples:

            self.epoch_completed +=1

            #reset indices
            start = 0
            self.index_epoch_completed = size_batch
            assert size_batch <= self.no_samples

            if shuffle:

                #inplace permutation
                permute = np.arange(self.no_samples)
                np.random.shuffle(permute)

                #shuffle data
                self.state  = self.state[permute]
                self.action  = self.action[permute]
                self.next_state = self.next_state[permute]
                self.reward = self.reward[permute]


        end = self.index_epoch_completed

        return {
            "state":self.state[start:end],
            "next_state":self.next_state[start:end],
            "action":self.action[start:end],
            "reward":self.reward[start:end]
            }




