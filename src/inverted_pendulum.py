# -*- coding: utf-8 -*-
import math
import numpy as np
import random as rd


class InvertedPendulumWrapper(object):
    def __init__(self, _m=2, _M=8, _l=0.5, _deltaT=0.1):
        self.pendulum = InvertedPendulum(_m=_m, _M=_M, _l=_l, _deltaT=_deltaT)
        self.state_size = 2
        self.action_size = 1

        self.state = []
        self.reset()

        class ActionSpace(object):
            def sample(self):
                return [((rd.random() * 2) - 1)]
        self.action_space = ActionSpace()

    def render(self, close=False):
        pass

    def reset(self):
        self.state = [0.01 * rd.random(), 0.01 * rd.random()]
        return self.state

    def step(self, action):
        reward, self.state = self.pendulum.NextStep(self.state, action)
        done = (reward != 0)
        info = None
        return self.state, reward, done, info


# creation of the pendulum class
class InvertedPendulum:

    def __init__(self, _m=2, _M=8, _l=0.5, _deltaT=0.1, _Continuous=True, _Na=3):

        # definition of the attributes
        self.m = float(_m)
        self.M = float(_M)
        self.l = float(_l)
        self.deltaT = float(_deltaT)
        self.alpha = 1.0/(self.m + self.M)
        self.continuous = _Continuous
        self.Na = _Na
        self.g = 9.8

    # definition of the dynamics
    def NextStep(self, s, a):

        theta0 = s[0]
        theta1 = s[1]

        if self.continuous:
            u = 100*a[0]
        else:
            # discrete actions -50;0;+50
            u = 50.0 * ((2.0 * a / (self.Na - 1)) - 1)

        # dynamics definition
        theta2 = (self.g * math.sin(theta0) - self.alpha * self.m * self.l * theta1**2 * math.sin(2*theta0) * 0.5- self.alpha * math.cos(theta0) * u)/((4 * self.l / 3) - self.alpha * self.m * self.l * math.cos(theta0)**2)
        # integration
        theta1Next = theta1 + self.deltaT * theta2
        # integration
        theta0Next = theta0 + self.deltaT * (theta1 + theta1Next)/2
        # definition of the reward
        reward = -1 if math.fabs(theta0Next) > (math.pi / 2) else 0

        return reward, [theta0Next, theta1Next]

    # generation of the data
    def GenerateData(self, N):
        data = list()
        Rewards = list()
        for _ in range(N):
            sStart = [0.01 * rd.random(), 0.01 * rd.random()]
            s = sStart
            reward = 0
            while reward == 0:
                if self.continuous:
                    a=np.zeros(1)
                    a[0] = ((rd.random() * 2) - 1)
                else:
                    a = rd.sample(range(0, self.Na), 1)[0]

                reward, sNext = self.NextStep(s, a)

                data.append([s, a, sNext])

                Rewards.append(reward)
                s = sNext
        rewards=np.zeros(len(Rewards))
        for i in range(0,len(Rewards)):
            rewards[i]=Rewards[i]
        return data, rewards

    # get the next action for the discrete pendulum
    def GetNextAction(self, w, s, mu):

        Phi = np.zeros((self.Na, self.Na * (len(mu) + 1)))
        vector = np.zeros((len(mu) + 1))

        vector[0] = 1

        for i in range(len(mu)):
            vector[i + 1] = np.exp(-(np.linalg.norm(s - mu[i], 2) ** 2) / 2)

        for j in range(self.Na):
            Phi[j, j * (len(mu) + 1): (j + 1) * (len(mu) + 1)] = vector

        a = np.argmax(np.dot(Phi, w))

        return a


def BaseFromSample(s,mu):
    ''' this function gives you the features for a given state s where the centers of the radial basis are stored in the vector mu '''
    PhiSample = np.empty((len(mu) + 1))
    PhiSample[0] = 1
    for p in range(len(mu)):
        PhiSample[p + 1] = np.exp(-(np.linalg.norm(s - mu[p], 2) ** 2) / 2)
    return PhiSample

def BaseFromData(Data, mu):

    ''' this function gives you the features for the states in Data'''
    PhiSampleList = list()
    PhiSampleNextList = list()

    for i, d in enumerate(Data):
        PhiSample = np.empty((len(mu) + 1))
        PhiSampleNext = np.empty((len(mu) + 1))

        PhiSample[0] = 1
        PhiSampleNext[0] = 1

        for p in range(len(mu)):
            PhiSample[p + 1] = np.exp(-(np.linalg.norm(d[0] - mu[p], 2) ** 2) / 2)
            PhiSampleNext[p + 1] = np.exp(-(np.linalg.norm(d[2] - mu[p], 2) ** 2) / 2)

        PhiSampleList.append(PhiSample)
        PhiSampleNextList.append(PhiSampleNext)

    return PhiSampleList, PhiSampleNextList


def BaseDiscreteFromData(Data, mu, Na):

    PhiSampleList = np.zeros((len(Data), Na * (len(mu) + 1)))
    PhiSampleNextList = np.zeros((len(Data) * Na, Na * (len(mu) + 1)))

    vector = np.zeros((len(mu) + 1))

    for k, d in enumerate(Data):
        # placer Vecteur dans phiSample en fct de k
        vector[0] = 1
        for p in range(len(mu)):
            vector[p + 1] = np.exp(-(np.linalg.norm(d[0] - mu[p], 2) ** 2) / 2)
        PhiSampleList[k, int(d[1]) * (len(mu) + 1): (int(d[1]) + 1) * (len(mu) + 1)] = vector

    for k, d in enumerate(Data):
        # placer Vecteur dans phiSampleNext en fct de k
        vector[0] = 1
        for p in range(len(mu)):
            vector[p + 1] = np.exp(-(np.linalg.norm(d[2] - mu[p], 2) ** 2) / 2)
        for j in range(Na):
            PhiSampleNextList[k + j * len(Data), j * (len(mu) + 1): (j + 1) * (len(mu) + 1)] = vector

    return PhiSampleList, PhiSampleNextList


def CreateCenter(X, Y):
    ''' This function create a vector of centers for your radial basis from a list of x and list of y'''
    center = np.zeros((len(X) * len(Y), 2))

    for i in range(len(X)):
        for j in range(len(Y)):
            center[i + j * len(X), :] = [X[i], Y[j]]

    return center


def CreateMuDico(Data, e):
    NFeature = len(Data[0][0])
    realData = [d[0] for d in Data]

    DataDico = np.zeros((1, NFeature), dtype=np.double)
    DataDico[0, :] = np.array(realData[0])

    K_t = [np.exp(-(np.linalg.norm(DataDico[0] - DataDico[0], 2) ** 2) / 2)]

    K_t = np.reshape(K_t, (1, 1))
    Kinv_t = 1 / K_t

    for iS, sample in enumerate(realData):
        sample = np.array(sample)
        k_t = np.empty((len(DataDico), 1), dtype=np.double)

        for i in range(len(DataDico)):
            k_t[i] = np.exp(-(np.linalg.norm(DataDico[i] - sample, 2) ** 2) / 2)

        at = np.dot(Kinv_t, k_t)

        ktt = np.exp(-(np.linalg.norm(sample - sample, 2) ** 2) / 2)

        dt = ktt - np.dot(np.transpose(k_t), at)
        if dt > e:

            DataDico = np.vstack((DataDico, sample))

            K_t = np.vstack((K_t, np.transpose(k_t)))

            new_kt = np.zeros((len(k_t) + 1, 1), dtype=np.double)
            new_kt[0:len(k_t)] = k_t
            new_kt[len(new_kt) - 1] = ktt
            K_t = np.hstack((K_t, new_kt))

            Kinv_t = np.linalg.inv(K_t)

    return DataDico

if __name__ == '__main__':
    PendulumDiscrete = InvertedPendulum(_Continuous=True)
    data,rewards= PendulumDiscrete.GenerateData(10)
    print(rewards)