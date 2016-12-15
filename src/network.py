import tensorflow as tf
import numpy as np

from network_tools import *
from iterator import *


class Network(object):

    def __init__(self, state_size, action_size, layer_size):

        self.state_size = state_size
        self.action_size = action_size

        self.state = tf.placeholder(tf.float32, [None, state_size], name='state')
        self.action = tf.placeholder(tf.float32, [None, action_size], name='action')
        self.next_state = tf.placeholder(tf.float32, [None, state_size], name='next_state')
        self.reward = tf.placeholder(tf.float32, [None], name='reward')

        self.gamma = 0.99
        self.alpha = 1

        # Zero order: Value function
        with tf.name_scope('zero_order'):
            value_fct_layers = create_layers([state_size] + layer_size + [1])
            with tf.name_scope('value_state'):
                zero_order = build_multilayer_perceptron(self.state, value_fct_layers)
            with tf.name_scope('value_next_state'):
                zero_order_next = build_multilayer_perceptron(self.next_state, value_fct_layers)

        # First order: Gradient
        with tf.name_scope('gradient'):
            grad = build_multilayer_perceptron(self.state, create_layers([state_size] + layer_size + [action_size]))

        # Second order: Hessian
        # TODO

        # Policy network
        with tf.name_scope('policy'):
            policy_layer_size = [state_size] + layer_size + [action_size]
            with tf.name_scope('current'):
                self.policy = build_multilayer_perceptron(self.state, create_layers(policy_layer_size))
            with tf.name_scope('old'): # Create a mirror policy that will be used
                prev_policy = build_multilayer_perceptron(self.state, create_layers(policy_layer_size))
                prev_policy = tf.stop_gradient(prev_policy)

        # Compute the taylor first order
        with tf.name_scope('first_order'):
            diff = self.action - self.policy
            diff = tf.stop_gradient(diff) # Prevent the policy network to be updated with q_loss
            first_order = dot_product(grad, diff)

        # Compute q-network loss
        with tf.name_scope('q_loss'):
            self.q_output = zero_order + first_order
            self.q_target = self.reward + self.gamma*zero_order_next
            self.q_loss = tf.nn.l2_loss(self.q_output - self.q_target)
        self.q_optimizer = tf.train.AdamOptimizer().minimize(self.q_loss)

        # Compute policy-network loss
        with tf.name_scope('policy_loss'):
            self.pi_output = self.policy
            self.pi_target = prev_policy + self.alpha*tf.stop_gradient(grad) #TODO normalize by norm of grad
            self.pi_loss = tf.nn.l2_loss(self.pi_output - self.pi_target)
        self.pi_optimizer = tf.train.AdamOptimizer().minimize(self.pi_loss)


    def execute_q(self, sess, iterator, mini_batch=20, is_training=True):

        target = [self.q_loss]
        if is_training:
            target.append(self.q_optimizer)

        return self.__execute(sess, iterator, mini_batch, target)


    def execute_policy(self, sess, iterator, mini_batch=20, is_training=True):

        # Get policy network variables
        policy_cur_variables = [v for v in tf.trainable_variables() if v.name.startswith("policy/current")]
        policy_old_variables = [v for v in tf.trainable_variables() if v.name.startswith("policy/old")]

        # TODO -> Fail! either bufferize policy or manage to copy-past weigth
        # Copy variable values from current network to the old one
        for cur, old in zip(policy_cur_variables, policy_old_variables):
            old.assign(cur)

        # define training/eval target
        target = [self.pi_loss]
        if is_training:
            target.append(self.pi_optimizer)

        return self.__execute(sess, iterator, mini_batch, target)


    def __execute(self, sess, iterator, mini_batch, target):

        # Compute the number of required samples
        n_iter = int(iterator.NoSamples()/mini_batch) + 1

        loss = 0
        for i in range(n_iter):

            # Creating the mini-batch
            batch = iterator.NextBatch(mini_batch)

            # Running one step of the optimization method on the mini-batch
            res = sess.run(target, feed_dict={
                    self.state: batch.state,
                    self.next_state: batch.next_state,
                    self.action: batch.action,
                    self.reward: batch.reward})

            loss += res[0]
        loss /= n_iter

        return loss


    def eval_next_action(self, sess, state):

        # use the correct shape for the action
        s = np.zeros((1,self.state_size))
        s[0,:] = state

        return sess.run(self.policy, feed_dict={
            self.state: s,
            self.next_state: np.zeros((1,self.state_size)),
            self.action: np.zeros((1,self.action_size)),
            self.reward: np.zeros(1)})[0] # Return one single action


import gym
from collections import namedtuple

Sample = namedtuple('Sample', ['state', 'next_state', 'action', 'reward'])
env = gym.make('MountainCarContinuous-v0')

state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

network = Network(state_size,action_size, layer_size=[20, 20])



def compute_samples(sess, network, env, no_episodes=1, max_length=200):

    samples = []
    for i_episode in range(no_episodes):
        state = env.reset()
        for t in range(max_length):

            # sample the environment by using network policy
            action = network.eval_next_action(sess, state)
            next_state, reward, done, info = env.step(action)

            one_sample = Sample(state=state, action=action, reward=reward, next_state=next_state)
            samples.append(one_sample)

            if done:
                break

    return samples



# Define session
with tf.Session() as sess:
    writer = tf.train.SummaryWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
    sess.run(tf.initialize_all_variables())

    samples = compute_samples(sess, network, env, no_episodes=20, max_length=50)

    dataset_iterator = Dataset(samples)

    print(network.execute_q(sess, iterator=dataset_iterator, mini_batch=2, is_training=False))
    network.execute_q(sess, iterator=dataset_iterator, mini_batch=2, is_training=True)


    network.execute_policy(sess, iterator=dataset_iterator, mini_batch=2, is_training=True)
    print(network.execute_policy(sess, iterator=dataset_iterator, mini_batch=2, is_training=False))

    network.execute_q(sess, iterator=dataset_iterator, mini_batch=2, is_training=True)
    print(network.execute_q(sess, iterator=dataset_iterator, mini_batch=2, is_training=False))


    network.execute_policy(sess, iterator=dataset_iterator, mini_batch=2, is_training=True)
    print(network.execute_policy(sess, iterator=dataset_iterator, mini_batch=2, is_training=False))


    for i_episode in range(5):
        observation = env.reset()
        for t in range(200):
            env.render()
            #print(observation)
            action = network.eval_next_action(sess, observation)
            observation, reward, done, info = env.step(action)
            print(action)
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                break