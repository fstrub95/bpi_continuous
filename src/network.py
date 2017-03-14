import tensorflow as tf
import numpy as np
import itertools
from network_tools import *

#layer_size + [action_size]
def create_q_network(state, action, policy, layer_size, taylor_order):

    derivatives = []
    taylor_decomposition = []

    if taylor_order >= 0:
        with tf.variable_scope("zero_order"):
            with tf.variable_scope('V'):
                zero_order = create_nlp(state, layer_size)
        taylor_decomposition += [zero_order]

    if taylor_order >= 1:
        with tf.variable_scope('first_order'):
            with tf.variable_scope('grad_q'):
                grad_q = create_nlp(state, layer_size)
            diff = action - policy
            first_order = dot_product(grad_q, diff)
        taylor_decomposition += [first_order]
        derivatives += [grad_q]


    if taylor_order >= 2:
        with tf.variable_scope('first_order'):
            pass
            # Second order: Hessian
            # TODO
        taylor_decomposition += []
        derivatives += []



    with tf.variable_scope('Q'):
        q_output = tf.add_n(taylor_decomposition)
        q_output = tf.reshape(q_output, shape=[-1])

    return q_output, derivatives


def create_policy_network(state, layer_size):
    policy = create_nlp(state, layer_size)

    return policy



class Network(object):

    def __init__(self, state_size, action_size, layer_size, gamma, tau, alpha, lmbda, taylor_order):

        ####################
        # Input parameters
        ####################

        self.state_size = state_size
        self.action_size = action_size

        self._state = tf.placeholder(tf.float32, [None, state_size], name='state')
        self._action = tf.placeholder(tf.float32, [None, action_size], name='action')
        self._next_state = tf.placeholder(tf.float32, [None, state_size], name='next_state')
        self._reward = tf.placeholder(tf.float32, [None], name='reward')

        self._tau = tf.placeholder_with_default(tau, shape=[], name='tau')
        #self._is_training = tf.placeholder(tf.bool, name="is_training")

        self.gamma = gamma
        self.alpha = alpha

        ####################
        # Create networks
        ####################

        # Policy network
        with tf.variable_scope('policy'):
            policy_layer_size = layer_size + [action_size]

            # Create a policy network that will be trained
            with tf.variable_scope('policy_network'):
                self.policy_output = create_policy_network(self._state, policy_layer_size)

            # Create a mirror policy to store the previous policy
            with tf.variable_scope('policy_mirror_network'):
                self.policy_mirror = create_policy_network(self._state, policy_layer_size)
                self.policy_mirror = tf.stop_gradient(self.policy_mirror)

        with tf.variable_scope('Q'):
            q_layer_size =  layer_size + [action_size]
            no_grad_policy = tf.stop_gradient(self.policy_output) # prevent to update the policy when training Q

            # create the Q network
            with tf.variable_scope("q_network"):
                self.q_output, derivatives = create_q_network(self._state, self._action, no_grad_policy, q_layer_size, taylor_order=taylor_order)

            # create the Q target network (for next state). Note that taylor expansion is limited to 0 order
            with tf.variable_scope("q_target_network"):
                self.q_next, _ = create_q_network(self._next_state, None, no_grad_policy, q_layer_size, taylor_order=0)
                self.q_next = tf.stop_gradient(self.q_next)

        ####################
        # Loss of networks
        ####################

        # Compute Q Loss
        with tf.variable_scope('q_loss'):
            with tf.variable_scope('q_target'):
                self.q_target = self._reward + self.gamma*self.q_next
                self.q_target = tf.reshape(self.q_target, shape=[-1])

            self.q_loss = tf.nn.l2_loss(self.q_output - self.q_target)

            self.q_loss_decay = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.name.startswith("Q/q_network")]
            self.q_loss_decay = self.q_loss + lmbda * tf.add_n(self.q_loss_decay)

        # Compute policy Loss
        with tf.variable_scope('policy_loss'):
            with tf.variable_scope('policy_target'):
                grad_q = derivatives[0]
                grad_q = tf.stop_gradient(grad_q)

                self.policy_target = self.policy_mirror + self.alpha * grad_q  # TODO normalize by norm of grad

            self.policy_loss = tf.nn.l2_loss(self.policy_output - self.policy_target)

            self.policy_loss_decay = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.name.startswith("policy/policy_network")]
            self.policy_loss_decay = self.policy_loss + lmbda * tf.add_n(self.policy_loss_decay)


        ####################
        # Update networks
        ####################

        # Get policy network variables
        with tf.variable_scope('policy_update'):
            policy_cur_variables = [v for v in tf.trainable_variables() if v.name.startswith("policy/policy_network")]
            policy_mir_variables = [v for v in tf.trainable_variables() if v.name.startswith("policy/policy_mirror_network")]

            # Copy variable values from current network to the old one
            self.policy_parameters = []
            for cur, target in zip(policy_cur_variables, policy_mir_variables):
                self.policy_parameters += [target.assign(self._tau*cur + (1-self._tau)*target)]
                self.policy_update = tf.group(*self.policy_parameters)

        # Get policy network variables
        with tf.variable_scope('q_update'):

            q_mir_variables = [v for v in tf.trainable_variables() if v.name.startswith("Q/q_target_network")]
            q_mir_var_names = [v.name.replace("Q/q_target_network", "") for v in tf.trainable_variables() if v.name.startswith("Q/q_target_network")]

            # Only update the variable that are used by the target network
            q_cur_variables = [ v for v in tf.trainable_variables() if v.name.replace("Q/q_network", "") in q_mir_var_names]

            # Copy variable values from current network to the old one
            self.q_parameters = []
            for cur, target in zip(q_cur_variables, q_mir_variables):
                self.q_parameters += [target.assign(self._tau*cur + (1-self._tau)*target)]
            self.q_update = tf.group(*self.q_parameters)

        self.update_networks = tf.group(self.q_update, self.policy_update)


class Runner(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def eval_loss(self, sess, iterator, loss):
        return self.__execute__(sess, iterator, loss, False)

    def train(self, sess, iterator, optimizer):
        return self.__execute__(sess, iterator, optimizer, True)

    def __execute__(self, sess, iterator, output, is_training):

        # Compute the number of required samples
        n_iter = int(iterator.no_samples / self.batch_size) + 1

        loss = 0
        for i in range(n_iter):

            # Creating the feed_dict
            batch = iterator.next_batch(self.batch_size, shuffle=is_training)
            #batch["is_training"] = is_training
            tflearn.config.is_training(is_training=is_training, session=sess)

            # compute loss
            res = self.execute(sess, output, batch)
            if res is not None:
                loss += res
        loss /= n_iter

        # By default, there is no training
        tflearn.config.is_training(is_training=False, session=sess)

        return loss

    def execute(self, sess, output, sample):
        return sess.run(output, feed_dict={ key+":0" : value for key, value in sample.items()})



