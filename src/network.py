import tensorflow as tf
import numpy as np
import itertools
import math
from network_tools import *

#layer_size + [action_size]
def create_q_network(state, action, policy, layer_size, taylor_order, training_taylor):

    derivatives = []
    taylor_decomposition = []

    if taylor_order >= 0:
        with tf.variable_scope("zero_order"):
            with tf.variable_scope('V'):
                zero_order = create_nlp(state, layer_size + [1])
        taylor_decomposition += [zero_order]

        tf.summary.histogram('V0', zero_order)

    if taylor_order >= 1:
        action_size = int(action.get_shape()[1])
        with tf.variable_scope('first_order'):
            with tf.variable_scope('grad_q'):
                taylor_1_input = tf.concat([state, policy], axis=1)
                grad_q = create_nlp(taylor_1_input, layer_size+[action_size])
            diff = action - policy
            first_order = dot_product(grad_q, diff)
        taylor_decomposition += [first_order]
        derivatives += [grad_q]

        variable_summaries(grad_q, 'grad_Q',)
        tf.summary.histogram('taylor_1', first_order)


    if taylor_order >= 2:
        action_size = int(action.get_shape()[1])
        with tf.variable_scope('second_order'):
            taylor_2_input = tf.concat([state, policy], axis=1)

            # create lower triangular matrix
            diag_hessian_dim = math.pow(max(action_size-1, 0),2)+1
            diag_hessian = create_nlp(taylor_2_input, layer_size + [diag_hessian_dim])
            diag_hessian = tf.matrix_band_part(diag_hessian, -1, 0)

            hessian = tf.matmul(diag_hessian, diag_hessian, transpose_b=True)

            action_policy = tf.expand_dims(action - policy, axis=2)
            second_order = tf.matmul(action_policy, hessian, transpose_a=True) # TODO, check that the transpose is correct
            second_order = tf.matmul(second_order, action_policy)

        taylor_decomposition += [second_order]
        derivatives += [hessian]

    with tf.variable_scope('Q'):

        q_output = taylor_decomposition[0]
        for i, taylor in enumerate(taylor_decomposition[:]):
            propagate_gradient = tf.less_equal(i, training_taylor, name="propagate_taylor_"+str(i))
            q_output += tf.cond(propagate_gradient, lambda: taylor, lambda: tf.stop_gradient(taylor))

        q_output = tf.reshape(q_output, shape=[-1])

    return q_output, taylor_decomposition, derivatives


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
        self._is_training = tf.placeholder(tf.bool, name="is_training")

        self._taylor_training = tf.placeholder_with_default(taylor_order, shape=[], name='taylor_training')

        self.gamma = gamma
        self.alpha = alpha

        self.policy_pretrain_target = tf.placeholder(tf.float32, [None, action_size], name='policy_pretrain_target')

        #batch_size = tf.shape(self._state)[0]

        ####################
        # Create networks
        ####################

        # Policy network
        with tf.variable_scope('policy'):
            policy_layer_size = layer_size + [action_size]

            # Create a policy network that will be trained
            with tf.variable_scope('policy_network'):
                self.policy_output = create_policy_network(self._state, policy_layer_size)
                variable_summaries(self.policy_output)

            # Create a mirror policy to store the previous policy
            with tf.variable_scope('policy_mirror_network'):
                self.policy_mirror = create_policy_network(self._state, policy_layer_size)
                self.policy_mirror = tf.stop_gradient(self.policy_mirror)
                variable_summaries(self.policy_mirror)

        with tf.variable_scope('Q'):
            no_grad_policy = tf.stop_gradient(self.policy_output) # prevent to update the policy when training Q

            # create the Q network
            with tf.variable_scope("q_network"):
                self.q_output, self.taylor, derivatives = create_q_network(self._state, self._action, no_grad_policy, layer_size,
                                                                           taylor_order=taylor_order,
                                                                           training_taylor=self._taylor_training)

            # create the Q target network (for next state). Note that taylor expansion is limited to 0 order
            with tf.variable_scope("q_target_network"):
                self.q_next, _, _ = create_q_network(self._next_state, None, no_grad_policy, layer_size,
                                                     taylor_order=0,
                                                     training_taylor=0)
                self.q_next = tf.stop_gradient(self.q_next)

        ####################
        # Loss of networks
        ####################

        # Compute Q Loss
        with tf.variable_scope('q_loss'):
            with tf.variable_scope('q_target'):
                self.q_target = self._reward + self.gamma*self.q_next
                self.q_target = tf.reshape(self.q_target, shape=[-1])

            self.q_loss = tf.pow(self.q_output - self.q_target, 2)
            self.q_loss = tf.reduce_mean(self.q_loss)

            # self.q_loss_decay = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.name.startswith("Q/q_network")]
            # self.q_loss_decay = self.q_loss + lmbda * tf.add_n(self.q_loss_decay)

        # Compute policy Loss
        with tf.variable_scope('policy_loss'):
            with tf.variable_scope('policy_target'):
                grad_q = derivatives[0]
                # grad_q = tf.nn.l2_normalize(grad_q, dim=1)
                grad_q = tf.stop_gradient(grad_q)

                if taylor_order < 2:
                    self.policy_target = self.policy_mirror + self.alpha * grad_q  # TODO normalize by norm of grad
                else:
                    hessian = derivatives[1]
                    inv_hessian = tf.matrix_inverse(hessian)
                    self.policy_target = self.policy_mirror + tf.matmul(inv_hessian,tf.expand_dims(grad_q, axis=2))


            self.policy_loss = tf.pow(self.policy_output - self.policy_target, 2)
            self.policy_loss = tf.reduce_mean(self.policy_loss)

            # self.policy_loss_decay = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if v.name.startswith("policy/policy_network")]
            # self.policy_loss_decay = self.policy_loss + lmbda * tf.add_n(self.policy_loss_decay)


        with tf.variable_scope('pretrain_policy_loss'):
            self.pretrain_policy_loss = tf.pow(self.policy_output - self.policy_pretrain_target, 2)
            self.pretrain_policy_loss = tf.reduce_mean(self.pretrain_policy_loss)

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

        self.merged = tf.summary.merge_all()




