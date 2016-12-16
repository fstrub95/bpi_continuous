import tensorflow as tf
import numpy as np

from network_tools import *



class Network(object):

    def __init__(self, state_size, action_size, layer_size, gamma=0.99, alpha=1, q_lrt=0.001, pi_lrt=0.0001):

        self.state_size = state_size
        self.action_size = action_size

        self.state = tf.placeholder(tf.float32, [None, state_size], name='state')
        self.action = tf.placeholder(tf.float32, [None, action_size], name='action')
        self.next_state = tf.placeholder(tf.float32, [None, state_size], name='next_state')
        self.reward = tf.placeholder(tf.float32, [None], name='reward')

        self.is_training  = tf.placeholder(tf.bool, name="is_training")

        self.gamma = gamma
        self.alpha = alpha

        # Zero order: Value function
        with tf.variable_scope("zero_order") as zero_scope:
            # Create value zero order taylor network, (aka value fct network)
            with tf.name_scope('state'):
                zero_order = create_nlp(self.state, layer_size + [action_size], use_scope=True, is_training=self.is_training)

            # Reuse the network to compute the value fct on next_state
            with tf.name_scope('next_state'):
                zero_scope.reuse_variables()
                zero_order_next = create_nlp(self.next_state, layer_size + [action_size], use_scope=True, is_training=self.is_training)


        # First order: Gradient
        with tf.name_scope('q_gradient'):
            grad = create_nlp(self.state, layer_size + [action_size], is_training=self.is_training)


        # Second order: Hessian
        # TODO


        # Policy network
        with tf.name_scope('policy'):
            policy_layer_size = layer_size + [action_size]

            # Create a policy network that will be trained
            with tf.name_scope('current'):
                self.policy = create_nlp(self.state, policy_layer_size, is_training=self.is_training)

            # Create a mirror policy to store the previous policy
            with tf.name_scope('mirror'):
                mirror_policy = create_nlp(self.state, policy_layer_size, is_training=self.is_training)
                mirror_policy = tf.stop_gradient(mirror_policy)


        # Compute the taylor first order
        with tf.name_scope('first_order'):
            diff = self.action - self.policy
            diff = tf.stop_gradient(diff) # Prevent the policy network to be updated with q_loss
            first_order = dot_product(grad, diff)


        # Compute q-network loss
        with tf.name_scope('q_output'):
            self.q_output = zero_order + first_order

        with tf.name_scope('q_target'):
            self.q_target = self.reward + self.gamma*zero_order_next

        with tf.name_scope('q_loss'):
            self.q_loss = tf.nn.l2_loss(self.q_output - self.q_target)
        self.q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lrt).minimize(self.q_loss)


        # Compute policy-network loss
        with tf.name_scope('policy_output'):
            self.pi_output = self.policy

        with tf.name_scope('policy_target'):
            self.pi_target = mirror_policy + self.alpha*tf.stop_gradient(grad) #TODO normalize by norm of grad

        with tf.name_scope('policy_loss'):
            self.pi_loss = tf.nn.l2_loss(self.pi_output - self.pi_target)
        self.pi_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lrt).minimize(self.pi_loss)


    def execute_q(self, sess, iterator, mini_batch=20, is_training=True):

        # define training/eval network output
        target = [self.q_loss]
        if is_training:
            target.append(self.q_optimizer)

        return self.__execute(sess, iterator, mini_batch, target, is_training)


    def execute_policy(self, sess, iterator, mini_batch=20, is_training=True):

        # Get policy network variables
        policy_cur_variables = [v for v in tf.trainable_variables() if v.name.startswith("policy/current")]
        policy_mir_variables = [v for v in tf.trainable_variables() if v.name.startswith("policy/mirror")]

        # Copy variable values from current network to the old one
        for cur, old in zip(policy_cur_variables, policy_mir_variables):
            sess.run(old.assign(cur))

        # define training/eval network output
        target = [self.pi_loss]
        if is_training:
            target.append(self.pi_optimizer)

        return self.__execute(sess, iterator, mini_batch, target, is_training)


    def __execute(self, sess, iterator, mini_batch, target, is_training):

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
                    self.reward: batch.reward,
                    self.is_training: is_training})

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
            self.reward: np.zeros(1),
            self.is_training: False})[0] # Return one single action


