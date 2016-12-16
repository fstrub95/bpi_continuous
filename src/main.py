import argparse
import gym

from network import *
from gym_wrapper import *
from iterator import *


if __name__ == '__main__':

    # Initialize environment
    gym = GymWrapper('MountainCarContinuous-v0')

    # Initialize network
    network = Network(gym.state_size, gym.action_size, layer_size=[20])
    print("Network built!")

    # Define session
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
        sess.run(tf.global_variables_initializer())

        # Sample by using random policy
        samples = gym.compute_samples(no_episodes=50, max_length=50)
        dataset_iterator = Dataset(samples)

        # Train
        network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
        network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
        network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
        network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)

        gym.evaluate(sess=sess, network=network, max_length=50, display=True)
        #res = gym.evaluate(sess=sess, network=network, no_episodes=5, gamma=0.99)  # to compute average reward/length

        ##############################################################

        # Sample by using network policy
        samples = gym.compute_samples(sess=sess, network=network, no_episodes=500, max_length=50)
        dataset_iterator = Dataset(samples)

        # Train
        network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
        network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
        network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
        network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)

        gym.evaluate(sess=sess, network=network, display=True)
        #res = gym.evaluate(sess=sess, network=network, no_episodes=5, gamma=0.99)  # to compute average reward/length
        #print(res)