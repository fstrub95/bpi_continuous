import argparse

import random as rd

from network import *
from gym_wrapper import *
from iterator import *





# BATCH INFO
no_episodes = 50
max_length = 100
keep_ratio = 0.20

# VALUE ITERATION INFO
no_first_q_iteration = 5
no_network_iteration = 2
no_sampling_iteration = 4

# TRAINING INFO
layer_size=[64, 64]
mini_batch_size = 64
gamma = 0.99
alpha = 1
q_lrt=0.001
pi_lrt=0.0001

if __name__ == '__main__':

    # Initialize environment
    gym = GymWrapper('MountainCarContinuous-v0')

    # Initialize network
    network = Network(gym.state_size, gym.action_size,
                      layer_size=layer_size,
                      gamma=gamma,
                      alpha=alpha,
                      q_lrt=q_lrt,
                      pi_lrt=pi_lrt,
                      )

    print("Network built!")







    # Define session
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
        sess.run(tf.global_variables_initializer())


        samples = gym.compute_samples(sess=sess, network=network, no_episodes=no_episodes, max_length=max_length)
        dataset_iterator = Dataset(samples)

        #######################
        # First round
        ######################

        ### Q-network
        for _ in range(no_first_q_iteration):
            network.execute_q(sess, iterator=dataset_iterator, mini_batch=mini_batch_size, is_training=True)

        l2 = network.execute_q(sess, iterator=dataset_iterator, mini_batch=mini_batch_size, is_training=False)
        print("First Q error: " + str(l2))

        ### Pi-network
        network.execute_policy(sess, iterator=dataset_iterator, mini_batch=mini_batch_size, is_training=True)

        ### Evaluate
        gym.evaluate(sess=sess, network=network, max_length=50, display=True)
        res, _ = gym.evaluate(sess=sess, no_episodes=5, network=network, max_length=max_length, display=False)
        print("step 0 \t Reward/time : " + str(res))

        #######################
        # Next Round round
        ######################

        for t in range(no_sampling_iteration-1):

            # Resample
            new_samples = gym.compute_samples(sess=sess, network=network, no_episodes=no_episodes, max_length=max_length)
            old_samples = [s for s in samples if rd.random() < keep_ratio]
            samples = new_samples + old_samples

            # Train
            for _ in range(no_network_iteration):
                network.execute_q(sess, iterator=dataset_iterator, mini_batch=mini_batch_size, is_training=True)
                network.execute_policy(sess, iterator=dataset_iterator, mini_batch=mini_batch_size, is_training=True)

            # Evaluate
            res, _ = gym.evaluate(sess=sess, no_episodes=5, network=network, max_length=max_length, display=False)
            print("step "+ str(t) + " \t Reward/time : " + str(res))

            gym.evaluate(sess=sess, network=network, max_length=max_length, display=True)

    print("Done!")

    # Some doc...



    # # Sample by using random policy
    # samples = gym.compute_samples(no_episodes=50, max_length=50)
    # dataset_iterator = Dataset(samples)
    #
    # # Train
    # network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    # network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    # network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    # network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    #
    # gym.evaluate(sess=sess, network=network, max_length=50, display=True)
    # #res = gym.evaluate(sess=sess, network=network, no_episodes=5, gamma=0.99)  # to compute average reward/length
    #
    # ##############################################################
    #
    # # Sample by using network policy
    # samples = gym.compute_samples(sess=sess, network=network, no_episodes=500, max_length=50)
    # dataset_iterator = Dataset(samples)
    #
    # # Train
    # network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    # network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    # network.execute_q(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    # network.execute_policy(sess, iterator=dataset_iterator, mini_batch=20, is_training=True)
    #
    # gym.evaluate(sess=sess, network=network, display=True)
    # #res = gym.evaluate(sess=sess, network=network, no_episodes=5, gamma=0.99)  # to compute average reward/length
    # #print(res)