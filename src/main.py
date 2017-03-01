import argparse

import random as rd

from network import *
from gym_wrapper import *
from iterator import *
from inverted_pendulum import InvertedPendulumWrapper





# BATCH INFO
no_episodes = 50
max_length = 100
keep_ratio = 0.20

# VALUE ITERATION INFO
no_first_q_iteration = 5
no_network_iteration = 2
no_sampling_iteration = 4

# TRAINING INFO
layer_size=[64]
mini_batch_size = 64
gamma = 0.9
alpha = 1
q_lrt=0.001
pi_lrt=0.0001

if __name__ == '__main__':

    # Initialize environment
    # gym = Sampler.create_from_gym_name('Pendule-v0') #Sampler('Pendule-v0')
    gym = Sampler.create_from_perso_env(InvertedPendulumWrapper())  # Sampler('Pendule-v0')

#    gym.compute_samples(no_episodes=no_episodes, max_length=max_length)

    # Initialize network
    network = Network(gym.state_size, gym.action_size,
                      layer_size=layer_size,
                      gamma=gamma,
                      alpha=alpha,
                      q_lrt=q_lrt,
                      pi_lrt=pi_lrt,
                      )

    print("Network built!")


    # Start training!
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
        sess.run(tf.global_variables_initializer())

        ### Pi-network Learn random policy
        samples = gym.compute_samples(no_episodes=no_episodes, max_length=max_length)
        network.execute_policy(sess, iterator=Dataset(samples), mini_batch=mini_batch_size, is_training=True)


        #######################
        # First round
        ######################
        samples = gym.compute_samples(sess=sess, network=network, no_episodes=no_episodes, max_length=max_length)
        dataset_iterator = Dataset(samples)

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
