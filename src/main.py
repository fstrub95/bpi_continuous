import argparse

import random as rd

from network import *
from gym_wrapper import *
from iterator import *
from inverted_pendulum import InvertedPendulumWrapper





# BATCH INFO
no_episodes = 100
max_length = 100
keep_ratio = 0.8

# VALUE ITERATION INFO
no_first_q_iteration = 10
no_network_iteration = 5
no_sampling_iteration = 20

# TRAINING INFO
layer_size = [64, 64]
mini_batch_size = 2
gamma = 0.95
alpha = 1
q_lrt = 0.001
pi_lrt = 0.001



if __name__ == '__main__':

    # Initialize environment
    gym = Sampler.create_from_gym_name('Pendulum-v0') #Sampler('Pendule-v0')
    # gym = Sampler.create_from_gym_name('MountainCarContinuous-v0')  # Sampler('Pendule-v0')
    # gym = Sampler.create_from_perso_env(InvertedPendulumWrapper())  # Sampler('Pendule-v0')

    gym.compute_samples(no_episodes=no_episodes, max_length=max_length)

    # Initialize network
    network = Network(gym.state_size, gym.action_size,
                      layer_size=layer_size,
                      gamma=gamma,
                      alpha=alpha)

    policy_optimizer = tf.group(
        tf.train.AdamOptimizer(learning_rate=pi_lrt).minimize(network.q_loss),
        network.q_update)

    q_optimizer = tf.group(
        tf.train.AdamOptimizer(learning_rate=pi_lrt).minimize(network.policy_loss),
        network.policy_update)

    runner = Runner(mini_batch_size)
    print("Network built!")



    # Start training!
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
        sess.run(tf.global_variables_initializer())

        samples = gym.compute_samples(no_episodes=no_episodes, max_length=max_length)  # Random sampling
        dataset_iterator = Dataset(samples)

        #######################
        # function evaluator
        ######################

        def evaluate(sess, gym, policy, runner, max_length, i=0):
            ### Evaluate
            l2 = runner.eval_loss(sess, iterator=dataset_iterator, loss=network.q_loss)
            print("Q error: {}".format(l2))

            gym.evaluate(sess, runner, policy, gamma=gamma, no_episodes=1, max_length=200, display=True)
            res, _ = gym.evaluate(sess, runner, policy, gamma=gamma, no_episodes=50, max_length=max_length, display=False)
            print("step {} \t Reward/time : {}".format(i, res))

        #######################
        # First round
        ######################
        for _ in range(no_first_q_iteration):
            runner.train(sess, iterator=dataset_iterator, optimizer=q_optimizer)
            runner.execute(sess, output=network.update_networks, sample={})


        evaluate(sess, gym, network.policy_output, runner, max_length)

        #######################
        # Next Rounds
        ######################

        for t in range(no_sampling_iteration-1):

            # Resample (to optimize)
            new_samples = gym.compute_samples(sess, runner, network.policy_output, no_episodes=no_episodes, max_length=max_length, std=0.05)
            old_samples = [s for s in samples if rd.random() < keep_ratio]
            samples = new_samples + old_samples
            dataset_iterator = Dataset(samples)

            # Train
            for _ in range(no_network_iteration):
                runner.train(sess, iterator=dataset_iterator, optimizer=q_optimizer)
                runner.train(sess, iterator=dataset_iterator, optimizer=policy_optimizer)

            # Evaluate
            evaluate(sess, gym, network.policy_output, runner, max_length, t+1)

    print("Done!")
