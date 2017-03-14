import argparse

import random as rd

from network import *
from gym_wrapper import *
from iterator import *
from inverted_pendulum import InvertedPendulumWrapper





# BATCH INFO
no_episodes = 150
max_length = 100
keep_ratio = 0.9

# VALUE ITERATION INFO
no_first_q_iteration = 5
no_network_iteration = 5
no_sampling_iteration = 10

# TRAINING INFO
layer_size = [64, 64]
mini_batch_size = 20
gamma = 0.99
alpha = 1
tau=0.1
lmbda=0.00
q_lrt = 0.01
pi_lrt = 0.001


def initial_pendulum_policy(state):
    return [np.random.uniform(-2, 2)]


def initial_moutain_car_policy(state):

    position = state[0]
    velocity = state[1]

    if velocity > 0:
        action = np.random.uniform(0,0.5)
    else:
        action = np.random.uniform(-0.5,0)

    action += np.random.normal()*0.2
    return [action]



if __name__ == '__main__':

    # Initialize environment
    gym = Sampler.create_from_gym_name('Pendulum-v0')
    initial_policy = initial_pendulum_policy

    # gym = Sampler.create_from_gym_name('MountainCarContinuous-v0')
    # initial_policy = initial_moutain_car_policy

    # gym = Sampler.create_from_perso_env(InvertedPendulumWrapper())  # Sampler('Pendule-v0')
    #gym = Sampler.create_from_gym_name('InvertedPendulum-v1')


    # # Initialize network
    network = Network(gym.state_size, gym.action_size,
                      layer_size=layer_size,
                      tau=tau,
                      gamma=gamma,
                      alpha=alpha,
                      lmbda=lmbda,
                      taylor_order=1)

    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lrt).minimize(network.q_loss)
    policy_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lrt).minimize(network.policy_loss)

    # q_optimizer = tf.group(
    #    tf.train.AdamOptimizer(learning_rate=q_lrt).minimize(network.q_loss),
    #    network.q_update)

    #policy_optimizer = tf.group(
    #    tf.train.AdamOptimizer(learning_rate=pi_lrt).minimize(network.policy_loss),
    #    network.policy_update)

    runner = Runner(mini_batch_size)
    print("Network built!")



    # Start training!
    with tf.Session() as sess:

        writer = tf.summary.FileWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
        sess.run(tf.global_variables_initializer())

        # equalize q/policy networks
        runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})

        samples = gym.compute_samples(policy=initial_policy, no_episodes=no_episodes, max_length=max_length)
        dataset_iterator = Dataset(samples)

        # sample = {'state': np.array([[ 0.98905226,  0.14756566,  0.23730638]]), 'next_state': np.array([[ 0.98508077,  0.17209263,  0.49694141]]), 'reward': np.array([-0.02855317]), 'action': np.array([[ 0.99307193]])}
        # sess.run([], feed_dict={ key+":0" : value for key, value in sample.items()})

        #######################
        # function evaluator
        ######################

        def evaluate(sess, gym, policy, runner, max_length, i=0):
            ### Evaluate
            l2 = runner.eval_loss(sess, iterator=dataset_iterator, loss=network.q_loss)
            print("Q error: {}".format(l2))

            gym.evaluate(sess, runner, policy, gamma=gamma, no_episodes=1, max_length=max_length, display=True)
            res, _ = gym.evaluate(sess, runner, policy, gamma=gamma, no_episodes=50, max_length=max_length, display=False)
            print("step {} \t Reward/std : {} +/- {}".format(i, res[0], res[1]))

            return res

        #######################
        # First round
        ######################
        for _ in range(no_first_q_iteration):
            runner.train(sess, iterator=dataset_iterator, optimizer=q_optimizer)
            runner.execute(sess, output=network.q_update, sample={"tau": 1.0})
            l2 = runner.eval_loss(sess, iterator=dataset_iterator, loss=network.q_loss)
            print(" - Q error: {}".format(l2))

        evaluate(sess, gym, network.policy_output, runner, max_length)
        runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})
        runner.train(sess, iterator=dataset_iterator, optimizer=policy_optimizer)
        #evaluate(sess, gym, network.policy_output, runner, max_length)

        #######################
        # Next Rounds
        ######################

        for t in range(no_sampling_iteration-1):

            # Resample (to optimize)
            noise_fct = create_normal_noise(0.05)

            # samples, old_samples = [], samples
            # for old_sample in old_samples:
            #     if rd.random() < keep_ratio:
            #         sample = old_sample
            #     else:
            #         sample = gym.compute_samples(sess, runner, network.policy_output, no_episodes=1, max_length=max_length, noise_fct=noise_fct)[0]
            #     samples.append(sample)
            #
            # dataset_iterator = Dataset(samples)

            # Train
            for _ in range(no_network_iteration):
                runner.train(sess, iterator=dataset_iterator, optimizer=q_optimizer)
                runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})
                runner.train(sess, iterator=dataset_iterator, optimizer=policy_optimizer)
                runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})

            # Evaluate
            evaluate(sess, gym, network.policy_output, runner, max_length, t+1)
    #gym.commit()

    print("Done!")
