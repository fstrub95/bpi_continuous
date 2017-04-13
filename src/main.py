import argparse

import random as rd

from network import *
from evaluator import *
from gym_wrapper import *
from iterator import *
from inverted_pendulum import InvertedPendulumWrapper

from initial_policies import *



# BATCH INFO
no_episodes = 150
max_length = 200
keep_ratio = 0.9

# VALUE ITERATION INFO
no_pretrained_policy=0
no_first_q_iteration = 3
no_network_iteration = 5
no_sampling_iteration = 10

# TRAINING INFO
taylor_order = 1
layer_size = [16, 16]
mini_batch_size = 20
gamma = 0.99
alpha = 1
tau=0.005
lmbda=0.001
q_lrt = 0.01
pi_lrt = 0.001
pi_lrt_pretrain=0.001

display = True

if __name__ == '__main__':

    # Initialize environment
    gym = Sampler.create_from_gym_name('Pendulum-v0')
    initial_policy = initial_pendulum_policy()

    # gym = Sampler.create_from_gym_name('MountainCarContinuous-v0')
    # gym = Sampler.create_from_gym_name('CartPole-v0')
    initial_policy = initial_moutain_car_policy(mean=0.5, std=0.01)

    samples = gym.compute_samples(policy=initial_policy, no_episodes=no_episodes,
                                  max_length=max_length, flatten=False)




    # # Initialize network
    network = Network(gym.state_size, gym.action_size,
                      layer_size=layer_size,
                      tau=tau,
                      gamma=gamma,
                      alpha=alpha,
                      lmbda=lmbda,
                      taylor_order=taylor_order)

    q_optimizer = tf.train.AdamOptimizer(learning_rate=q_lrt).minimize(network.q_loss)
    policy_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lrt).minimize(network.policy_loss)
    pretrained_policy_optimizer = tf.train.AdamOptimizer(learning_rate=pi_lrt_pretrain).minimize(network.pretrain_policy_loss)


    #summary_plotter = network.merged
    print("Network built!")

    #policy_optimizer = tf.group(
    #    tf.train.AdamOptimizer(learning_rate=pi_lrt).minimize(network.policy_loss),
    #    network.policy_update)




    samples = gym.compute_samples(policy=initial_policy, no_episodes=no_episodes,
                                  max_length=max_length, flatten=False)
    v0_samples = []
    v0 = []
    for trajectory in samples:
        v0_samples.append(trajectory[0])
        reward = 0
        for t, step in enumerate(trajectory[::-1]):
            reward = step.reward + gamma*reward
        v0.append(reward)
    v0 = np.mean(v0)
    samples = list(itertools.chain(*samples))
    print("V0 from samples: {}".format(v0))


    def evaluate(sess, gym, policy, runner, max_length, i=0):
        ### Evaluate
        l2 = runner.process(sess, iterator=dataset_iterator, output=network.q_loss)
        print("Q error: {}".format(l2))

        gym.evaluate(sess, runner, policy, gamma=gamma, no_episodes=1, max_length=max_length, display=display)
        res, _ = gym.evaluate(sess, runner, policy, gamma=gamma, no_episodes=50, max_length=max_length, display=False)
        print("step {} \t Reward/std : {} +/- {}".format(i, res[0], res[1]))

        return res

    with tf.Session() as sess:

        writer = tf.summary.FileWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
        runner = Evaluator(mini_batch_size, "", writer=writer)
        sess.run(tf.global_variables_initializer())

        dataset_iterator = Dataset(samples)
        dataset_v0_iterator = Dataset(v0_samples)

        pretrained_dataset_iterator = DatasetPretrained(samples)
        for _ in range(no_pretrained_policy):
            [loss] = runner.process(sess, iterator=pretrained_dataset_iterator,
                           output=network.pretrain_policy_loss,
                           optimizer=pretrained_policy_optimizer)
            print("CE policy loss (training): {}".format(loss))
        runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})
        evaluate(sess, gym, network.policy_output, runner, max_length, -1)


        #######################
        # First round
        ######################

        for _ in range(no_first_q_iteration):
            [l2] = runner.process(sess, iterator=dataset_iterator, summary=network.merged,
                           output=[network.q_loss],
                           optimizer=q_optimizer)
            runner.execute(sess, output=network.q_update, sample={"tau": tau})
            print(" - Q error (train): {}".format(l2))


            # [v0, q, q_next] = runner.process(sess, iterator=dataset_v0_iterator, summary=None,
            #                output=[tf.reduce_mean(network.taylor[0]),
            #                        tf.reduce_mean(network.q_output),
            #                        tf.reduce_mean(network.q_next)])
            #
            # print(" - v0 : {}".format(v0))
            # print(" - q : {}".format(q))
            # print(" - q_next : {}".format(q_next))
            # print()

        evaluate(sess, gym, network.policy_output, runner, max_length)

        runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})
        runner.process(sess, iterator=dataset_iterator, optimizer=policy_optimizer)
        evaluate(sess, gym, network.policy_output, runner, max_length)

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
                runner.process(sess, iterator=dataset_iterator,
                               optimizer=q_optimizer)
                runner.execute(sess, output=network.q_update, sample={"tau": tau})
                runner.process(sess, iterator=dataset_iterator, optimizer=policy_optimizer)
                runner.execute(sess, output=network.policy_update, sample={"tau": tau})

            # Evaluate
            evaluate(sess, gym, network.policy_output, runner, max_length, t+1)
    #gym.commit()

    print("Done!")
