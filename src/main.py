import argparse

import random as rd

from network import *
from evaluator import *
from gym_wrapper import *
from iterator import *
from inverted_pendulum import InvertedPendulumWrapper

from initial_policies import *

from argparse import Namespace

from tensorflow.python.ops import control_flow_ops


params = Namespace(

    # name="Pendulum-v0",
    name="MountainCarContinuous-v0",

    # BATCH INFO
    no_episodes=150,
    max_length = 200,
    keep_ratio = 0.9,

    # VALUE ITERATION INFO
    no_pretrained_policy = 3,
    no_first_q_iteration = 3,
    no_network_iteration = 5,
    no_sampling_iteration = 20,

    # TRAINING INFO
    taylor_order = 1,
    q_layer_size = [16],
    pi_layer_size = [16, 16],
    mini_batch_size = 32,
    gamma = 0.99,
    alpha = 1,
    tau = 0.001,
    lmbda = 0.001,
    q_lrt = 1e-2,
    pi_lrt = 1e-3,
    pi_lrt_pretrain = 1e-2,

    display = True
)


def start(params):

    # Initialize environment
    gym = Sampler.create_from_gym_name(params.name)
    if params.name == 'Pendulum-v0':
        initial_policy = initial_pendulum_policy()
    elif params.name == "MountainCarContinuous-v0":
        initial_policy = initial_moutain_car_policy(mean=0.5, std=0.01)
    else:
        assert False




    # # Initialize network
    network = Network(gym.state_size, gym.action_size,
                      q_layer_size=params.q_layer_size,
                      pi_layer_size=params.pi_layer_size,
                      tau=params.tau,
                      gamma=params.gamma,
                      alpha=params.alpha,
                      lmbda=params.lmbda,
                      taylor_order=params.taylor_order)

    q_optimizer = tf.train.AdamOptimizer(learning_rate=params.q_lrt).minimize(network.q_loss)
    policy_optimizer = tf.train.AdamOptimizer(learning_rate=params.pi_lrt).minimize(network.policy_loss)
    pretrained_policy_optimizer = tf.train.AdamOptimizer(learning_rate=params.pi_lrt_pretrain).minimize(network.pretrain_policy_loss)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # if update_ops:
    #     with tf.control_dependencies(update_ops):
    #         q_optimizer = tf.train.AdamOptimizer(learning_rate=params.q_lrt).minimize(network.q_loss)
    #         policy_optimizer = tf.train.AdamOptimizer(learning_rate=params.pi_lrt).minimize(network.policy_loss)
    #         pretrained_policy_optimizer = tf.train.AdamOptimizer(learning_rate=params.pi_lrt_pretrain).minimize(network.pretrain_policy_loss)


    #summary_plotter = network.merged
    print("Network built!")


    samples = gym.compute_samples(policy=initial_policy, no_episodes=params.no_episodes,
                                  max_length=params.max_length, flatten=False)
    v0_samples = []
    v0 = []
    for trajectory in samples:
        v0_samples.append(trajectory[0])
        reward = 0
        for t, step in enumerate(trajectory[::-1]):
            reward = step.reward + params.gamma*reward
        v0.append(reward)
    v0 = np.mean(v0)
    samples = list(itertools.chain(*samples))
    print("V0 from samples: {}".format(v0))


    def evaluate(sess, gym, policy, runner, max_length, i=0):
        ### Evaluate
        l2 = runner.process(sess, iterator=dataset_iterator, output=network.q_loss)
        print("Q error: {}".format(l2))

        gym.evaluate(sess, runner, policy, gamma=params.gamma, no_episodes=1, max_length=max_length, display=params.display)
        res, _, _ = gym.evaluate(sess, runner, policy, gamma=params.gamma, no_episodes=50, max_length=max_length, display=False)
        print("step {} \t Reward/std : {} +/- {}".format(i, res[0], res[1]))

        return res

    res = []

    with tf.Session() as sess:

        writer = None # tf.summary.FileWriter("/home/fstrub/Projects/bpi_continuous/graph_log", sess.graph)
        runner = Evaluator(params.mini_batch_size, "", writer=writer)

        sess.run(tf.global_variables_initializer())
        runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})



        dataset_iterator = Dataset(samples)
        dataset_v0_iterator = Dataset(v0_samples)

        pretrained_dataset_iterator = DatasetPretrained(samples)
        for _ in range(params.no_pretrained_policy):
            [loss] = runner.process(sess, iterator=pretrained_dataset_iterator,
                           output=network.pretrain_policy_loss,
                           optimizer=pretrained_policy_optimizer)
            print("CE policy loss (training): {}".format(loss))
        runner.execute(sess, output=network.update_networks, sample={"tau": 1.0})

        res.append(evaluate(sess, gym, network.policy_output, runner, params.max_length, -1))


        #######################
        # First round
        ######################

        for _ in range(params.no_first_q_iteration):
            [l2] = runner.process(sess, iterator=dataset_iterator,
                           summary=network.merged,
                           output=[network.q_loss],
                           optimizer=q_optimizer,
                           update=network.q_update)
            print(" - Q error (train): {}".format(l2))


            [v0, q, q_next] = runner.process(sess, iterator=dataset_v0_iterator, summary=None,
                           output=[tf.reduce_mean(network.taylor[0]),
                                   tf.reduce_mean(network.q_output),
                                   tf.reduce_mean(network.q_next)])

            print(" - v0 : {}".format(v0))
            print(" - q : {}".format(q))
            print(" - q_next : {}".format(q_next))
            print()


        #######################
        # Next Rounds
        ######################

        for t in range(params.no_sampling_iteration-1):


            # # Resample (to optimize)
            # if t > 0:
            # noise_fct = create_normal_noise(0.05)
            #
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
            for _ in range(params.no_network_iteration):


                # runner.process(sess, iterator=dataset_iterator,
                #                optimizer=q_optimizer,
                #                update=network.q_update)
                # runner.process(sess, iterator=dataset_iterator,
                #                optimizer=policy_optimizer,
                #                update=network.policy_update)


                # Compute the number of required samples
                n_iter = int(dataset_iterator.no_samples / params.mini_batch_size) + 1

                q_loss, pi_loss = 0.0, 0.0
                for i in range(n_iter):

                    batch = dataset_iterator.next_batch(params.mini_batch_size, shuffle=True)

                    # Appending is_training flag to the feed_dict
                    batch["is_training"] = True

                    # evaluate the network on the batch
                    q_res = runner.execute(sess, [network.q_loss, q_optimizer], batch)
                    pi_res = runner.execute(sess, [network.policy_loss, policy_optimizer], batch)
                    runner.execute(sess, network.update_networks, batch)

                    q_loss += q_res[0]
                    pi_loss += pi_res[0]

                    # writer.add_summary(q_res[-1])
                    # writer.add_summary(pi_res[-1])

                q_loss /= n_iter
                pi_loss /= n_iter

                print(" q_loss : {}".format(q_loss))
                print(" pi_loss : {}".format(pi_loss))


            # Evaluate
            res.append(evaluate(sess, gym, network.policy_output, runner, params.max_length, t))

            [v0, q, q_next] = runner.process(sess, iterator=dataset_v0_iterator, summary=None,
                                             output=[tf.reduce_mean(network.taylor[0]),
                                                     tf.reduce_mean(network.q_output),
                                                     tf.reduce_mean(network.q_next)])

            print(" - v0 : {}".format(v0))
            print(" - q : {}".format(q))
            print(" - q_next : {}".format(q_next))
            print()

    print("Done!")
    return res

if __name__ == '__main__':
    start(params)