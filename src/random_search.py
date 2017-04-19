import random
import math
from argparse import Namespace
import main

import tensorflow as tf
import pickle

no_trial = 1000

xp = []

for t in range(no_trial):

        params = Namespace(

            # name="Pendulum-v0",
            name="MountainCarContinuous-v0",

            # BATCH INFO
            no_episodes=150,
            max_length = 200,
            keep_ratio = 0.9,

            # VALUE ITERATION INFO
            no_pretrained_policy = 5,
            no_first_q_iteration = random.randint(0, 10),
            no_network_iteration = 5,
            no_sampling_iteration = 10,

            # TRAINING INFO
            taylor_order = 1,
            q_layer_size=[16, 16],
            pi_layer_size=[16, 16],
            mini_batch_size = 32,
            gamma = 0.99,
            alpha = math.pow(10.0,random.uniform(2,-3)),
            tau = math.pow(10.0,random.uniform(-2,-6)),
            lmbda = 0.001,
            q_lrt = math.pow(10.0,random.uniform(-1,-4)),
            pi_lrt =  math.pow(10.0,random.uniform(-1,-4)),
            pi_lrt_pretrain = 0.0005,

            display = False
        )

        res = main.start(params)
        tf.reset_default_graph()

        print("===================================================")
        print("alpha:  {}".format(params.alpha))
        print("tau:    {}".format(params.tau))
        print("q_lrt:  {}".format(params.q_lrt))
        print("pi_lrt: {}".format(params.pi_lrt))
        print("q_it:   {}".format(params.no_first_q_iteration))
        m = 0
        for t, score in enumerate(res):
            print("step {} \t Reward/std : {} +/- {}".format(t, score[0], score[1]))
            m = max(m, score[0])
        print("===================================================")
        xp.append((res, params, m))

for x in xp:

    print("===================================================")
    print("alpha:  {}".format(x[1].alpha))
    print("tau:    {}".format(x[1].tau))
    print("q_lrt:  {}".format(x[1].q_lrt))
    print("pi_lrt: {}".format(x[1].pi_lrt))
    print("q_it:   {}".format(x[1].no_first_q_iteration))
    for t, score in enumerate(x[0]):
        print("step {} \t Reward/std : {} +/- {}".format(t, score[0], score[1]))
    print("===================================================")

for i, x in enumerate(xp):
    print("{} \t max: {}".format(i, x[2]))
pickle.dump(xp, file="res.pkl")
