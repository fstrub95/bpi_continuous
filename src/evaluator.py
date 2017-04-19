import numpy as np

from tqdm import tqdm


class Evaluator(object):

    def __init__(self, batch_size, allow_inputs, scope="", writer=None,
                 network=None, tokenizer=None): #debug purpose only
        self.batch_size = batch_size
        self.allow_inputs = allow_inputs
        self.scope = scope
        self.writer = writer
        if len(scope) > 0 and not scope.endswith("/"):
            self.scope += "/"
        self.use_summary = False

        # debug purpose only
        self.tokenizer = tokenizer
        self.network = network


    def process(self, sess, iterator, output=[], dico={}, optimizer=None, summary=None, update=None):

        if not isinstance(output, list):
            output = [output]

        is_training = False
        if optimizer is not None:
            is_training = True
            output = [optimizer] + output

        use_summary = False
        if summary is not None and self.writer is not None:
            output += [summary]
            use_summary = True

        res = self.__execute__(sess, iterator, output, update, dico, is_training, use_summary)


        if is_training:
            res = res[1:]

        if use_summary:
            res = res[:-1]

        return res


    def __execute__(self, sess, iterator, output, update, dico, is_training, use_summary):

        res = [0.0 for _ in output]
        # Compute the number of required samples
        n_iter = int(iterator.no_samples / self.batch_size) + 1
        for i in range(n_iter):

            batch = iterator.next_batch(self.batch_size, shuffle=is_training)

            # Appending is_training flag to the feed_dict
            batch["is_training"] = is_training
            batch = {**batch, **dico}  # merge both dictionary (python > 3.5)

            # evaluate the network on the batch
            one_res = self.execute(sess, output, batch)

            if update is not None:
                self.execute(sess, update, batch)

            # process the results
            for i, r in enumerate(one_res):
                if isinstance(r, np.float32): # Loss
                    res[i] += r

            if use_summary:
                self.writer.add_summary(one_res[-1])

        # compute loss
        for i, r in enumerate(res):
            if isinstance(r, float):
                res[i] = 1.0*r/n_iter
        return res









    def execute(self, sess, output, sample):
        feed_dict = { self.scope + key + ":0" : value for key, value in sample.items() } #if key in self.allow_inputs
        return sess.run(output, feed_dict=feed_dict)



# class Runner(object):
#
#     def __init__(self, batch_size, writer=None):
#         self.batch_size = batch_size
#         self.writer = writer
#
#     def eval_loss(self, sess, iterator, loss, summary=None):
#         return self.__execute__(sess, iterator, loss, summary, False)
#
#     def train(self, sess, iterator, optimizer, summary=None):
#         return self.__execute__(sess, iterator, [optimizer], summary, True)
#
#     def __execute__(self, sess, iterator, output, summary, is_training):
#
#         if self.writer is not None and summary is not None:
#             output.append(summary)
#
#         # Compute the number of required samples
#         n_iter = int(iterator.no_samples / self.batch_size) + 1
#
#         loss = 0
#         for i in range(n_iter):
#
#
#             # Creating the feed_dict
#             batch = iterator.next_batch(self.batch_size, shuffle=is_training)
#             # tflearn.config.is_training(is_training=is_training, session=sess)
#
#             # compute loss
#
#             res = self.execute(sess, output, batch)
#             #if writer
#
#             if res is not None:
#                 loss += res
#         loss /= n_iter
#
#
#         # By default, there is no training
#         # tflearn.config.is_training(is_training=False, session=sess)
#
#         return loss
#
#     def execute(self, sess, output, sample):
#         return sess.run(output, feed_dict={ key+":0" : value for key, value in sample.items()})
#
