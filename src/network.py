import tensorflow as tf
from network_tools import *

def build_multilayer_perceptron(input, layer_size, layers=None):

    if layers is None:
        layers = [Layer(layer_size[i], layer_size[i + 1]) for i in range(len(layer_size) - 1)]

    output = None
    for layer in layers[:-1]:
        output = tf.matmul(input, layer.w) + layer.b
        output = tf.nn.elu(output)

        # Regularisation
        #output = tf.nn.dropout(y_Qa, self.dropout)
        #output = tf.nn.dropout(y_Qa, self.dropout)

        input = output

    last_layer = layers[-1]

    output = tf.matmul(input, last_layer.w) + last_layer.b

    return output, layers




class Network(object):

    def __init__(self, state_size, action_size):

        self.state = tf.placeholder(tf.float32, [None, state_size], name='state')
        self.action = tf.placeholder(tf.float32, [None, action_size], name='action')
        self.reward = tf.placeholder(tf.float32, [None,], name='state')
        self.next_state = tf.placeholder(tf.float32, [None, state_size], name='next_state')

        self.gamma = tf.constant(tf.float32, 0.9, "gamma")

        state_action = tf.concat(1, (self.state,self.action))

        def build_q

        # Zero order: Value function
        with tf.name_scope('value_fct'):
            zero_order_state, _ = build_multilayer_perceptron(self.state, [20, 1])
            zero_order_next_state, _ = build_multilayer_perceptron(self.next_state, [20, 1])

        # First order: Gradient
        with tf.name_scope('grad'):
            self.grad = build_multilayer_perceptron(state_action, [20, action_size])

        # Policy network
        with tf.name_scope('policy'):
            policy_state = build_multilayer_perceptron(self.state, [20, action_size])
            policy_next_state = build_multilayer_perceptron(self.state, [20, action_size])

        with tf.name_scope('first_order'):
            first_order_state = dot_product(self.grad, self.action - policy_state)
            first_order_next_state = dot_product(self.grad, self.action - policy_next_state)


        self.output = zero_order_state + first_order_state
        self.target = self.reward + self.gamma*(zero_order_next_state + first_order_next_state)

        self.qloss = tf.nn.l2_loss(self.output - self.target)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.qloss)


        # Second order: Hessian


        # QUESTION
        # question_config = config["input"]["question"]
        # if question_config["to_use"] == 'True':
        #
        #     self._question = tf.placeholder(tf.int32, [self.batch_size, None], name='question')
        #     self._seq_length = tf.placeholder(tf.int32, [self.batch_size], name='seq_length')
        #
        #     tf.add_to_collection('inputs', self._question)
        #     tf.add_to_collection('inputs', self._seq_length)
        #
        #     no_words = len(self.tokenizer.word2i)
        #     word_emb = utils.get_embedding(self._question,
        #                                    n_words=no_words,
        #                                    n_dim=int(question_config["embedding_dim"]),
        #                                    scope="word_embedding")
        #
        #     lstm_states = rnn.variable_length_LSTM(word_emb,
        #                                            num_hidden=int(question_config["no_LSTM_hiddens"]),
        #                                            seq_length=self._seq_length)
        #
        #     tf.add_to_collection('embedding', lstm_states)
        #
        # # CROP
        # crop_config = config["input"]["crop"]
        # if crop_config["to_use"] == 'True':
        #     self._crop_fc8 = tf.placeholder(tf.float32, [self.batch_size, 1000], name='crop_fc8')
        #     tf.add_to_collection('inputs', self._crop_fc8)
        #     tf.add_to_collection('embedding', self._crop_fc8)
        #
        # # PICTURE
        # picture_config = config["input"]["picture"]
        # if picture_config["to_use"] == 'True':
        #     self._picture_fc8 = tf.placeholder(tf.float32, [self.batch_size, 1000], name='picture_fc8')
        #     tf.add_to_collection('inputs', self._picture_fc8)
        #     tf.add_to_collection('embedding', self._picture_fc8)
        #
        # # CATEGORY
        # category_config = config["input"]["category"]
        # if category_config["to_use"] == 'True':
        #     self._category = tf.placeholder(tf.int32, [self.batch_size], name='category')
        #     tf.add_to_collection('inputs', self._category)
        #
        #     cat_emb = utils.get_embedding(self._category,
        #                                   int(category_config["n_categories"]),
        #                                   int(category_config["cat_embedding_dim"]),
        #                                   scope="cat_embedding")
        #
        #     tf.add_to_collection('embedding', cat_emb)
        #
        # # SPATIAL
        # spatial_config = config["input"]["spatial"]
        # if spatial_config["to_use"] == 'True':
        #     self._spatial = tf.placeholder(tf.float32, [self.batch_size, 8], name='spatial')
        #     tf.add_to_collection('inputs', self._spatial)
        #     tf.add_to_collection('embedding', self._spatial)
        #
        # # Compute te final embedding
        # emb = tf.concat(1, tf.get_collection('embedding'))
        #
        # # OUTPUT
        # num_classes = int(config["output"]["num_classes"])
        # self._answer = tf.placeholder(tf.float32, [self.batch_size, num_classes], name='answer')
        # tf.add_to_collection('inputs', self._answer)
        #
        # with tf.variable_scope('mlp'):
        #     # l1 = utils.residual_layer(emb, k=2, scope='res_l1', is_training=self._is_training)
        #     num_hiddens = int(config['model']['MLP']['num_hiddens'])
        #     l1 = utils.linear_relu(emb, num_hiddens, scope='l1')
        #     self._pred = utils.softmax(utils.linear(l1, num_classes, scope='softmax'))
        #
        # self._loss = utils.cross_entropy(self._pred, self._answer)
        # self._error = utils.error(self._pred, self._answer)
        #
        # print ('Model... build!')
        #
        # lrt = float(self.training_config["learning_rate"])
        # print lrt
        #
        # train_vars = [v for v in tf.trainable_variables()]
        # optimizer = tf.train.AdamOptimizer(learning_rate=lrt)
        # gvs = optimizer.compute_gradients(self._loss, var_list=train_vars)
        #
        # clipped_gvs = [(tf.clip_by_norm(grad, 5), var) for grad, var in gvs]
        #
        # self._optimize = optimizer.apply_gradients(clipped_gvs)


network = Network(1,2)