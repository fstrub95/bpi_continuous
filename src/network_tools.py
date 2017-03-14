import tensorflow as tf
import tflearn


def dot_product(x, y):
    with tf.name_scope('dot_product'):
        return tf.reduce_sum(tf.mul(x, y),axis=1, keep_dims=True)



# Helper to create multilayer perceptron
def create_nlp(input, layer_size, activ="relu"):

    x = input
    for i, layer in enumerate(layer_size[:-1]):
        x = tflearn.fully_connected(x, layer, activation=activ)
        # x = tflearn.layers.normalization.batch_normalization(x)

    x = tflearn.fully_connected(x, layer_size[-1], activation='linear')


    return x