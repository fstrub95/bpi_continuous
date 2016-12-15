import tensorflow as tf
from math import sqrt

############# Usefull functions #################
def weight_variable(shape):
    nOut = shape[-1] # nOut
    std = 1/sqrt(nOut)

    #tf.uniform_unit_scaling_initializer()
    initial = tf.random_uniform(shape, minval=-std, maxval=std, dtype=tf.float32, seed=None)
    return tf.Variable(initial, trainable=True, name="weights")

def bias_variable(shape):
    return tf.Variable(tf.zeros_initializer(shape=shape), trainable=True, name="bias")


class Layer:
    def __init__(self, input_size, output_size):
        with tf.name_scope('layer'):
            self.w = weight_variable([input_size, output_size]) #warning regarding line/column
            self.b = bias_variable([output_size])


def dot_product(x, y):
    with tf.name_scope('dot_product'):
        return tf.reduce_sum(tf.mul(x, y),axis=1, keep_dims=True)


def create_layers(layer_size):
    with tf.name_scope('variables'):
        return [Layer(layer_size[i], layer_size[i + 1]) for i in range(len(layer_size) - 1)]


def build_multilayer_perceptron(input, layers):

    output = None
    for layer in layers[:-1]:
        output = tf.matmul(input, layer.w) + layer.b
        output = tf.nn.elu(output)

        input = output

    last_layer = layers[-1]

    output = tf.matmul(input, last_layer.w) + last_layer.b

    return output