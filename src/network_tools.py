import tensorflow as tf


def dot_product(x, y):
    with tf.name_scope('dot_product'):
        return tf.reduce_sum(tf.multiply(x, y),axis=1, keep_dims=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, name='summaries'):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, output_dim, layer_name="mlp", activation=None, use_summary=False):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """

    input_dim = int(input_tensor.get_shape()[1])

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            if use_summary:
                variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            if use_summary:
                variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            if use_summary:
                tf.summary.histogram('pre_activations', preactivate)

        if activation is not None:
            activations = activation(preactivate, name='activation')
            if use_summary:
                tf.summary.histogram('activations', activations)
        else:
            activations = preactivate

        return activations


# Helper to create multilayer perceptron
def create_nlp(input, layer_size, is_training, activ=tf.nn.relu, use_summary=False):

    x = input
    for i, layer in enumerate(layer_size[:-1]):
        x = nn_layer(x, layer, use_summary=use_summary)
        # x = tf.contrib.layers.batch_norm(x, is_training=is_training, center=True, scale=True)
        x = activ(x)
        if use_summary and activ is not None:
            tf.summary.histogram('activations', x)

    x = nn_layer(x, layer_size[-1], activation=None, use_summary=use_summary)


    return x


