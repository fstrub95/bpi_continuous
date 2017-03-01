import tensorflow as tf
import tflearn


def dot_product(x, y):
    with tf.name_scope('dot_product'):
        return tf.reduce_sum(tf.mul(x, y),axis=1, keep_dims=True)


def standard_batch_norm(x, is_training, name='batch_normalization', scope=None):
    # avoid using batch_normalization during evaluation
    return  tf.cond(is_training,
                    lambda: tflearn.layers.normalization.batch_normalization(x, name=name, scope=scope),
                    lambda: x,
                    name="cond_"+name)


# Helper to create multilayer perceptron
def create_nlp(input, layer_size, is_training, activ="relu",  use_scope=False):

    x = input
    for i, layer in enumerate(layer_size[:-1]):

        # Compute scope if required
        if use_scope:
            scope_layer = "layer_" + str(i)
            scope_batch = "batch_normalization_1_" + str(i)
        else:
            scope_layer = None
            scope_batch = None

        # compute layer
        x = tflearn.fully_connected(x, layer, activation=activ, scope=scope_layer)
        x = standard_batch_norm(x, is_training, scope=scope_batch)

    # Output layer
    scope_layer = "output_layer" if use_scope else None
    x = tflearn.fully_connected(x, layer_size[-1], activation='linear', scope=scope_layer)

    return x