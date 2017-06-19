import tensorflow as tf


def gen_graph():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    r = tf.multiply(a, b)
    ag = tf.gradients(r, a)[0]
    return tf.identity(ag, name='result')
