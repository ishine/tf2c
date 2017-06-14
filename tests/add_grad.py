import tensorflow as tf


def gen_graph():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[1, 2], [3, 4]])
    r = tf.add(a, b)
    g = tf.gradients(r, a)[0]
    return tf.identity(g, name='result')
