import tensorflow as tf


def gen_graph():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant(3)
    r = tf.less_equal(a, b, name='result')
    return r
