import tensorflow as tf


def gen_graph():
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    r = tf.matmul(a, b, transpose_a=True, transpose_b=True, name='result')
    return r
