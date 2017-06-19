import tensorflow as tf


def gen_graph():
    a = tf.constant([[1, 2], [3, 4], [5, 6]])
    b = tf.constant([[7, 8, 9], [10, 11, 12]])
    r = tf.matmul(a, b, name='result')
    return r
