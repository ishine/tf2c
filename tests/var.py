import tensorflow as tf


def gen_graph():
    a = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name='a')
    b = tf.Variable([[5.0, 6.0], [7.0, 8.0]], name='b')
    r = tf.add(a, b, name='result')
    return r


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
