import tensorflow as tf


def gen_graph():
    a = tf.Variable(tf.random_normal([1, 1024 * 3]), name='a')
    b = tf.Variable(tf.random_normal([1024 * 3, 1024 * 4]), name='b')
    r = tf.matmul(a, b, name='result')
    return r


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
