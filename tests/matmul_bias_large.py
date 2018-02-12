import tensorflow as tf


N = 1024


def gen_graph():
    a = tf.Variable(tf.random_normal([N, N]), name='a')
    b = tf.Variable(tf.random_normal([N, N]), name='b')
    c = tf.Variable(tf.random_normal([N, N]), name='c')
    r = tf.matmul(a, b, name='result')
    r = tf.add(tf.matmul(a, b), c, name='result')
    return r


def gen_model(sess):
    sess.run(tf.global_variables_initializer())


def num_ops():
    return N * N * N * 2 + N * N
