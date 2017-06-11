import tensorflow as tf


def gen_graph():
    a = tf.Variable(tf.random_normal([1, 1024 * 4]), name='a')
    b = tf.Variable(tf.random_normal([1024 * 4, 1024 * 4]), name='b')
    r = tf.matmul(a, b, transpose_b=True, name='result')
    return r


def gen_model(sess):
    sess.run(tf.initialize_all_variables())
