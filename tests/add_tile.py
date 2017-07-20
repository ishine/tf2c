import tensorflow as tf


def gen_graph():
    a = tf.Variable(tf.random_normal([24, 20, 1024]), name='a')
    b = tf.Variable(tf.random_normal([24, 1, 1024]), name='b')
    r = tf.add(a, b, name='result')
    return r


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
