import tensorflow as tf


def gen_graph():
    a = tf.Variable(tf.random_normal([1, 1024]), name='a')
    r = tf.sigmoid(a, name='result')
    return r


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
