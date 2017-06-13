import tensorflow as tf


def gen_graph():
    a = tf.Variable(tf.random_normal([100, 1024]), name='a')
    r = tf.clip_by_value(a, -0.5, 0.5, name='result')
    return r


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
