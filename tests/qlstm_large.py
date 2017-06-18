import tensorflow as tf


def gen_graph():
    i = tf.Variable(tf.random_normal([100, 1024]), name='i')
    j = tf.Variable(tf.random_normal([100, 1024]), name='j')
    f = tf.Variable(tf.random_normal([100, 1024]), name='f')
    o = tf.Variable(tf.random_normal([100, 1024]), name='o')
    c = tf.Variable(tf.random_normal([100, 1024]), name='c')

    i = tf.sigmoid(i)
    j = tf.tanh(j)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    nc = tf.clip_by_value(c * f + j * i, -0.5, 0.5)
    nh = tf.multiply(nc, o, name='result')
    return nh


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
