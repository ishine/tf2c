import tensorflow as tf


def gen_graph():
    i = tf.Variable(tf.random_normal([500, 1024]), name='i')
    j = tf.Variable(tf.random_normal([500, 1024]), name='j')
    f = tf.Variable(tf.random_normal([500, 1024]), name='f')
    o = tf.Variable(tf.random_normal([500, 1024]), name='o')
    c = tf.Variable(tf.random_normal([500, 1024]), name='c')

    i = tf.sigmoid(i)
    j = tf.tanh(j)
    f = tf.sigmoid(f)
    o = tf.sigmoid(o)
    nc = c * f + j * i
    nh = tf.multiply(nc, o, name='result')
    return nh


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
