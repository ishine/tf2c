import tensorflow as tf


def gen_graph():
    vi = tf.Variable(tf.random_normal([10, 16]), name='i')
    vj = tf.Variable(tf.random_normal([10, 16]), name='j')
    vf = tf.Variable(tf.random_normal([10, 16]), name='f')
    vo = tf.Variable(tf.random_normal([10, 16]), name='o')
    vc = tf.Variable(tf.random_normal([10, 16]), name='c')

    i = tf.sigmoid(vi)
    j = tf.tanh(vj)
    f = tf.sigmoid(vf)
    o = tf.sigmoid(vo)
    nc = tf.clip_by_value(vc * f + j * i, -0.5, 0.5)
    nh = tf.multiply(nc, o)
    g = tf.gradients(nh, vj)[0]
    return tf.identity(g, name='result')


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
