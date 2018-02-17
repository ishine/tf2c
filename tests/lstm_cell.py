import tensorflow as tf


def gen_graph():
    x = tf.Variable(tf.random_normal([500, 1024]), name='x')
    w = tf.Variable(tf.random_normal([1024, 4096]), name='w')
    z = tf.matmul(x, w)
    i, j, f, o = tf.split(z, num_or_size_splits=4, axis=1)
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


def num_ops():
    return 500 * 1024 * 4096
