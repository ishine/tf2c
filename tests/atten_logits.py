import tensorflow as tf


def gen_graph():
  a = tf.Variable(tf.random_normal([24, 20, 1024]), name='a')
  b = tf.Variable(tf.random_normal([24, 1, 1024]), name='b')
  v = tf.Variable(tf.random_normal([1024, 1]), name='v')
  h = a + b
  h = tf.tanh(h)
  h = tf.reshape(h, [-1, 1024])
  h = tf.matmul(h, v, name='result')
  return h


def gen_model(sess):
    sess.run(tf.global_variables_initializer())
