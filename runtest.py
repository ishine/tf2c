#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import subprocess
import sys
import time

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_util


def dump_tensor(value):
    print(value.shape)
    print(list(value.flatten()))

def undump_tensor(input):
    lines = input.splitlines()
    assert len(lines) == 3
    return map(eval, lines)

mode = sys.argv[1]
name = sys.argv[2]
if mode == 'output':
    module = importlib.import_module('tests.' + name)
    op = module.gen_graph()
    tf.train.write_graph(op.graph.as_graph_def(), 'out', name + '.pbtxt')

    sess = tf.Session()
    if hasattr(module, 'gen_model'):
        module.gen_model(sess)
        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(sess, 'out/%s.ckpt' % name)

    start = time.time()
    result = sess.run(op)
    elapsed = time.time() - start
    dump_tensor(result)
    print(elapsed)

elif mode == 'test':
    with open('out/%s.out' % name) as f:
        expected = undump_tensor(f.read())
    output = subprocess.check_output(['out/%s.exe' % name])
    actual = undump_tensor(output)
    if expected == actual:
        sys.exit(0)

    if expected[0] != actual[0]:
        sys.stderr.write('expected shape %s but %s\n' %
                         (expected[0], actual[0]))
        sys.exit(1)

    expected = expected[1]
    actual = actual[1]
    ok = True
    for i in range(len(expected)):
        e = expected[i]
        a = actual[i]
        if abs(e - a) > 1e-3:
            sys.stderr.write('expected %f but %f at %d\n' % (e, a, i))
            ok = False

    if not ok:
        sys.exit(1)

else:
    raise RuntimeError('Unknown mode: %s' % mode)
