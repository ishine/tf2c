#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import subprocess
import sys

import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import tensor_util


def dump_tensor(value):
    print(value.shape)
    print(list(value.flatten()))

def undump_tensor(input):
    lines = input.splitlines()
    assert len(lines) == 2
    return map(eval, lines)

mode = sys.argv[1]
name = sys.argv[2]
if mode == 'output':
    module = importlib.import_module('tests.' + name)
    op = module.gen_graph()
    tf.train.write_graph(op.graph.as_graph_def(), 'out', name + '.pbtxt')

    sess = tf.Session()
    result = sess.run(op)
    dump_tensor(result)

elif mode == 'test':
    with open('out/%s.out' % name) as f:
        expected = undump_tensor(f.read())
    output = subprocess.check_output(['out/%s.exe' % name])
    actual = undump_tensor(output)
    if expected != actual:
        sys.stderr.write('expected %s but %s' % (expected, actual))
        sys.exit(1)

else:
    raise RuntimeError('Unknown mode: %s' % mode)
