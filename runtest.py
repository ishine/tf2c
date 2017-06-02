#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os
import sys

import tensorflow as tf
from tensorflow.python.framework import tensor_util


try:
    os.mkdir('out')
except:
    pass

def needs_regen(output, input):
    if not os.path.exists(output):
        return True
    return os.path.getmtime(output) <= os.path.getmtime(input)

def save_tensor(output, value):
    pb = tensor_util.make_tensor_proto(value)
    with open(output, 'w') as f:
        f.write(pb.SerializeToString())

name = sys.argv[1]

if needs_regen('out/%s.out' % name, 'tests/%s.py' % name):
    module = importlib.import_module('tests.' + name)
    op = module.gen_graph()
    tf.train.write_graph(op.graph.as_graph_def(), 'out', name + '.pbtxt')

    sess = tf.Session()
    result = sess.run(op)
    save_tensor('out/%s.out' % name, result)
