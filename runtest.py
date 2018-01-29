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


jit_scope = tf.contrib.compiler.jit.experimental_jit_scope


def dump_tensor(value):
    print(value.shape)
    print(list(value.flatten()))

def undump_tensor(input):
    lines = input.splitlines()
    assert len(lines) == 3
    return map(eval, lines)

args = sys.argv[1:]

use_jit = False
if args[0].startswith('--'):
    opt = args.pop(0)
    if opt == '--jit':
        use_jit = True
    else:
        raise RuntimeError('Unknown flag: %s' % opt)

mode = args[0]
name, _ = os.path.splitext(os.path.basename(args[1]))
if mode == 'output':
    module = importlib.import_module('tests.' + name)
    with jit_scope(compile_ops=use_jit):
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

elif mode == 'bench':
    if not os.path.exists('out/%s.pbtxt' % name):
        raise RuntimeError('Run runtest.py output first')

    module = importlib.import_module('tests.' + name)
    with jit_scope(compile_ops=use_jit):
        op = module.gen_graph()

    sess = tf.Session()
    if hasattr(module, 'gen_model'):
        saver = tf.train.Saver(tf.trainable_variables())
        saver.restore(sess, 'out/%s.ckpt' % name)

    # Warm-up.
    sess.run(op)

    start = time.time()
    n = 0
    while True:
        result = sess.run(op)
        n += 1
        elapsed = time.time() - start
        if elapsed > 1.0:
            break
    print('%f' % (elapsed / n))

elif mode == 'test' or mode == 'test_misc':
    with open('out/%s.out' % name) as f:
        expected = undump_tensor(f.read())
    prefix = 'misc_' if mode == 'test_misc' else ''
    output = subprocess.check_output(['out/%s%s.exe' % (prefix, name)])
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
