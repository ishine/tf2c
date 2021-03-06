from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import struct
import tensorflow as tf

from tf2c import emitter


OpType = collections.namedtuple('OpType', [
    'name', 'arity', 'is_simple'
])

OP_MAP = {}
for op in [
        ('Const', 0, False),
        ('Variable', 0, False),
        ('VariableV2', 0, False),
        ('Identity', 1, True),
        ('Tanh', 1, True),
        ('Sigmoid', 1, True),
        ('Fill', 2, True),
        ('Add', 2, True),
        ('Mul', 2, True),
        ('Sum', 2, True),
        ('BroadcastGradientArgs', 2, True),
        ('Reshape', 2, True),
        ('Minimum', 2, True),
        ('Maximum', 2, True),
        ('LessEqual', 2, True),
        ('GreaterEqual', 2, True),
        ('MatMul', 2, False),
        ('SigmoidGrad', 2, True),
        ('TanhGrad', 2, True),
        ('Select', 3, True),
]:
    OP_MAP[op[0]] = OpType(*op)


class Compiler(object):
    def __init__(self, model=None):
        self._ce = emitter.CodeEmitter()
        self._model = model

    def _emit_args(self, node, op):
        args = []
        for i in xrange(op.arity):
            self._ce.emit_line('Tensor* a%d = %s();' %
                               (i, node.inputs[i].ident))
            args.append('a%d' % i)
        return args

    def _compile_node(self, node):
        ce = self._ce
        op = OP_MAP[node.op]
        assert op.arity == len(node.inputs)
        name = node.ident

        if op.is_simple:
            ce.emit_line('Tensor* %s() {' % name)
            args = self._emit_args(node, op)
            ce.emit_line('return tf2c_%s<void>(%s);' %
                         (op.name, ','.join(args)))
            ce.emit_line('}')
            return

        if (op.name == 'Const' or op.name == 'Variable' or
            op.name == 'VariableV2'):
            ce.emit_line('Tensor* g_%s;' % name)
            ce.emit_line('Tensor* %s() {' % name)
            ce.emit_line('return g_%s;' % name)
            ce.emit_line('}')

        elif op.name == 'MatMul':
            transpose_a = '1' if node.attr('transpose_a').b else '0'
            transpose_b = '1' if node.attr('transpose_b').b else '0'
            ce.emit_line('Tensor* %s() {' % name)
            args = self._emit_args(node, op)
            args.append(transpose_a)
            args.append(transpose_b)
            ce.emit_line('return tf2c_%s<void>(%s);' %
                         (op.name, ','.join(args)))
            ce.emit_line('}')

    def _emit_shape(self, ce, shape, var_name='shape'):
        ce.emit_line('static const int %s_dims[] = {%s};' %
                     (var_name, shape.dims_str()))
        ce.emit_line('const Shape %s = tf2c_shape(%s_dims);' %
                     (var_name, var_name))

    def _compile_init(self, node):
        ce = self._ce
        op = OP_MAP[node.op]
        name = node.ident

        if op.name == 'Const':
            ce.emit_line('{')
            value = node.value
            self._emit_shape(ce, value.shape)
            ce.emit_line('g_%s = tf2c_tensor(%s, shape);' %
                         (name, node.dtype.upper()))
            if value.shape.size == 0:
                pass
            elif value.shape.size == 1:
                ce.emit_line('tf2c_fill<%s>(g_%s, %s);' %
                             (node.dtype, name, str(value.value[0])))
            else:
                ce.emit_line('static const %s v[] = {' % node.dtype)
                ce.emit_line(', '.join(map(str, value.value)))
                ce.emit_line('};')
                ce.emit_line('tf2c_assign(g_%s, v);' % name)
            ce.emit_line('}')

        elif op.name == 'Variable' or op.name == 'VariableV2':
            ce.emit_line('{')
            self._emit_shape(ce, node.shape)
            ce.emit_line('g_%s = tf2c_tensor(%s, shape);' %
                         (name, node.dtype.upper()))

            assert self._model
            reader = tf.train.NewCheckpointReader(self._model)
            values = reader.get_tensor(node.name).flatten()
            fname = self._model.replace('.ckpt', '-%s.data' % node.ident)
            with open(fname, 'w') as f:
                pack_type = node.dtype[0]
                for v in values:
                    f.write(struct.pack(pack_type, v))
            ce.emit_line('tf2c_load(g_%s, "%s");' % (name, fname))
            ce.emit_line('}')

    def compile(self, g):
        self._ce.emit_line('#include "lib/tf2c.h"')
        self._ce.emit_line('')

        for node in g.all_nodes():
            self._compile_node(node)

        self._ce.emit_line('void init() {')
        for node in g.all_nodes():
            self._compile_init(node)
        self._ce.emit_line('}')
