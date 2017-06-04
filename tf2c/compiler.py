from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tf2c import emitter


OpType = collections.namedtuple('OpType', [
    'name', 'arity', 'is_simple'
])

OP_MAP = {}
for op in [
        ('const', 0, False),
        ('add', 2, True),
]:
    OP_MAP[op[0]] = OpType(*op)


class Compiler(object):
    def __init__(self):
        self._ce = emitter.CodeEmitter()

    def _compile_node(self, node):
        ce = self._ce
        op = OP_MAP[node.op.lower()]
        assert op.arity == len(node.inputs)
        name = node.ident

        if op.is_simple:
            ce.emit_line('Tensor* %s() {' % name)
            args = []
            for i in xrange(op.arity):
                ce.emit_line('Tensor* a%d = %s();' %
                             (i, node.inputs[i].ident))
                args.append('a%d' % i)
            ce.emit_line('return tf2c_%s<float>(%s);' %
                         (op.name, ','.join(args)))
            ce.emit_line('}')
            return

        if op.name == 'const':
            ce.emit_line('Tensor* g_%s;' % name)
            ce.emit_line('Tensor* %s() {' % name)
            ce.emit_line('return g_%s;' % name)
            ce.emit_line('}')

    def _emit_shape(self, ce, node, var_name='shape'):
        tensor = node.value
        ce.emit_line('static const int %s_dims[] = {%s};' %
                     (var_name, tensor.dims_str()))
        ce.emit_line('const Shape %s = tf2c_shape(%s_dims);' %
                     (var_name, var_name))

    def _compile_init(self, node):
        ce = self._ce
        op = OP_MAP[node.op.lower()]
        name = node.ident
        if op.name == 'const':
            ce.emit_line('{')
            self._emit_shape(ce, node)
            ce.emit_line('g_%s = tf2c_tensor(%s, shape);' %
                         (name, node.dtype.upper()))
            value = node.value
            assert value.size
            if value.size == 1:
                ce.emit_line('tf2c_fill(g_%s, %s);' %
                             (name, str(value.value[0])))
            else:
                ce.emit_line('static const %s v[] = {' % node.dtype)
                ce.emit_line(', '.join(map(str, value.value)))
                ce.emit_line('};')
                ce.emit_line('tf2c_assign(g_%s, v);' % name)
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
