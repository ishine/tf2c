from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import struct

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.framework import types_pb2


def get_dtype_str(dt):
    if dt == types_pb2.DT_INT32:
        return 'int'
    elif dt == types_pb2.DT_FLOAT:
        return 'float'
    else:
        raise RuntimeError('Unsupported type: %s' % dt)


class Shape(object):
    def __init__(self, shape_def):
        self.dims = [d.size for d in shape_def.dim]
        self.size = reduce(lambda a, b: a * b, self.dims)

    def dims_str(self):
        return ', '.join(map(str, self.dims) + ['-1'])


class Tensor(object):
    def __init__(self, tensor_def):
        self._tensor_def = tensor_def
        self.dtype = get_dtype_str(tensor_def.dtype)
        self.shape = Shape(tensor_def.tensor_shape)
        if tensor_def.int_val:
            self.value = tensor_def.int_val
        elif tensor_def.float_val:
            self.value = tensor_def.float_val
        elif tensor_def.string_val:
            self.value = tensor_def.string_val
        elif tensor_def.tensor_content:
            self.value = struct.unpack('i' * self.shape.size,
                                       tensor_def.tensor_content)
        else:
            raise RuntimeError('Unsupported tensor value: %s' % tensor_def)

    def __str__(self):
        return str(self._tensor_def)


class Node(object):
    def __init__(self, node_def):
        self._node_def = node_def
        self._inputs = []
        self._outputs = []
        self._value = None

        if self.hasattr('shape'):
            self.shape = Shape(self.attr('shape').shape)

    def __str__(self):
        return str(self._node_def)

    @property
    def name(self):
        return self._node_def.name

    @property
    def op(self):
        return self._node_def.op

    @property
    def ident(self):
        return self._node_def.name.replace('/', '_')

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def hasattr(self, key):
        return key in self._node_def.attr

    def attr(self, key):
        for k, v in self._node_def.attr.iteritems():
            if key == k:
                return v
        raise RuntimeError('no %s in %s' % (key, self))

    @property
    def dtype(self):
        return get_dtype_str(self.attr('dtype').type)

    @property
    def value(self):
        if self._value is None:
            self._value = Tensor(self.attr('value').tensor)
        return self._value


class Graph(object):
    def __init__(self, graph_filename, output_names):
        with open(graph_filename) as f:
            pbtxt = f.read()
        self._graph_def = tf.GraphDef()
        text_format.Merge(pbtxt, self._graph_def)

        self._node_map = {}
        for node_def in self._graph_def.node:
            self._node_map[node_def.name] = Node(node_def)

        self._output_nodes = []
        for output_name in output_names:
            self._output_nodes.append(self._node_map[output_name])

        self._all_nodes = []
        seen_nodes = set()
        def _create_edges(node):
            if node.name in seen_nodes:
                return
            seen_nodes.add(node.name)

            for input_name in node._node_def.input:
                input = self._node_map[input_name]
                node._inputs.append(input)
                input._outputs.append(node)
                _create_edges(input)

            self._all_nodes.append(node)

        for node in self._output_nodes:
            _create_edges(node)

    def show(self, out):
        def _show_node(out, node, depth):
            out.write('// ' + ' ' * depth + node.name + '\n')
            for input in node.inputs:
                _show_node(out, input, depth + 1)
        for node in self._output_nodes:
            _show_node(out, node, 0)

    def all_nodes(self):
        return self._all_nodes
