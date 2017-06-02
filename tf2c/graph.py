from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from google.protobuf import text_format


class Node(object):
    def __init__(self, node_def):
        self._node_def = node_def
        self._inputs = []
        self._outputs = []

    def name(self):
        return self._node_def.name

    def inputs(self):
        return self._inputs

    def outputs(self):
        return self._outputs


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

        seen_nodes = set()
        def _create_edges(node):
            if node.name() in seen_nodes:
                return
            seen_nodes.add(node.name())

            for input_name in node._node_def.input:
                input = self._node_map[input_name]
                node._inputs.append(input)
                input._outputs.append(node)
                _create_edges(input)

        for node in self._output_nodes:
            _create_edges(node)

    def show(self, out):
        def _show_node(out, node, depth):
            out.write('// ' + ' ' * depth + node.name() + '\n')
            for input in node.inputs():
                _show_node(out, input, depth + 1)
        for node in self._output_nodes:
            _show_node(out, node, 0)
