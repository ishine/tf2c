from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tf2c import compiler
from tf2c import graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', required=True)
    parser.add_argument('--model', required=False)
    parser.add_argument('--outputs', required=True)
    parser.add_argument('--mode')
    args = parser.parse_args()

    g = graph.Graph(args.graph, args.outputs.split(','))
    if args.mode == 'show':
        g.show(sys.stdout)
        return
    c = compiler.Compiler(model=args.model)
    c.compile(g)
