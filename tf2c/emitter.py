from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys


class CodeEmitter(object):
    def __init__(self):
        self.out = sys.stdout
        self.indent = 0

    def emit_line(self, line):
        if line.startswith('}'):
            self.indent -= 2
        self.out.write(' ' * self.indent + line + '\n')
        if line.endswith('{'):
            self.indent += 2
        if line.startswith('}') and self.indent == 0:
            self.out.write('\n')
