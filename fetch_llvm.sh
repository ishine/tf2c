#!/bin/sh

n="$@"

TF_CPP_MIN_VLOG_LEVEL=2 TF_XLA_FLAGS='--xla_log_hlo_text=.*' \
  python runtest.py --jit bench $n 2>&1 | grep 'compiler_functor.cc:159'  | sed 's/.*159\]//' > $n.bc