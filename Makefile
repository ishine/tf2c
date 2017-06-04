PYTHON := python
CXXFLAGS := -std=c++11 -g -W -Wall -MMD -MP -O3 -I.

TF2C_PY := tf2c.py $(wildcard tf2c/*.py)

TESTS_PY := $(filter-out tests/__init__.py, $(wildcard tests/*.py))
TESTS_OUT := $(TESTS_PY:tests/%.py=out/%.out)
TESTS_C := $(TESTS_OUT:out/%.out=out/%.c)
TESTS_O := $(TESTS_OUT:out/%.out=out/%.o)
TESTS_EXE := $(TESTS_OUT:out/%.out=out/%.exe)

$(shell mkdir -p out)

all: $(TESTS_EXE)

$(TESTS_OUT): out/%.out: tests/%.py runtest.py
	$(PYTHON) runtest.py output $*

$(TESTS_C): out/%.c: out/%.out $(TF2C_PY)
	$(PYTHON) tf2c.py --graph out/$*.pbtxt --model out/$*.ckpt-0 --outputs result > $@.tmp && mv $@.tmp $@

$(TESTS_O): out/%.o: out/%.c
	$(CXX) -c $(CXXFLAGS) $< -o $@

out/main.o: tests/main.c
	$(CXX) -c $(CXXFLAGS) $< -o $@

out/tf2c.o: lib/tf2c.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(TESTS_EXE): out/%.exe: out/%.o out/main.o out/tf2c.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f out/*

.PHONY: all clean

-include out/*.d
