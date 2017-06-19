PYTHON := python
CXXFLAGS := -std=c++11 -g -W -Wall -MMD -MP -O3 -I. -mavx2 -mfma

TF2C_PY := tf2c.py $(wildcard tf2c/*.py)

TESTS_PY := $(filter-out tests/__init__.py, $(wildcard tests/*.py))
TESTS_OUT := $(TESTS_PY:tests/%.py=out/%.out)
TESTS_C := $(TESTS_OUT:out/%.out=out/%.c)
TESTS_O := $(TESTS_OUT:out/%.out=out/%.o)
TESTS_EXE := $(TESTS_OUT:out/%.out=out/%.exe)
TESTS_OK := $(TESTS_OUT:out/%.out=out/%.ok)

LIB_C := lib/tf2c.cc lib/mainutil.cc
LIB_O := $(LIB_C:lib/%.cc=out/%.o)

MISC_C := $(wildcard misc/*.cc)
MISC_O := $(MISC_C:misc/%.cc=out/misc_%.o)
MISC_EXE := $(MISC_O:out/%.o=out/%.exe)

$(shell mkdir -p out)

all: $(TESTS_OK) $(MISC_EXE)

$(TESTS_OUT): out/%.out: tests/%.py runtest.py
	$(PYTHON) runtest.py output $* > $@.tmp && mv $@.tmp $@

$(TESTS_C): out/%.c: out/%.out $(TF2C_PY)
	$(PYTHON) tf2c.py --graph out/$*.pbtxt --model out/$*.ckpt --outputs result > $@.tmp && mv $@.tmp $@

$(TESTS_O): out/%.o: out/%.c
	$(CXX) -c $(CXXFLAGS) $< -o $@

out/main.o: tests/main.c
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(LIB_O): out/%.o: lib/%.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(MISC_O): out/misc_%.o: misc/%.cc
	$(CXX) -c $(CXXFLAGS) $< -o $@

out/libtf2c.a: $(LIB_O)
	ar crus out/libtf2c.a $^

$(MISC_EXE): out/%.exe: out/%.o out/tf2c.o out/libtf2c.a
	$(CXX) $(CXXFLAGS) -lpthread $^ -o $@

$(TESTS_EXE): out/%.exe: out/%.o out/main.o out/libtf2c.a
	$(CXX) $(CXXFLAGS) -lpthread $^ -o $@

$(TESTS_OK): out/%.ok: out/%.exe
	$(PYTHON) runtest.py test $* && touch $@

clean:
	rm -f out/*

.PHONY: all clean

-include out/*.d
