# C/C++ mesh subprocesses. Built as separate binaries on purpose: in-process
# cgo calls into these kernels corrupt the heap on this box, so the math runs
# in its own process and Python passes CMESH binary dumps over argv paths.
#
#   make            -> ../../bin/meshproc ../../bin/fsdecimate
#   make clean
#
# cmesh.c MUST be compiled by gcc as C11: it relies on implicit void* casts
# from malloc/calloc/realloc, which g++ rejects (-fpermissive errors).
#
# NO -ffast-math. Simplify.h's vec3f::normalize() has its zero-guard commented
# out, so degenerate triangles yield NaN quadrics; the decimator's main loop
# (`if (t.err[3] > threshold) continue;`) depends on IEEE NaN compares being
# false, and calculate_error() can leave p_result uninitialized when all three
# errors are NaN. -ffinite-math-only changes which triangles get collapsed.

BIN      := ../../bin
CC       := gcc
CXX      := g++
CFLAGS   := -O2 -std=c11 -Wall -I.
CXXFLAGS := -O2 -std=c++17 -Wall -I. -Ivendor

all: $(BIN)/meshproc $(BIN)/fsdecimate

$(BIN)/meshproc: cxx/meshproc.cpp cmesh.o cmesh.h | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ cxx/meshproc.cpp cmesh.o

# fsdecimate is decimator-only (no C kernels) — kept for isolating the
# vendored Simplify.h against the pip wheel on the same input.
$(BIN)/fsdecimate: cxx/fsdecimate_cli.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

cmesh.o: cmesh.c cmesh.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -f cmesh.o $(BIN)/meshproc $(BIN)/fsdecimate

.PHONY: all clean
