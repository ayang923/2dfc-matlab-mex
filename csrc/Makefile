CC = cc
CFLAGS = -std=c11 -Wall -Wextra -O2 -Iinclude
LDFLAGS =

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework Accelerate
else ifdef MKLROOT
    ifneq ($(wildcard $(MKLROOT)/lib/intel64),)
        MKL_LIBDIR := $(MKLROOT)/lib/intel64
    else
        MKL_LIBDIR := $(MKLROOT)/lib
    endif
    CFLAGS += -DUSE_MKL_CBLAS -I$(MKLROOT)/include
    LDFLAGS += -L$(MKL_LIBDIR) -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
else
    LDFLAGS += -lopenblas -lm
endif

LIBS = num_linalg curve_eval patch_kernels fc_kernels grid_interp cartesian_kernels ie_kernels
OBJS = $(addprefix out/,$(LIBS:=.o))

all: bin/smoke_test

out/%.o: src/%.c
	@mkdir -p out
	$(CC) $(CFLAGS) -c $< -o $@

bin/smoke_test: $(OBJS) out/smoke_test.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

out/smoke_test.o: examples/smoke_test.c
	@mkdir -p out
	$(CC) $(CFLAGS) -c $< -o $@

test: bin/smoke_test
	./bin/smoke_test

clean:
	rm -rf out bin/smoke_test

.PHONY: all test clean
