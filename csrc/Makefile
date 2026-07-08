CC = cc
CFLAGS = -std=c11 -Wall -Wextra -O2 -Iinclude
LDFLAGS =

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    LDFLAGS += -framework Accelerate
else
    LDFLAGS += -lopenblas -lm
endif

LIBS = num_linalg curve_eval patch_kernels fc_kernels grid_interp cartesian_kernels
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
