ROCM     ?= /opt/rocm
CLANG    := $(ROCM)/lib/llvm/bin/clang
LLD      := $(ROCM)/lib/llvm/bin/ld.lld

# Auto-detect GPU_ARCH from rocm_agent_enumerator if available
ifeq ($(GPU_ARCH),)
  DETECTED_GPU := $(shell $(ROCM)/bin/rocm_agent_enumerator -t GPU 2>/dev/null | head -n1)
  ifneq ($(DETECTED_GPU),)
    GPU_ARCH := $(DETECTED_GPU)
  else
    GPU_ARCH := gfx950
  endif
else
  # Allow override from command line or environment
endif

HSA_INC  := $(ROCM)/include
HSA_LIB  := $(ROCM)/lib

all: vecadd_kernel.co hsa_vecadd

# Kernel: pure C → amdgcn object → shared ELF code object
# Uses only hardware intrinsics + s_memrealtime, zero HIP dependency.
vecadd_kernel.o: vecadd_kernel.c
	$(CLANG) --target=amdgcn-amd-amdhsa -mcpu=$(GPU_ARCH) -DGPU_ARCH=$(GPU_ARCH) -O2 -c -o $@ $<

vecadd_kernel.co: vecadd_kernel.o
	$(LLD) -shared -o $@ $<

# Host: pure C, links only libhsa-runtime64
hsa_vecadd: hsa_vecadd.c
	gcc -O2 -I$(HSA_INC) -L$(HSA_LIB) -Wl,-rpath,$(HSA_LIB) \
	    -o $@ $< -lhsa-runtime64 -lm

clean:
	rm -f vecadd_kernel.o vecadd_kernel.co hsa_vecadd

# Run rocprofv3 + convert DB → Perfetto JSON in one step
# Usage: make trace  OR  make trace DISPATCHES=40 ELEMENTS=4096
DISPATCHES ?= 20
ELEMENTS   ?= 1024
trace: hsa_vecadd vecadd_kernel.co
	./gen_perfetto_trace.sh $(DISPATCHES) $(ELEMENTS)

.PHONY: all clean trace
