# HSA Timestamp Validation Test

Minimal, zero-HIP test framework for validating CP (Command Processor) timestamp
correctness on AMD Instinct GPUs.

## Quick Start

```bash
# 1. Build (adjust GPU_ARCH for your target)
make GPU_ARCH=gfx950      # MI350X
# make GPU_ARCH=gfx1250   # MI450

# 2. Run
./hsa_vecadd 20 1024
```

Expected: Both Sequential and Burst modes PASS with zero CP timestamp overlaps.

> **Note:** An earlier version showed spurious overlap due to interleaving
> `hsa_amd_profiling_convert_tick_to_system_domain()` with
> `hsa_amd_profiling_get_dispatch_time()`. This has been fixed by splitting
> timestamp collection into two passes. See Section 7 of the report for details.

## Prerequisites

- AMD Instinct GPU (tested on MI350X / gfx950)
- ROCm 7.2.0+ with HSA profiling support
- ROCm clang: `/opt/rocm/lib/llvm/bin/clang`
- ROCm lld: `/opt/rocm/lib/llvm/bin/ld.lld`
- gcc
- `libhsa-runtime64` (comes with ROCm)

## Usage

```
./hsa_vecadd [num_dispatches] [num_elements]
```

- `num_dispatches`: number of kernel launches per mode (default: 10)
- `num_elements`: vector size (default: 1024)
- Exit code: 0 = no overlaps, 1 = overlaps detected

## Example Output

```
━━━ Burst dispatch (dispatch-all-then-wait, 20 runs) ━━━
  ── Burst overlap analysis ──
  PASS: no burst CP/shader timestamp overlaps.

╔══════════════════════════════════════════════════════════╗
║  RESULT: PASS — no timestamp overlaps detected         ║
╚══════════════════════════════════════════════════════════╝
```

## What It Does

1. Compiles a minimal `C = A + B` kernel using **only AMDGPU intrinsics** (no HIP)
2. Dispatches it via **raw HSA/AQL** (no HIP runtime)
3. Collects timestamps from 3 independent sources:
   - **Host**: `hsa_system_get_info(TIMESTAMP)` before/after dispatch
   - **CP**: `hsa_amd_profiling_get_dispatch_time()` hardware timestamps
   - **Shader**: `s_memrealtime` inside the kernel
4. Runs two dispatch patterns:
   - **Sequential**: dispatch → wait → dispatch → wait (baseline)
   - **Burst**: dispatch → dispatch → ... → wait (stress test)
5. Checks for CP timestamp overlaps (`prev_end > curr_start`) on the same queue
6. Cross-validates with shader timestamps to confirm serial execution

## Files

| File | Description |
|------|-------------|
| `vecadd_kernel.c` | GPU kernel — pure AMDGPU C with `s_memrealtime` timestamps |
| `hsa_vecadd.c` | Host program — HSA lifecycle, profiling, overlap detection |
| `Makefile` | Build: `clang --target=amdgcn` for kernel, `gcc` for host |
| `CP_TIMESTAMP_OVERLAP_REPORT.md` | Full analysis report (English) |
| `CP_TIMESTAMP_OVERLAP_REPORT_zh-TW.md` | Full analysis report (繁體中文) |

## Software Stack

```
vecadd_kernel.c  →  clang --target=amdgcn-amd-amdhsa  →  vecadd_kernel.co
hsa_vecadd.c     →  gcc + libhsa-runtime64             →  hsa_vecadd

Runtime path:  hsa_vecadd → libhsa-runtime64.so → KFD driver → CP firmware → GPU
```

**NOT involved:** HIP runtime, hipcc, PyTorch, Triton, rocprofv3, RTL tracer.

## Build Details

The kernel is compiled in two steps:

```bash
# 1. Compile to AMDGPU object
/opt/rocm/lib/llvm/bin/clang --target=amdgcn-amd-amdhsa -mcpu=gfx950 -O2 \
    -c -o vecadd_kernel.o vecadd_kernel.c

# 2. Link to shared ELF code object
/opt/rocm/lib/llvm/bin/ld.lld -shared -o vecadd_kernel.co vecadd_kernel.o
```

The host is compiled as plain C:

```bash
gcc -O2 -I/opt/rocm/include -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
    -o hsa_vecadd hsa_vecadd.c -lhsa-runtime64 -lm
```

## Tested Environment

- **GPU:** AMD Instinct MI350X (gfx950)
- **ROCm:** 7.2.0
- **OS:** Ubuntu 22.04, Linux 6.8.0-90-generic
- **Container:** `rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260420`
