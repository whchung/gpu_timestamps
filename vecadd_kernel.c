/*
 * vecadd_kernel.c — Pure AMDGPU kernel: per-chiplet dual-clock timestamp measurement.
 *
 * Zero HIP dependency.  Uses only hardware intrinsics:
 *   __builtin_amdgcn_workgroup_id_x()   — SGPR
 *   __builtin_amdgcn_workitem_id_x()    — VGPR
 *   __builtin_amdgcn_s_memrealtime()    — GPU real-time clock (gfx9xx)
 *   __builtin_amdgcn_s_sendmsg_rtnl()   — GPU real-time clock (gfx1250)
 *   __builtin_amdgcn_s_memtime()        — Shader cycles counter (gfx9xx)
 *   __builtin_readcyclecounter()        — Shader cycles counter (gfx1250)
 *
 * Per-chiplet measurement protocol (8 workgroups, 1 per chiplet):
 *   Each workgroup (lane 0 only):
 *   1. Capture REALTIME #1 + SHADER_CYCLES #1 (store locally)
 *   2. Store those out, s_wait_kmcnt, barrier (no other work)
 *   3. Capture REALTIME #2 + SHADER_CYCLES #2
 *   4. Store those out
 *
 * Compile (gfx950):
 *   clang --target=amdgcn-amd-amdhsa -mcpu=gfx950 -O2 -c -o vecadd_kernel.o vecadd_kernel.c
 *   ld.lld -shared -o vecadd_kernel.co vecadd_kernel.o
 * Compile (gfx1250):
 *   clang --target=amdgcn-amd-amdhsa -mcpu=gfx1250 -O2 -c -o vecadd_kernel.o vecadd_kernel.c
 *   ld.lld -shared -o vecadd_kernel.co vecadd_kernel.o
 */

typedef unsigned int   uint32_t;
typedef unsigned short uint16_t;
typedef unsigned long  uint64_t;

__attribute__((amdgpu_kernel, visibility("default")))
void vecadd_timestamp(
    __attribute__((address_space(1))) const float *A,
    __attribute__((address_space(1))) const float *B,
    __attribute__((address_space(1)))       float *C,
    __attribute__((address_space(1)))    uint64_t *ts_shader,
    uint32_t N)
{
    uint32_t wg_id = __builtin_amdgcn_workgroup_id_x();
    uint32_t wi_id = __builtin_amdgcn_workitem_id_x();

    /* Each workgroup (lane 0 only) captures dual timestamps for its chiplet */
    if (wi_id == 0) {
        /* Measurement #1: REALTIME and SHADER_CYCLES (store locally) */
#if defined(__gfx950__) || defined(__gfx942__) || defined(__gfx908__) || defined(__gfx900__)
        uint64_t realtime_1 = __builtin_amdgcn_s_memrealtime();
        uint64_t cycles_1   = __builtin_amdgcn_s_memtime();
#elif defined(__gfx1250__)
        uint64_t realtime_1 = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
        uint64_t cycles_1   = __builtin_readcyclecounter();
#endif

        /* Store first pair (each workgroup uses offset wg_id * 4) */
        uint32_t base_offset = wg_id * 4;
        ts_shader[base_offset + 0] = realtime_1;
        ts_shader[base_offset + 1] = cycles_1;

        /* Wait for stores to complete */
#if defined(__gfx950__) || defined(__gfx942__) || defined(__gfx908__) || defined(__gfx900__)
        __builtin_amdgcn_s_waitcnt(0);
#elif defined(__gfx1250__)
        __asm__ volatile("s_wait_kmcnt 0x0" ::: "memory");
#endif

        /* Barrier (no other work) */
        __builtin_amdgcn_s_barrier();

        /* Measurement #2: REALTIME and SHADER_CYCLES */
#if defined(__gfx950__) || defined(__gfx942__) || defined(__gfx908__) || defined(__gfx900__)
        uint64_t realtime_2 = __builtin_amdgcn_s_memrealtime();
        uint64_t cycles_2   = __builtin_amdgcn_s_memtime();
#elif defined(__gfx1250__)
        uint64_t realtime_2 = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
        uint64_t cycles_2   = __builtin_readcyclecounter();
#endif

        /* Store second pair */
        ts_shader[base_offset + 2] = realtime_2;
        ts_shader[base_offset + 3] = cycles_2;
    }

    /* Optional: still do the vector add for functional testing */
    const __attribute__((address_space(4))) uint16_t *dispatch =
        (const __attribute__((address_space(4))) uint16_t *)
            __builtin_amdgcn_dispatch_ptr();
    uint32_t wg_size = dispatch[2];
    uint32_t global_id = wg_id * wg_size + wi_id;

    if (global_id < N) {
        C[global_id] = A[global_id] + B[global_id];
    }
}
