/*
 * vecadd_kernel.c — Pure AMDGPU kernel: vector-add + shader timestamps.
 *
 * Zero HIP dependency.  Uses only hardware intrinsics:
 *   __builtin_amdgcn_workgroup_id_x()   — SGPR
 *   __builtin_amdgcn_workitem_id_x()    — VGPR
 *   __builtin_amdgcn_dispatch_ptr()     — SGPR → AQL packet
 *   __builtin_amdgcn_s_memrealtime()    — GPU real-time clock
 *
 * Compile:
 *   clang --target=amdgcn-amd-amdhsa -mcpu=gfx950 -O2 -c -o vecadd_kernel.o vecadd_kernel.c
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

    const __attribute__((address_space(4))) uint16_t *dispatch =
        (const __attribute__((address_space(4))) uint16_t *)
            __builtin_amdgcn_dispatch_ptr();
    uint32_t wg_size = dispatch[2];  /* byte offset 4 = workgroup_size_x */

    /* Shader timestamp start — only lane 0 of each workgroup */
    if (wi_id == 0) {
#if defined(__gfx950__) || defined(__gfx942__) || defined(__gfx908__) || defined(__gfx900__)
        ts_shader[wg_id * 2] = __builtin_amdgcn_s_memrealtime();
#elif defined(__gfx1250__)
        ts_shader[wg_id * 2] = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
#endif
    }

    uint32_t global_id = wg_id * wg_size + wi_id;
    if (global_id < N) {
        C[global_id] = A[global_id] + B[global_id];
    }

    /* Barrier: ensure all lanes finished compute before end timestamp.
     * In AMDGPU ISA this maps to s_barrier. */
    __builtin_amdgcn_s_barrier();

    /* Shader timestamp end */
    if (wi_id == 0) {
#if defined(__gfx950__) || defined(__gfx942__) || defined(__gfx908__) || defined(__gfx900__)
        ts_shader[wg_id * 2 + 1] = __builtin_amdgcn_s_memrealtime();
#elif defined(__gfx1250__)
        ts_shader[wg_id * 2 + 1] = __builtin_amdgcn_s_sendmsg_rtnl(0x83);
#endif
    }
}
