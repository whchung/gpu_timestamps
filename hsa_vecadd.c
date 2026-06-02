/*
 * hsa_vecadd.c — Bare-Metal HSA Dual-Clock Timestamp Measurement
 *
 * Pure C.  Links only against libhsa-runtime64.  Zero HIP dependency.
 * The kernel is compiled with clang --target=amdgcn-amd-amdhsa,
 * using only hardware intrinsics.
 *
 * Single-wave measurement protocol:
 *   1. Shader captures s_memrealtime + s_memtime (store locally)
 *   2. Store out, waitcnt, barrier (no other work)
 *   3. Shader captures s_memrealtime + s_memtime again
 *   4. Store out
 *
 * Collects timestamps from:
 *   1. HSA system clock (host-side, before/after dispatch)
 *   2. CP timestamps (read directly from amd_signal_t structure)
 *   3. Shader dual clocks (s_memrealtime + s_memtime, before/after barrier)
 *
 * Two dispatch patterns:
 *   - Sequential: dispatch-wait-dispatch-wait (baseline)
 *   - Burst:      dispatch-dispatch-...-wait  (inference-like pattern)
 *
 * Build:  gcc -O2 -I/opt/rocm/include -L/opt/rocm/lib
 *             -Wl,-rpath,/opt/rocm/lib -o hsa_vecadd hsa_vecadd.c
 *             -lhsa-runtime64 -lm
 * Usage:  ./hsa_vecadd [num_dispatches] [num_elements]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <sched.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

/* ── AMD signal structure (for direct timestamp access) ─────────── */

typedef struct {
    int32_t  kind;
    union {
        volatile int32_t value;
        volatile uint32_t raw32[2];
        volatile uint64_t raw64;
    };
    uint64_t event_mailbox_ptr;
    uint32_t event_id;
    uint32_t reserved1;
    uint64_t start_ts;
    uint64_t end_ts;
    union {
        void *queue_ptr;
        uint64_t raw64_1;
    };
    uint32_t reserved2[2];
} amd_signal_t;

/* ── Configuration ──────────────────────────────────────────────── */

#define WORKGROUP_SIZE   256
#define NUM_WORKGROUPS   8
#define DEFAULT_N        (WORKGROUP_SIZE * NUM_WORKGROUPS)
#define DEFAULT_RUNS     10
#define CO_FILE          "vecadd_kernel.co"
#define KERNEL_SYM       "vecadd_timestamp.kd"

/* ── Error handling ─────────────────────────────────────────────── */

#define HSA_CHECK(msg, call) do {                                     \
    hsa_status_t _s = (call);                                         \
    if (_s != HSA_STATUS_SUCCESS) {                                   \
        const char *_str = NULL;                                      \
        hsa_status_string(_s, &_str);                                 \
        fprintf(stderr, "FATAL %s [%s:%d]: %s (0x%x)\n",             \
                msg, __FILE__, __LINE__,                              \
                _str ? _str : "unknown", (unsigned)_s);               \
        exit(1);                                                      \
    }                                                                 \
} while (0)

/* ── Timestamp record for one dispatch ──────────────────────────── */

typedef struct {
    uint64_t host_pre;
    uint64_t host_post;

    uint64_t cp_start;
    uint64_t cp_end;

    /* Shader dual-clock measurements (single wave) */
    uint64_t shader_realtime_1;
    uint64_t shader_cycles_1;
    uint64_t shader_realtime_2;
    uint64_t shader_cycles_2;
} ts_record_t;

/* ── SDMA timestamp record ──────────────────────────────────────── */

typedef struct {
    uint64_t start;
    uint64_t end;
} sdma_ts_t;

/* ── Kernel info from code object ───────────────────────────────── */

typedef struct {
    uint64_t kernel_object;
    uint32_t kernarg_segment_size;
    uint32_t group_segment_size;
    uint32_t private_segment_size;
} kernel_info_t;

/*
 * Kernarg layout for our pure AMDGPU kernel (explicit args only):
 *   offset  0: const float* A        (8 bytes)
 *   offset  8: const float* B        (8 bytes)
 *   offset 16: float* C              (8 bytes)
 *   offset 24: uint64_t* ts_shader   (8 bytes)
 *   offset 32: uint32_t N            (4 bytes)
 *   total: 36 bytes (no HIP implicit arguments)
 *
 * ts_shader buffer layout (4 uint64_t values):
 *   [0] = realtime_1
 *   [1] = cycles_1
 *   [2] = realtime_2
 *   [3] = cycles_2
 */

/* ── Global state ───────────────────────────────────────────────── */

static hsa_agent_t              g_gpu;
static hsa_agent_t              g_cpu;
static hsa_amd_memory_pool_t    g_fine_pool;
static hsa_amd_memory_pool_t    g_kernarg_pool;
static hsa_amd_memory_pool_t    g_vram_pool;
static int                      g_have_vram = 0;
static uint64_t                 g_sys_freq  = 0;

/* ── Agent discovery callbacks ──────────────────────────────────── */

static hsa_status_t cb_find_gpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t *)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t cb_find_cpu(hsa_agent_t agent, void *data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_CPU) {
        *(hsa_agent_t *)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

/* ── Memory pool discovery callbacks ────────────────────────────── */

static hsa_status_t cb_find_kernarg(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

    uint32_t flags;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT) {
        *(hsa_amd_memory_pool_t *)data = pool;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t cb_find_fine(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

    uint32_t flags;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
        bool alloc = false;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
        if (alloc) {
            *(hsa_amd_memory_pool_t *)data = pool;
            return HSA_STATUS_INFO_BREAK;
        }
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t cb_find_vram(hsa_amd_memory_pool_t pool, void *data) {
    hsa_amd_segment_t seg;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &seg);
    if (seg != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

    uint32_t flags;
    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
    if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
        bool alloc = false;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);
        if (alloc) {
            *(hsa_amd_memory_pool_t *)data = pool;
            return HSA_STATUS_INFO_BREAK;
        }
    }
    return HSA_STATUS_SUCCESS;
}

/* ── HSA system timestamp ───────────────────────────────────────── */

static uint64_t hsa_timestamp_now(void) {
    uint64_t ts = 0;
    hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ts);
    return ts;
}

/* ── Queue error callback ───────────────────────────────────────── */

static void queue_error_cb(hsa_status_t status, hsa_queue_t *q, void *data) {
    (void)data;
    const char *str = NULL;
    hsa_status_string(status, &str);
    fprintf(stderr, "Queue %p error: %s\n", (void *)q, str ? str : "?");
}

/* ── AQL packet header helper ───────────────────────────────────── */

static inline void packet_store_release(uint32_t *pkt,
                                         uint16_t header, uint16_t setup) {
    __atomic_store_n(pkt, header | ((uint32_t)setup << 16), __ATOMIC_RELEASE);
}

/* ── Init HSA runtime ───────────────────────────────────────────── */

static void init_hsa(void) {
    HSA_CHECK("hsa_init", hsa_init());

    memset(&g_gpu, 0, sizeof(g_gpu));
    memset(&g_cpu, 0, sizeof(g_cpu));

    hsa_status_t s;
    s = hsa_iterate_agents(cb_find_gpu, &g_gpu);
    if (s != HSA_STATUS_INFO_BREAK) {
        fprintf(stderr, "No GPU agent found\n");
        exit(1);
    }
    s = hsa_iterate_agents(cb_find_cpu, &g_cpu);
    if (s != HSA_STATUS_INFO_BREAK) {
        fprintf(stderr, "No CPU agent found\n");
        exit(1);
    }

    char gpu_name[64] = {0};
    hsa_agent_get_info(g_gpu, HSA_AGENT_INFO_NAME, gpu_name);
    uint32_t gpu_node = 0;
    hsa_agent_get_info(g_gpu, HSA_AGENT_INFO_NODE, &gpu_node);
    printf("GPU agent: %s  (node %u, handle 0x%lx)\n",
           gpu_name, gpu_node, g_gpu.handle);

    /* Find memory pools */
    memset(&g_kernarg_pool, 0, sizeof(g_kernarg_pool));
    memset(&g_fine_pool, 0, sizeof(g_fine_pool));
    memset(&g_vram_pool, 0, sizeof(g_vram_pool));

    /* Try CPU agent first for kernarg and fine-grained pools */
    if (hsa_amd_agent_iterate_memory_pools(g_cpu, cb_find_kernarg, &g_kernarg_pool)
            != HSA_STATUS_INFO_BREAK) {
        /* Fall back to GPU agent */
        if (hsa_amd_agent_iterate_memory_pools(g_gpu, cb_find_kernarg, &g_kernarg_pool)
                != HSA_STATUS_INFO_BREAK) {
            fprintf(stderr, "No kernarg pool found\n");
            exit(1);
        }
    }

    if (hsa_amd_agent_iterate_memory_pools(g_cpu, cb_find_fine, &g_fine_pool)
            != HSA_STATUS_INFO_BREAK) {
        if (hsa_amd_agent_iterate_memory_pools(g_gpu, cb_find_fine, &g_fine_pool)
                != HSA_STATUS_INFO_BREAK) {
            fprintf(stderr, "No fine-grained pool found\n");
            exit(1);
        }
    }

    g_have_vram = (hsa_amd_agent_iterate_memory_pools(g_gpu, cb_find_vram, &g_vram_pool)
                   == HSA_STATUS_INFO_BREAK);

    HSA_CHECK("sys_freq",
              hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &g_sys_freq));

    // HARD code g_sys_freq as 100MHz
    g_sys_freq = 100.0 * 1e6;

    printf("Memory pools: kernarg=OK  fine-grained=OK  VRAM=%s\n",
           g_have_vram ? "OK" : "N/A");
    printf("System timestamp frequency: %lu Hz (%.1f MHz)\n",
           g_sys_freq, g_sys_freq / 1e6);
}

/* ── Load code object ───────────────────────────────────────────── */

static void load_code_object(const char *path, const char *symbol_name,
                              kernel_info_t *ki) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open code object: %s\n", path);
        exit(1);
    }

    hsa_code_object_reader_t reader;
    HSA_CHECK("co_reader", hsa_code_object_reader_create_from_file(fd, &reader));

    hsa_executable_t exec;
    HSA_CHECK("exec_create",
              hsa_executable_create_alt(HSA_PROFILE_FULL,
                                        HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                        NULL, &exec));

    HSA_CHECK("load_co",
              hsa_executable_load_agent_code_object(exec, g_gpu, reader, NULL, NULL));
    HSA_CHECK("freeze", hsa_executable_freeze(exec, NULL));

    hsa_executable_symbol_t sym;
    HSA_CHECK("get_symbol",
              hsa_executable_get_symbol_by_name(exec, symbol_name, &g_gpu, &sym));

    HSA_CHECK("kernel_obj",
              hsa_executable_symbol_get_info(sym,
                  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ki->kernel_object));
    HSA_CHECK("kernarg_sz",
              hsa_executable_symbol_get_info(sym,
                  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                  &ki->kernarg_segment_size));
    HSA_CHECK("group_sz",
              hsa_executable_symbol_get_info(sym,
                  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                  &ki->group_segment_size));
    HSA_CHECK("priv_sz",
              hsa_executable_symbol_get_info(sym,
                  HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                  &ki->private_segment_size));

    printf("Kernel loaded: object=0x%lx  kernarg=%u  group=%u  private=%u\n",
           (unsigned long)ki->kernel_object,
           ki->kernarg_segment_size,
           ki->group_segment_size,
           ki->private_segment_size);

    close(fd);
}

/* ── Fill kernarg for our pure AMDGPU kernel ────────────────────── */

static void fill_kernarg(void *kernarg, uint32_t kernarg_size,
                          const float *A, const float *B, float *C,
                          uint64_t *ts_shader, uint32_t N) {
    memset(kernarg, 0, kernarg_size);
    char *kp = (char *)kernarg;

    *(const float **)(kp + 0)  = A;
    *(const float **)(kp + 8)  = B;
    *(float **)      (kp + 16) = C;
    *(uint64_t **)   (kp + 24) = ts_shader;
    *(uint32_t *)    (kp + 32) = N;
}

/* ── Dispatch one kernel with queue-full protection ─────────────── */

static void dispatch_one(hsa_queue_t *queue, const kernel_info_t *ki,
                          void *kernarg, hsa_signal_t signal,
                          uint32_t num_groups) {
    uint64_t index = hsa_queue_add_write_index_screlease(queue, 1);

    const uint32_t sw_queue_size = queue->size - 1;
    while ((index - hsa_queue_load_read_index_scacquire(queue)) >= sw_queue_size) {
        sched_yield();
    }

    uint32_t slot = (uint32_t)(index & (queue->size - 1));
    hsa_kernel_dispatch_packet_t *pkt =
        &((hsa_kernel_dispatch_packet_t *)(queue->base_address))[slot];

    memset(pkt, 0, sizeof(*pkt));
    pkt->workgroup_size_x     = WORKGROUP_SIZE;
    pkt->workgroup_size_y     = 1;
    pkt->workgroup_size_z     = 1;
    pkt->grid_size_x          = num_groups * WORKGROUP_SIZE;
    pkt->grid_size_y          = 1;
    pkt->grid_size_z          = 1;
    pkt->kernel_object        = ki->kernel_object;
    pkt->kernarg_address      = kernarg;
    pkt->group_segment_size   = ki->group_segment_size;
    pkt->private_segment_size = ki->private_segment_size;
    pkt->completion_signal    = signal;

    uint16_t header =
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
      | (1 << HSA_PACKET_HEADER_BARRIER)
      | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE)
      | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
    uint16_t setup = 1;  /* 1D */

    packet_store_release((uint32_t *)pkt, header, setup);
    hsa_signal_store_screlease(queue->doorbell_signal, (hsa_signal_value_t)index);
}

/* ── Sequential dispatch: dispatch-wait-dispatch-wait ───────────── */

static void run_sequential(hsa_queue_t *queue, const kernel_info_t *ki,
                            float *A, float *B, float *C,
                            uint32_t N, uint32_t num_groups,
                            int num_runs, ts_record_t *records) {
    size_t ts_bytes = 4 * sizeof(uint64_t);  /* 4 values: realtime1, cycles1, realtime2, cycles2 */

    for (int r = 0; r < num_runs; r++) {
        /* Per-dispatch shader timestamp buffer */
        uint64_t *ts_shader = NULL;
        HSA_CHECK("alloc ts",
                  hsa_amd_memory_pool_allocate(g_fine_pool, ts_bytes, 0,
                                               (void **)&ts_shader));
        HSA_CHECK("access ts",
                  hsa_amd_agents_allow_access(1, &g_gpu, NULL, ts_shader));
        memset(ts_shader, 0, ts_bytes);

        /* Allocate kernarg */
        void *kernarg = NULL;
        HSA_CHECK("alloc ka",
                  hsa_amd_memory_pool_allocate(g_kernarg_pool,
                                               ki->kernarg_segment_size, 0,
                                               &kernarg));
        HSA_CHECK("access ka",
                  hsa_amd_agents_allow_access(1, &g_gpu, NULL, kernarg));
        fill_kernarg(kernarg, ki->kernarg_segment_size, A, B, C, ts_shader, N);

        /* Signal */
        hsa_signal_t signal;
        HSA_CHECK("signal", hsa_signal_create(1, 0, NULL, &signal));

        /* Timestamp point 1: host pre */
        records[r].host_pre = hsa_timestamp_now();

        /* Dispatch */
        dispatch_one(queue, ki, kernarg, signal, num_groups);

        /* Wait */
        hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_EQ, 0,
                                  UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

        /* Timestamp point 1 cont: host post */
        records[r].host_post = hsa_timestamp_now();

        /* Timestamp point 2: CP timestamps from signal structure */
        amd_signal_t *amd_sig = (amd_signal_t *)signal.handle;
        records[r].cp_start = amd_sig->start_ts;
        records[r].cp_end   = amd_sig->end_ts;

        /* Timestamp point 3: shader dual-clock measurements */
        records[r].shader_realtime_1 = ts_shader[0];
        records[r].shader_cycles_1   = ts_shader[1];
        records[r].shader_realtime_2 = ts_shader[2];
        records[r].shader_cycles_2   = ts_shader[3];

        hsa_signal_destroy(signal);
        hsa_amd_memory_pool_free(kernarg);
        hsa_amd_memory_pool_free(ts_shader);
    }
}

/* ── Burst dispatch: dispatch-dispatch-...-wait ─────────────────── */

static void run_burst(hsa_queue_t *queue, const kernel_info_t *ki,
                       float *A, float *B, float *C,
                       uint32_t N, uint32_t num_groups,
                       int num_runs, ts_record_t *records) {
    size_t ts_bytes = 4 * sizeof(uint64_t);  /* 4 values: realtime1, cycles1, realtime2, cycles2 */

    hsa_signal_t *signals  = calloc(num_runs, sizeof(hsa_signal_t));
    void        **kernargs = calloc(num_runs, sizeof(void *));
    uint64_t    **ts_bufs  = calloc(num_runs, sizeof(uint64_t *));

    /* Allocate all resources up front */
    for (int r = 0; r < num_runs; r++) {
        HSA_CHECK("alloc ts_burst",
                  hsa_amd_memory_pool_allocate(g_fine_pool, ts_bytes, 0,
                                               (void **)&ts_bufs[r]));
        HSA_CHECK("access ts_burst",
                  hsa_amd_agents_allow_access(1, &g_gpu, NULL, ts_bufs[r]));
        memset(ts_bufs[r], 0, ts_bytes);

        HSA_CHECK("alloc ka_burst",
                  hsa_amd_memory_pool_allocate(g_kernarg_pool,
                                               ki->kernarg_segment_size, 0,
                                               &kernargs[r]));
        HSA_CHECK("access ka_burst",
                  hsa_amd_agents_allow_access(1, &g_gpu, NULL, kernargs[r]));
        fill_kernarg(kernargs[r], ki->kernarg_segment_size,
                     A, B, C, ts_bufs[r], N);

        HSA_CHECK("signal_burst",
                  hsa_signal_create(1, 0, NULL, &signals[r]));
    }

    /* Dispatch all kernels back-to-back */
    uint64_t burst_pre = hsa_timestamp_now();

    for (int r = 0; r < num_runs; r++) {
        dispatch_one(queue, ki, kernargs[r], signals[r], num_groups);
    }

    /* Wait for last signal (all on same queue → serial execution) */
    hsa_signal_wait_scacquire(signals[num_runs - 1], HSA_SIGNAL_CONDITION_EQ, 0,
                              UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    uint64_t burst_post = hsa_timestamp_now();
    printf("  Total burst wall time: %.1f us\n",
           (double)(burst_post - burst_pre) * 1e6 / (double)g_sys_freq);

    /* Read timestamps from signal structures */
    for (int r = 0; r < num_runs; r++) {
        records[r].host_pre  = burst_pre;
        records[r].host_post = burst_post;

        /* CP timestamps from signal structure */
        amd_signal_t *amd_sig = (amd_signal_t *)signals[r].handle;
        records[r].cp_start = amd_sig->start_ts;
        records[r].cp_end   = amd_sig->end_ts;

        /* Shader dual-clock measurements */
        records[r].shader_realtime_1 = ts_bufs[r][0];
        records[r].shader_cycles_1   = ts_bufs[r][1];
        records[r].shader_realtime_2 = ts_bufs[r][2];
        records[r].shader_cycles_2   = ts_bufs[r][3];
    }

    /* Cleanup */
    for (int r = 0; r < num_runs; r++) {
        hsa_signal_destroy(signals[r]);
        hsa_amd_memory_pool_free(kernargs[r]);
        hsa_amd_memory_pool_free(ts_bufs[r]);
    }
    free(signals);
    free(kernargs);
    free(ts_bufs);
}

/* ── SDMA copy with profiled timestamps ─────────────────────────── */

static void sdma_copy_profiled(void *dst, hsa_agent_t dst_agent,
                                const void *src, hsa_agent_t src_agent,
                                size_t size, sdma_ts_t *ts) {
    hsa_signal_t signal;
    HSA_CHECK("sdma_sig", hsa_signal_create(1, 0, NULL, &signal));

    HSA_CHECK("async_copy",
              hsa_amd_memory_async_copy(dst, dst_agent, src, src_agent,
                                        size, 0, NULL, signal));

    hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                              UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    hsa_amd_profiling_async_copy_time_t ct;
    HSA_CHECK("sdma_time", hsa_amd_profiling_get_async_copy_time(signal, &ct));
    ts->start = ct.start;
    ts->end   = ct.end;

    hsa_signal_destroy(signal);
}

/* ── Print records ──────────────────────────────────────────────── */

static void print_records(const ts_record_t *recs, int n) {
    double to_us = 1e6 / (double)g_sys_freq;

    printf("  %4s  %20s  %20s  %10s  %15s  %15s  %15s  %15s\n",
           "Run", "CP_start", "CP_end", "CP(us)",
           "RT_delta", "CYC_delta", "RT1", "CYC1");

    for (int i = 0; i < n; i++) {
        const ts_record_t *r = &recs[i];
        double cp_dur = (double)(r->cp_end - r->cp_start) * to_us;
        uint64_t rt_delta = r->shader_realtime_2 - r->shader_realtime_1;
        uint64_t cyc_delta = r->shader_cycles_2 - r->shader_cycles_1;

        printf("  %4d  %20lu  %20lu  %10.1f  %15lu  %15lu  %15lu  %15lu\n",
               i, r->cp_start, r->cp_end, cp_dur,
               rt_delta, cyc_delta,
               r->shader_realtime_1, r->shader_cycles_1);
    }
}

/* ── Print detailed dual-clock measurements ─────────────────────── */

static void print_dual_clock_detail(const ts_record_t *rec, int run_num) {
    printf("\n━━━ Run %d: Dual-Clock Measurement Detail ━━━\n", run_num);
    printf("  CP timestamps:\n");
    printf("    start = %lu\n", rec->cp_start);
    printf("    end   = %lu\n", rec->cp_end);
    printf("    delta = %lu ticks\n\n", rec->cp_end - rec->cp_start);

    printf("  Shader timestamps (single wave):\n");
    printf("    Measurement #1 (before barrier):\n");
    printf("      s_memrealtime = %lu\n", rec->shader_realtime_1);
    printf("      s_memtime     = %lu\n", rec->shader_cycles_1);
    printf("    Measurement #2 (after barrier):\n");
    printf("      s_memrealtime = %lu\n", rec->shader_realtime_2);
    printf("      s_memtime     = %lu\n\n", rec->shader_cycles_2);

    uint64_t rt_delta = rec->shader_realtime_2 - rec->shader_realtime_1;
    uint64_t cyc_delta = rec->shader_cycles_2 - rec->shader_cycles_1;

    printf("  Delta (measurement #2 - #1):\n");
    printf("    s_memrealtime delta = %lu ticks\n", rt_delta);
    printf("    s_memtime delta     = %lu cycles\n", cyc_delta);

    if (cyc_delta > 0 && rt_delta > 0) {
        double ratio = (double)cyc_delta / (double)rt_delta;
        printf("    cycles/realtime ratio = %.6f\n", ratio);
    }
}

/* ── Overlap detection ──────────────────────────────────────────── */

static int check_overlaps(const char *label, const ts_record_t *recs, int n) {
    double to_ns = 1e9 / (double)g_sys_freq;
    int overlaps = 0;

    for (int i = 1; i < n; i++) {
        const ts_record_t *prev = &recs[i - 1];
        const ts_record_t *curr = &recs[i];

        if (prev->cp_end > curr->cp_start) {
            double gap_ns = (double)((int64_t)curr->cp_start -
                                     (int64_t)prev->cp_end) * to_ns;
            printf("  !! CP OVERLAP %s [%d->%d]: prev_end=%lu > curr_start=%lu"
                   "  (%.1f ns)\n",
                   label, i - 1, i, prev->cp_end, curr->cp_start, gap_ns);
            overlaps++;
        }

        if (prev->shader_realtime_2 > curr->shader_realtime_1 &&
            prev->shader_realtime_1 != 0 && curr->shader_realtime_1 != 0) {
            printf("  !! SHADER REALTIME OVERLAP %s [%d->%d]: prev_rt2=%lu > "
                   "curr_rt1=%lu\n",
                   label, i - 1, i, prev->shader_realtime_2, curr->shader_realtime_1);
            overlaps++;
        }
    }
    return overlaps;
}

/* ── Ordering validation ────────────────────────────────────────── */

static int check_ordering(const ts_record_t *recs, int n) {
    int fails = 0;
    for (int r = 0; r < n; r++) {
        const ts_record_t *rc = &recs[r];
        int ok = 1;

        if (rc->cp_start >= rc->cp_end) {
            printf("  [%d] cp_start (%lu) >= cp_end (%lu)\n",
                   r, rc->cp_start, rc->cp_end);
            ok = 0;
        }
        if (rc->shader_realtime_1 != 0 && rc->shader_realtime_1 >= rc->shader_realtime_2) {
            printf("  [%d] shader_realtime_1 (%lu) >= shader_realtime_2 (%lu)\n",
                   r, rc->shader_realtime_1, rc->shader_realtime_2);
            ok = 0;
        }
        if (rc->shader_cycles_1 != 0 && rc->shader_cycles_1 >= rc->shader_cycles_2) {
            printf("  [%d] shader_cycles_1 (%lu) >= shader_cycles_2 (%lu)\n",
                   r, rc->shader_cycles_1, rc->shader_cycles_2);
            ok = 0;
        }
        if (!ok) fails++;
    }
    return fails;
}

/* ═══════════════════════════════════════════════════════════════════ */
/*                              MAIN                                  */
/* ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    int num_runs = DEFAULT_RUNS;
    if (argc > 1) num_runs = atoi(argv[1]);
    if (num_runs < 1) num_runs = 1;

    /* Force exactly 8 workgroups */
    uint32_t num_groups = NUM_WORKGROUPS;
    int N = WORKGROUP_SIZE * num_groups;

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Bare-Metal HSA Timestamp Validation Test (Zero HIP)   ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    printf("N=%d  workgroup_size=%d  num_groups=%u  dispatches=%d\n\n",
           N, WORKGROUP_SIZE, num_groups, num_runs);

    /* ── 1. Init HSA ── */
    init_hsa();

    /* ── 2. Create queue with profiling enabled ── */
    uint32_t queue_size = 0;
    HSA_CHECK("queue_max",
              hsa_agent_get_info(g_gpu, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));
    if (queue_size > 1024) queue_size = 1024;

    hsa_queue_t *queue = NULL;
    HSA_CHECK("queue_create",
              hsa_queue_create(g_gpu, queue_size, HSA_QUEUE_TYPE_SINGLE,
                               queue_error_cb, NULL,
                               UINT32_MAX, UINT32_MAX, &queue));
    HSA_CHECK("profiling_on",
              hsa_amd_profiling_set_profiler_enabled(queue, 1));

    printf("Queue created: %p  size=%u  profiling=ON\n\n", (void *)queue, queue_size);

    /* ── 3. Load code object ── */
    kernel_info_t ki;
    load_code_object(CO_FILE, KERNEL_SYM, &ki);

    /* ── 4. Allocate data buffers (fine-grained) ── */
    size_t data_bytes = (size_t)N * sizeof(float);
    float *A = NULL, *B = NULL, *C = NULL;

    HSA_CHECK("alloc A",
              hsa_amd_memory_pool_allocate(g_fine_pool, data_bytes, 0, (void **)&A));
    HSA_CHECK("alloc B",
              hsa_amd_memory_pool_allocate(g_fine_pool, data_bytes, 0, (void **)&B));
    HSA_CHECK("alloc C",
              hsa_amd_memory_pool_allocate(g_fine_pool, data_bytes, 0, (void **)&C));

    HSA_CHECK("access A", hsa_amd_agents_allow_access(1, &g_gpu, NULL, A));
    HSA_CHECK("access B", hsa_amd_agents_allow_access(1, &g_gpu, NULL, B));
    HSA_CHECK("access C", hsa_amd_agents_allow_access(1, &g_gpu, NULL, C));

    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(N - i);
    }
    printf("Buffers: A=%p  B=%p  C=%p  (%zu bytes each)\n\n", 
           (void *)A, (void *)B, (void *)C, data_bytes);

    int total_overlaps = 0;

    /* ── 5. Warm-up dispatch ── */
    printf("━━━ Warm-up dispatch ━━━\n");
    {
        ts_record_t dummy;
        memset(C, 0, data_bytes);
        run_sequential(queue, &ki, A, B, C, N, num_groups, 1, &dummy);

        int ok = 1;
        for (int i = 0; i < N; i++) {
            float expected = (float)i + (float)(N - i);
            if (fabsf(C[i] - expected) > 1e-5f) {
                printf("  MISMATCH at [%d]: got %f, expected %f\n",
                       i, C[i], expected);
                ok = 0;
                if (i >= 10) break;
            }
        }
        printf("  Correctness: %s\n\n", ok ? "PASS" : "FAIL");
        if (!ok) {
            fprintf(stderr, "Kernel produced wrong results — aborting.\n");
            exit(1);
        }
    }

    /* ── 6. Sequential dispatches (measured) ── */
    printf("━━━ Sequential dispatch (dispatch-wait pattern, %d runs) ━━━\n",
           num_runs);
    ts_record_t *seq_recs = calloc(num_runs, sizeof(ts_record_t));
    memset(C, 0, data_bytes);
    run_sequential(queue, &ki, A, B, C, N, num_groups, num_runs, seq_recs);
    print_records(seq_recs, num_runs);

    /* Print dual-clock detail for first run */
    if (num_runs > 0) {
        print_dual_clock_detail(&seq_recs[1], 1);
    }

    printf("\n  ── Sequential overlap analysis ──\n");
    int seq_overlaps = check_overlaps("SEQ", seq_recs, num_runs);
    if (seq_overlaps == 0)
        printf("  PASS: no sequential CP/shader timestamp overlaps.\n");
    else
        printf("  FAIL: %d sequential overlapping pairs found.\n", seq_overlaps);
    total_overlaps += seq_overlaps;

    printf("\n  ── Sequential ordering check ──\n");
    int ord_fails = check_ordering(seq_recs, num_runs);
    if (ord_fails == 0)
        printf("  PASS: all %d dispatches have correct timestamp ordering.\n",
               num_runs);
    else
        printf("  FAIL: %d/%d ordering violations.\n", ord_fails, num_runs);

    /* ── 7. Burst dispatches ── */
    printf("\n━━━ Burst dispatch (dispatch-all-then-wait, %d runs) ━━━\n",
           num_runs);
    ts_record_t *burst_recs = calloc(num_runs, sizeof(ts_record_t));
    memset(C, 0, data_bytes);
    run_burst(queue, &ki, A, B, C, N, num_groups, num_runs, burst_recs);
    print_records(burst_recs, num_runs);

    /* Print dual-clock detail for first run */
    if (num_runs > 0) {
        print_dual_clock_detail(&burst_recs[1], 1);
    }

    printf("\n  ── Burst overlap analysis ──\n");
    int burst_overlaps = check_overlaps("BURST", burst_recs, num_runs);
    if (burst_overlaps == 0)
        printf("  PASS: no burst CP/shader timestamp overlaps.\n");
    else
        printf("  FAIL: %d burst overlapping pairs found.\n", burst_overlaps);
    total_overlaps += burst_overlaps;

    printf("\n  ── Burst ordering check ──\n");
    int burst_ord = check_ordering(burst_recs, num_runs);
    if (burst_ord == 0)
        printf("  PASS: all %d burst dispatches have correct ordering.\n",
               num_runs);
    else
        printf("  FAIL: %d/%d burst ordering violations.\n",
               burst_ord, num_runs);

    /* ── 8. Dual-clock analysis ── */
    if (num_runs > 0 && seq_recs[0].shader_realtime_1 != 0) {
        printf("\n━━━ Dual-clock frequency analysis ━━━\n");

        /* Compute average deltas across all sequential runs */
        uint64_t sum_rt_delta = 0, sum_cyc_delta = 0;
        int valid_runs = 0;

        for (int i = 0; i < num_runs; i++) {
            uint64_t rt_d = seq_recs[i].shader_realtime_2 - seq_recs[i].shader_realtime_1;
            uint64_t cyc_d = seq_recs[i].shader_cycles_2 - seq_recs[i].shader_cycles_1;
            if (rt_d > 0 && cyc_d > 0) {
                sum_rt_delta += rt_d;
                sum_cyc_delta += cyc_d;
                valid_runs++;
            }
        }

        if (valid_runs > 0) {
            double avg_rt = (double)sum_rt_delta / (double)valid_runs;
            double avg_cyc = (double)sum_cyc_delta / (double)valid_runs;
            double ratio = avg_cyc / avg_rt;

            printf("  Average barrier overhead (across %d runs):\n", valid_runs);
            printf("    s_memrealtime delta = %.1f ticks\n", avg_rt);
            printf("    s_memtime delta     = %.1f cycles\n", avg_cyc);
            printf("    cycles/realtime ratio = %.6f\n", ratio);

            /* Estimate frequencies assuming s_memrealtime uses system clock */
            double rt_ns = avg_rt * 1e9 / (double)g_sys_freq;
            if (rt_ns > 0) {
                double shader_freq_mhz = avg_cyc / rt_ns * 1e3;
                printf("    Estimated shader clock: %.1f MHz\n", shader_freq_mhz);
                printf("    (assuming s_memrealtime runs at %.1f MHz)\n",
                       (double)g_sys_freq / 1e6);
            }
        }
    }

    /* ── 9. Final summary ── */
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    if (total_overlaps == 0) {
        printf("║  RESULT: PASS — no timestamp overlaps detected         ║\n");
    } else {
        printf("║  RESULT: FAIL — %3d timestamp overlap(s) detected      ║\n",
               total_overlaps);
    }
    printf("╚══════════════════════════════════════════════════════════╝\n");

    /* ── Cleanup ── */
    free(seq_recs);
    free(burst_recs);
    hsa_amd_memory_pool_free(A);
    hsa_amd_memory_pool_free(B);
    hsa_amd_memory_pool_free(C);
    hsa_queue_destroy(queue);
    hsa_shut_down();

    return (total_overlaps > 0) ? 1 : 0;
}
