# Changes for New XCC Timestamp ABI

## Summary
Updated the microbenchmark to support the new AMD signal ABI that includes per-XCC CP dispatch start timestamps.

## Changes Made

### 1. Updated `amd_signal_t` Structure (hsa_vecadd.c:46-78)

Changed from:
```c
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
```

To the new ABI:
```c
typedef struct {
    int64_t  kind;
    union {
        volatile int64_t value;
        volatile uint64_t* hardware_doorbell_ptr;
    };
    uint64_t event_mailbox_ptr;
    uint32_t event_id;
    uint32_t reserved1;
    uint64_t start_ts;
    uint64_t end_ts;
    union {
        void* queue_ptr;
        uint64_t reserved2;
    };
    uint32_t reserved3[2];
    /* Debug-only per-XCC dispatch start timestamps (MI450, 8 XCCs/chiplets).
     * The master CP on XCC0 records start_ts/end_ts above; because XCC0 observes
     * work later than the chiplet that actually launched it, start_ts can be
     * reported too late. To investigate this each XCC's CP additionally records
     * the low 16 bits of its own dispatch-start RTC sample (100 MHz, 10 ns/tick)
     * here. These fields are appended after reserved3 so the frozen offsets of
     * start_ts/end_ts (and all preceding fields) are unchanged. They are purely
     * for debug inspection and are NOT consumed by the runtime timing path. The
     * SDMA path does not write them. A slot of 0 means the corresponding XCC did
     * not record a sample. Reconstruct each XCC's absolute start against start_ts
     * via modular signed subtraction (exact to 10 ns while |skew| < 327.68 us):
     *   int16_t d = (int16_t)(xcc_start_ts_lo[i] - (uint16_t)start_ts);
     *   uint64_t xcc_abs = start_ts + d;
     */
    uint16_t xcc_start_ts_lo[8];
    uint32_t reserved4[12];
} amd_signal_t;
```

**Key Changes:**
- `kind` changed from `int32_t` to `int64_t` (per ABI spec uses `amd_signal_kind64_t`)
- Union value member changed to `int64_t` (matching spec)
- Added `xcc_start_ts_lo[8]` array for per-XCC timestamps
- Added `reserved4[12]` to maintain structure alignment
- `reserved2` renamed to `reserved3` for clarity
- Added comprehensive documentation comment explaining the per-XCC timestamp reconstruction

### 2. Extended `ts_record_t` Structure (hsa_vecadd.c:105-120)

Added field to store per-XCC timestamps:
```c
typedef struct {
    uint64_t host_pre;
    uint64_t host_post;

    uint64_t cp_start;
    uint64_t cp_end;

    /* Per-XCC CP start timestamps (low 16 bits from signal structure) */
    uint16_t xcc_start_ts_lo[8];

    /* Shader dual-clock measurements (per-chiplet, 8 chiplets) */
    uint32_t chiplet_xcc_id[NUM_WORKGROUPS];
    uint64_t chiplet_realtime_1[NUM_WORKGROUPS];
    uint64_t chiplet_cycles_1[NUM_WORKGROUPS];
    uint64_t chiplet_realtime_2[NUM_WORKGROUPS];
    uint64_t chiplet_cycles_2[NUM_WORKGROUPS];
} ts_record_t;
```

### 3. Updated Sequential Dispatch (hsa_vecadd.c:~508-515)

Added code to capture per-XCC timestamps:
```c
/* Timestamp point 2: CP timestamps from signal structure */
amd_signal_t *amd_sig = (amd_signal_t *)signal.handle;
records[r].cp_start = amd_sig->start_ts;
records[r].cp_end   = amd_sig->end_ts;

/* Per-XCC CP start timestamps (low 16 bits) */
for (int xcc = 0; xcc < 8; xcc++) {
    records[r].xcc_start_ts_lo[xcc] = amd_sig->xcc_start_ts_lo[xcc];
}
```

### 4. Updated Burst Dispatch (hsa_vecadd.c:~598-610)

Added code to capture per-XCC timestamps in burst mode:
```c
/* Read timestamps from signal structures */
for (int r = 0; r < num_runs; r++) {
    records[r].host_pre  = burst_pre;
    records[r].host_post = burst_post;

    /* CP timestamps from signal structure */
    amd_signal_t *amd_sig = (amd_signal_t *)signals[r].handle;
    records[r].cp_start = amd_sig->start_ts;
    records[r].cp_end   = amd_sig->end_ts;

    /* Per-XCC CP start timestamps (low 16 bits) */
    for (int xcc = 0; xcc < 8; xcc++) {
        records[r].xcc_start_ts_lo[xcc] = amd_sig->xcc_start_ts_lo[xcc];
    }
    ...
}
```

### 5. Added New Display Function (hsa_vecadd.c:786-810)

New function to display and reconstruct per-XCC timestamps:
```c
static void print_xcc_timestamps(const ts_record_t *rec, int run_num) {
    printf("\n━━━ Run %d: Per-XCC CP Dispatch Start Timestamps ━━━\n", run_num);
    printf("  CP start_ts (master XCC0) = %lu\n", rec->cp_start);
    printf("  CP end_ts   (master XCC0) = %lu\n\n", rec->cp_end);

    printf("  Per-XCC CP start timestamp reconstruction:\n");
    printf("  %4s  %10s  %20s  %20s  %20s\n",
           "XCC", "lo_16bits", "Reconstructed (abs)", "Δ from master", "Δ (ns)");

    double to_ns = 1e9 / (double)g_sys_freq;

    for (int xcc = 0; xcc < 8; xcc++) {
        uint16_t lo16 = rec->xcc_start_ts_lo[xcc];

        if (lo16 == 0) {
            printf("  %4d  %10s  %20s  %20s  %20s\n",
                   xcc, "0", "N/A", "N/A", "N/A");
        } else {
            /* Reconstruct absolute timestamp via modular signed subtraction */
            int16_t delta = (int16_t)(lo16 - (uint16_t)rec->cp_start);
            uint64_t xcc_abs = rec->cp_start + delta;
            int64_t diff_from_master = (int64_t)xcc_abs - (int64_t)rec->cp_start;
            double diff_ns = (double)diff_from_master * to_ns;

            printf("  %4d  %10u  %20lu  %20ld  %20.1f\n",
                   xcc, lo16, xcc_abs, diff_from_master, diff_ns);
        }
    }
}
```

**This function:**
- Displays raw 16-bit values from the signal structure
- Reconstructs full 64-bit absolute timestamps using modular signed subtraction (as specified in ABI)
- Shows delta from master XCC0 timestamp in both ticks and nanoseconds
- Handles zero values (XCC did not record a sample)

### 6. Integrated Display in Main Loop

Updated both sequential and burst dispatch sections to call the new display function:
```c
/* Print dual-clock detail and XCC timestamps for all runs */
for (int r = 0; r < num_runs; r++) {
    print_xcc_timestamps(&seq_recs[r], r);
    print_dual_clock_detail(&seq_recs[r], r);
}
```

## Output Format

The new output will show for each dispatch:

```
━━━ Run 0: Per-XCC CP Dispatch Start Timestamps ━━━
  CP start_ts (master XCC0) = 123456789012345
  CP end_ts   (master XCC0) = 123456789023456

  Per-XCC CP start timestamp reconstruction:
   XCC  lo_16bits    Reconstructed (abs)       Δ from master                Δ (ns)
     0      12345           123456789012345                    0                 0.0
     1      12348           123456789012348                    3                30.0
     2      12350           123456789012350                    5                50.0
     3      12352           123456789012352                    7                70.0
     4      12354           123456789012354                    9                90.0
     5      12356           123456789012356                   11               110.0
     6      12358           123456789012358                   13               130.0
     7      12360           123456789012360                   15               150.0
```

This allows investigating which XCCs observed the dispatch first and measuring the skew between XCCs.

## Testing

To build and test:
```bash
make clean
make
./hsa_vecadd 10
```

The microbenchmark will now display per-XCC CP timestamps alongside the existing shader dual-clock measurements.
