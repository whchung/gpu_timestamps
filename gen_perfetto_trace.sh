#!/usr/bin/env bash
# gen_perfetto_trace.sh — 一鍵產生 Perfetto timeline JSON
#
# 用法:
#   ./gen_perfetto_trace.sh [dispatch次數] [元素數量] [輸出檔名]
#
# 範例:
#   ./gen_perfetto_trace.sh                    # 預設 20 次 dispatch, 1024 元素
#   ./gen_perfetto_trace.sh 40 4096            # 40 次 dispatch, 4096 元素
#   ./gen_perfetto_trace.sh 20 1024 my_trace   # 輸出到 my_trace.json
#
# 前置條件: 已執行 make，hsa_vecadd 可執行檔存在於同一目錄

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NUM_DISPATCHES="${1:-20}"
NUM_ELEMENTS="${2:-1024}"
OUTPUT_NAME="${3:-trace_perfetto}"
OUTPUT_JSON="${SCRIPT_DIR}/${OUTPUT_NAME}.json"

ROCPROF_DIR=$(mktemp -d /tmp/rocprof_trace_XXXXXX)
ROCPROF_PREFIX="${ROCPROF_DIR}/result"

if [[ ! -x "${SCRIPT_DIR}/hsa_vecadd" ]]; then
    echo "ERROR: ${SCRIPT_DIR}/hsa_vecadd not found. Run 'make' first." >&2
    exit 1
fi

if ! command -v rocprofv3 &>/dev/null; then
    echo "ERROR: rocprofv3 not found in PATH." >&2
    exit 1
fi

echo "=== Step 1/3: Collecting kernel trace with rocprofv3 ==="
echo "  dispatches=${NUM_DISPATCHES}  elements=${NUM_ELEMENTS}"
echo "  rocprof output dir: ${ROCPROF_DIR}"

rocprofv3 --kernel-trace -o "${ROCPROF_PREFIX}" -- \
    "${SCRIPT_DIR}/hsa_vecadd" "${NUM_DISPATCHES}" "${NUM_ELEMENTS}"

DB_FILE="${ROCPROF_PREFIX}_results.db"
if [[ ! -f "${DB_FILE}" ]]; then
    echo "ERROR: Expected DB not found at ${DB_FILE}" >&2
    echo "  Files in ${ROCPROF_DIR}:"
    ls -la "${ROCPROF_DIR}" >&2
    exit 1
fi

echo ""
echo "=== Step 2/3: Converting SQLite DB → Chrome Trace JSON ==="

python3 - "${DB_FILE}" "${OUTPUT_JSON}" <<'PYEOF'
import sqlite3, json, sys, re

db_path = sys.argv[1]
output  = sys.argv[2]

conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch_%'")
tables = c.fetchall()
if not tables:
    print("ERROR: No rocpd_kernel_dispatch_* table found in DB.", file=sys.stderr)
    sys.exit(1)

uuid = re.sub(r'^rocpd_kernel_dispatch_', '', tables[0][0])
print(f"  Session UUID: {uuid}")

c.execute(f"SELECT id, display_name FROM rocpd_info_kernel_symbol_{uuid}")
ksyms = {r[0]: r[1] for r in c.fetchall()}

c.execute(f"""SELECT id, tid, kernel_id, queue_id, start, end,
                     workgroup_size_x, grid_size_x
              FROM rocpd_kernel_dispatch_{uuid}
              ORDER BY start""")
dispatches = c.fetchall()
conn.close()

if not dispatches:
    print("ERROR: No dispatch records found.", file=sys.stderr)
    sys.exit(1)

events = []
for row in dispatches:
    did, tid, kernel_id, queue_id, start_ns, end_ns, wg_x, grid_x = row
    name = ksyms.get(kernel_id, f"kernel_{kernel_id}")
    dur_ns = end_ns - start_ns

    events.append({
        "name": name,
        "cat": "gpu_kernel",
        "ph": "X",
        "ts": start_ns / 1000.0,
        "dur": dur_ns / 1000.0,
        "pid": 1,
        "tid": queue_id if queue_id else 1,
        "args": {
            "dispatch_id": did,
            "grid_size_x": grid_x,
            "workgroup_size_x": wg_x,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ns": dur_ns,
        }
    })

events.append({"name": "process_name", "ph": "M", "pid": 1,
               "args": {"name": "GPU (gfx950)"}})
events.append({"name": "thread_name", "ph": "M", "pid": 1,
               "tid": events[0]["tid"], "args": {"name": "Compute Queue"}})

trace = {"traceEvents": events}
with open(output, "w") as f:
    json.dump(trace, f, indent=1)

print(f"  Wrote {len(dispatches)} dispatch events")
PYEOF

echo ""
echo "=== Step 3/3: Cleanup ==="
rm -rf "${ROCPROF_DIR}"
echo "  Removed temp dir: ${ROCPROF_DIR}"

echo ""
echo "=== Done ==="
echo "  Output: ${OUTPUT_JSON}"
echo ""
echo "  Open in Perfetto UI:"
echo "    1. Browse to https://ui.perfetto.dev"
echo "    2. Click 'Open trace file' and select ${OUTPUT_JSON}"
