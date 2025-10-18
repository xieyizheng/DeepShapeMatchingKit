# Usage:
#   chmod +x parallel_preprocess.sh
#   ./parallel_preprocess.sh --opt configs/coco.yaml --workers 8
#   # or with a venv:
#   ./parallel_preprocess.sh --opt configs/coco.yaml --workers 8 --python .venv/bin/python

#!/usr/bin/env bash
set -euo pipefail

# ---- user args ----
OPT_PATH=""         # e.g. configs/coco.yaml
NUM_WORKERS=""      # e.g. 8
PYTHON_BIN="python" # or "python3" / path to venv python

usage() {
  echo "Usage: $0 --opt <path/to/config.yaml> --workers <N> [--python <python_bin>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --opt) OPT_PATH="$2"; shift 2;;
    --workers) NUM_WORKERS="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    -h|--help) usage;;
    *) echo "Unknown arg: $1"; usage;;
  esac
done

[[ -z "${OPT_PATH}" || -z "${NUM_WORKERS}" ]] && usage
if [[ ! -f "${OPT_PATH}" ]]; then
  echo "Config not found: ${OPT_PATH}"; exit 2
fi

# ---- sanity checks ----
TOTAL_CORES=$(nproc --all)
if (( NUM_WORKERS > TOTAL_CORES )); then
  echo "Requested workers (${NUM_WORKERS}) > available cores (${TOTAL_CORES})."
  echo "Reduce --workers or run on a larger machine."
  exit 3
fi

# Optional: pin to *physical* cores if you know siblings (advanced).
# For simplicity we just use first NUM_WORKERS logical cores: 0..NUM_WORKERS-1

# ---- create logs ----
LOG_DIR="logs/preprocess_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# ---- make subprocesses well-behaved (no oversubscription) ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# If you use PyTorch at all inside, also consider:
export PYTORCH_NUM_THREADS=1

# Optional: keep the machine responsive if you’re hammering I/O
# nice -n 5 and ionice -c2 -n5 are gentle defaults.
NICE="nice -n 5"
IONICE="ionice -c2 -n5"

# ---- trap to clean up all children on Ctrl-C ----
pids=()
cleanup() {
  echo "Stopping workers..."
  for pid in "${pids[@]:-}"; do
    kill -TERM "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup INT TERM

echo "Launching ${NUM_WORKERS} workers (1 core each), config=${OPT_PATH}"
for wid in $(seq 0 $((NUM_WORKERS-1))); do
  core_id=$wid   # simple mapping: worker i -> CPU core i

  # make sure the print buffer is printed immediately
  export PYTHONUNBUFFERED=1
  # Use taskset to pin each process to exactly one CPU core.
  # On NUMA machines you can swap taskset for numactl:
  # numactl --physcpubind=$core_id --localalloc
  cmd=(taskset -c "${core_id}" ${PYTHON_BIN} parallel_preprocess.py \
        --opt "${OPT_PATH}" \
        --worker_id "${wid}" \
        --num_workers "${NUM_WORKERS}")

  # tee each worker’s stdout/stderr to a log
  log="${LOG_DIR}/worker_${wid}.log"
  echo "  core ${core_id} -> worker ${wid} | log: ${log}"
  # shellcheck disable=SC2086
  ${NICE} ${IONICE} "${cmd[@]}" >"${log}" 2>&1 &
  pids+=($!)
done

# Wait for all workers
fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=1; fi
done

if (( fail )); then
  echo "One or more workers exited with non-zero status. See logs in ${LOG_DIR}"
  exit 4
else
  echo "All workers completed successfully. Logs in ${LOG_DIR}"
fi
