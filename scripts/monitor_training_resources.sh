#!/usr/bin/env bash

# Record GPU and system-memory use until no Python CUDA process remains.

set -u

OUTPUT="${1:?usage: monitor_training_resources.sh OUTPUT_CSV [INTERVAL_SECONDS]}"
INTERVAL_S="${2:-15}"

if [[ ! -s "$OUTPUT" ]]; then
    echo "timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,system_memory_used_mib,system_memory_total_mib" > "$OUTPUT"
fi

while true; do
    app_rows="$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null || true)"
    if ! grep -q "python" <<< "$app_rows"; then
        break
    fi
    timestamp="$(date --iso-8601=seconds)"
    # Match the data row by position because ``free`` localizes the "Mem:" label.
    read -r system_total system_used < <(free -m | awk 'NR == 2 {print $2, $3}')
    gpu_rows="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true)"
    while IFS= read -r gpu_row; do
        [[ -z "$gpu_row" ]] && continue
        echo "$timestamp,$gpu_row,$system_used,$system_total" >> "$OUTPUT"
    done <<< "$gpu_rows"
    sleep "$INTERVAL_S"
done
