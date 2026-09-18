#!/usr/bin/env bash

# Lightweight, non-destructive watchdog for the three clothes-stand runs.
# It records host RAM, aggregate GPU memory/utilization, per-process RSS, and
# per-process GPU memory.  It deliberately never kills or pauses a process.

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${MEMORY_LOG_FILE:-${PROJECT_ROOT}/exp/training/sub9_clothesstand_058/launcher_logs/online_memory_watchdog.log}"
INTERVAL_S="${INTERVAL_S:-30}"
GPU_WARN_MIB="${GPU_WARN_MIB:-22000}"
GPU_CRIT_MIB="${GPU_CRIT_MIB:-23500}"
HOST_WARN_GIB="${HOST_WARN_GIB:-8}"
HOST_CRIT_GIB="${HOST_CRIT_GIB:-4}"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

echo "[$(date -Is)] watchdog started interval=${INTERVAL_S}s gpu_warn=${GPU_WARN_MIB}MiB gpu_critical=${GPU_CRIT_MIB}MiB host_available_warn=${HOST_WARN_GIB}GiB host_available_critical=${HOST_CRIT_GIB}GiB"

while true; do
    timestamp="$(date -Is)"
    alerts=()

    # nvidia-smi reports MiB and percent without units in this query.
    gpu_csv="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>&1 || true)"
    gpu_used="$(awk -F',' 'NR==1 {gsub(/[[:space:]]/,"",$1); print $1}' <<<"$gpu_csv")"
    gpu_total="$(awk -F',' 'NR==1 {gsub(/[[:space:]]/,"",$2); print $2}' <<<"$gpu_csv")"
    gpu_util="$(awk -F',' 'NR==1 {gsub(/[[:space:]]/,"",$3); print $3}' <<<"$gpu_csv")"
    if [[ "$gpu_used" =~ ^[0-9]+$ && "$gpu_total" =~ ^[0-9]+$ ]]; then
        gpu_pct="$(awk -v u="$gpu_used" -v t="$gpu_total" 'BEGIN {if (t>0) printf "%.1f",100*u/t; else print "nan"}')"
        if (( gpu_used >= GPU_CRIT_MIB )); then
            alerts+=("GPU_CRITICAL=${gpu_used}/${gpu_total}MiB")
        elif (( gpu_used >= GPU_WARN_MIB )); then
            alerts+=("GPU_WARN=${gpu_used}/${gpu_total}MiB")
        fi
        echo "[$timestamp] gpu memory=${gpu_used}/${gpu_total}MiB (${gpu_pct}%) compute_util=${gpu_util}%"
    else
        echo "[$timestamp] gpu query failed: ${gpu_csv//$'\n'/ }"
    fi

    # free(1) gives bytes with -b; available is the relevant OOM headroom.
    # Force the stable English labels; the host's default locale is Chinese.
    mem_csv="$(LC_ALL=C free -b | awk '/^Mem:/ {print $2","$3","$4","$7} /^Swap:/ {print "swap,"$2","$3}')"
    mem_total="$(awk -F',' '$1 ~ /^[0-9]+$/ {print $1; exit}' <<<"$mem_csv")"
    mem_used="$(awk -F',' '$1 ~ /^[0-9]+$/ {print $2; exit}' <<<"$mem_csv")"
    mem_free="$(awk -F',' '$1 ~ /^[0-9]+$/ {print $3; exit}' <<<"$mem_csv")"
    mem_available="$(awk -F',' '$1 ~ /^[0-9]+$/ {print $4; exit}' <<<"$mem_csv")"
    swap_used="$(awk -F',' '$1=="swap" {print $3; exit}' <<<"$mem_csv")"
    if [[ "$mem_available" =~ ^[0-9]+$ ]]; then
        available_gib="$(awk -v b="$mem_available" 'BEGIN {printf "%.2f", b/1024/1024/1024}')"
        used_gib="$(awk -v b="$mem_used" 'BEGIN {printf "%.2f", b/1024/1024/1024}')"
        if (( mem_available < HOST_CRIT_GIB * 1024 * 1024 * 1024 )); then
            alerts+=("HOST_RAM_CRITICAL=${available_gib}GiB_available")
        elif (( mem_available < HOST_WARN_GIB * 1024 * 1024 * 1024 )); then
            alerts+=("HOST_RAM_WARN=${available_gib}GiB_available")
        fi
        echo "[$timestamp] host ram used=${used_gib}GiB available=${available_gib}GiB free=${mem_free}B swap_used=${swap_used:-unknown}B"
    else
        echo "[$timestamp] host memory query failed"
    fi

    # Match only the actual train_agent commands, not this watchdog or wrappers.
    process_lines="$(ps -eo pid=,rss=,etime=,cmd= | awk '/[t]rain_agent.py/ && /sub9_clothesstand_058/ {print}')"
    process_count=0
    if [[ -n "$process_lines" ]]; then
        while IFS= read -r line; do
            process_count=$((process_count + 1))
            echo "[$timestamp] process $line"
        done <<<"$process_lines"
    else
        echo "[$timestamp] process none-matched"
    fi
    if (( process_count != 3 )); then
        alerts+=("TRAINING_PROCESS_COUNT=${process_count}/3")
    fi

    # Attribute GPU memory to the three PIDs when available.
    gpu_apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$gpu_apps" ]]; then
        while IFS= read -r app; do
            [[ -z "$app" ]] || echo "[$timestamp] gpu_process $app"
        done <<<"$gpu_apps"
    fi

    if (( ${#alerts[@]} > 0 )); then
        echo "[$timestamp] ALERT ${alerts[*]}"
    else
        echo "[$timestamp] status=OK"
    fi
    sleep "$INTERVAL_S"
done
