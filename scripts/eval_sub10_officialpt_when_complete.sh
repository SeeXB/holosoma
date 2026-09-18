#!/usr/bin/env bash
# Wait for all three fresh official-PT trainings to exit successfully before
# launching the same 32-env x 10-episode Paper-DR eval used for the older runs.
# Run this in its own persistent tmux session; it never interrupts training.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$PWD"
TASK="sub10_largebox_089"
TAG="officialpt_default1mm_v1"
MANIFEST="$PROJECT_ROOT/exp/training/$TASK/launcher_logs/parallel_manifest.txt"
EVAL_ROOT="$PROJECT_ROOT/exp/eval/$TASK/20260916_${TAG}_paper_dr"
mkdir -p "$EVAL_ROOT"
exec >> "$EVAL_ROOT/auto_eval.log" 2>&1
trap 'status=$?; echo "$(date -Is) eval_watcher_exit_status=$status"' EXIT

RUN1="$PROJECT_ROOT/logs/WholeBodyTracking/20260913_181327-${TASK}_originaltraj_originalrl_s42_${TAG}-locomotion"
RUN2="$PROJECT_ROOT/logs/WholeBodyTracking/20260913_181327-${TASK}_semanticb4traj_originalrl_s42_${TAG}-locomotion"
RUN3="$PROJECT_ROOT/logs/WholeBodyTracking/20260913_181327-${TASK}_semanticb4traj_semanticadaptive_s42_${TAG}-locomotion"
REFERENCE_DIR="$PROJECT_ROOT/exp/training/$TASK/motions/official_intermimic_default1mm"

echo "$(date -Is) waiting_for_three_trainings tag=$TAG"
if [[ -s "$EVAL_ROOT/final_results.json" ]]; then
    echo "$(date -Is) final_results.json already exists; not repeating evaluation"
    exit 0
fi
for group in 1 2 3; do
    launch="$(rg "^group=$group pid=[0-9]+ started=.* tag=$TAG$" "$MANIFEST" | tail -n 1)"
    if [[ ! "$launch" =~ ^group=$group\ pid=([0-9]+)\  ]]; then
        echo "missing unique launch for group=$group tag=$TAG" >&2
        exit 1
    fi
    declare "pid$group=${BASH_REMATCH[1]}"
done

while true; do
    pending=0
    for group in 1 2 3; do
        pid_var="pid$group"
        pid="${!pid_var}"
        finished="$(rg "^group=$group pid=$pid finished=.* exit_status=" "$MANIFEST" | tail -n 1 || true)"
        if [[ -z "$finished" ]]; then
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "$(date -Is) group=$group process $pid exited without a manifest completion; refusing to evaluate" >&2
                exit 1
            fi
            pending=$((pending + 1))
        elif [[ ! "$finished" =~ exit_status=0$ ]]; then
            echo "$(date -Is) group=$group training failed: $finished" >&2
            exit 1
        fi
    done
    if (( pending == 0 )); then
        break
    fi
    echo "$(date -Is) training_still_running=$pending; evaluation_waiting"
    sleep 60
done

for run in "$RUN1" "$RUN2" "$RUN3"; do
    if [[ ! -s "$run/model_29999.pt" ]]; then
        echo "$(date -Is) missing final checkpoint: $run/model_29999.pt" >&2
        exit 1
    fi
done
echo "$(date -Is) all_three_trainings_complete; starting sequential Paper-DR evaluation"
export RUN1 RUN2 RUN3 REFERENCE_DIR EVAL_ROOT
export EVAL_PARALLEL=0 CHECKPOINT_STEP=29999 NUM_ENVS=32 SEED=42
export WANDB_NAME="sub10_largebox_089_${TAG}_29999_paper_dr_s42"
bash scripts/eval_sub10_largebox_089_final.sh
echo "$(date -Is) final_report=$EVAL_ROOT/FINAL_EVAL_REPORT.md"
