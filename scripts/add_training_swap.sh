#!/usr/bin/env bash
# Add a separate 16 GiB swap file for six concurrent IsaacSim workers.
# Leaves the existing swap and /etc/fstab unchanged; rerun after a reboot.
set -euo pipefail
swap_path=/swapfile-holosoma-training
if [[ $EUID -ne 0 ]]; then
  echo "Run: sudo bash /home/shixiongbo/holosoma/scripts/add_training_swap.sh" >&2
  exit 1
fi
if swapon --show=NAME --noheadings --raw | rg -Fxq "$swap_path"; then
  echo "Training swap is already active."
  swapon --show
  exit 0
fi
if [[ -e "$swap_path" || -L "$swap_path" ]]; then
  echo "Refusing to overwrite existing inactive path: $swap_path" >&2
  exit 1
fi
umask 077
fallocate -l 16G "$swap_path"
chmod 600 "$swap_path"
mkswap "$swap_path"
swapon "$swap_path"
swapon --show
free -h
