#!/usr/bin/env bash

set -euo pipefail

CONFIG_DIR="$(cd "$(dirname "$0")" && pwd)/../configs"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

shopt -s nullglob

for config_file in "$CONFIG_DIR"/*; do
	if [[ -f "$config_file" ]]; then
		echo "Launching job for config: $config_file"
		"$SCRIPT_DIR/gen_script.sh" "$config_file"
	fi
done
