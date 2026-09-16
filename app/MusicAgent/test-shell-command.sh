#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT

cd "${SCRIPT_DIR}"
swiftc \
  Sources/MusicAgent/ShellController.swift \
  Tests/ShellCommandContractProbe.swift \
  -o "${STAGING}/ShellCommandContractProbe"
"${STAGING}/ShellCommandContractProbe"
