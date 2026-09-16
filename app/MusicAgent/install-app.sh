#!/bin/zsh
# Build and install the development Music Agent bundle for Finder/Dock use.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
SOURCE_APP="${SCRIPT_DIR}/dist/Music Agent.app"
INSTALL_DIR="${HOME:?}/Applications"
DESTINATION_APP="${INSTALL_DIR}/Music Agent.app"

cd "${SCRIPT_DIR}"
./build-app.sh

mkdir -p "${INSTALL_DIR}"
if [[ -e "${DESTINATION_APP}" ]]; then
  rm -rf -- "${DESTINATION_APP}"
fi
/usr/bin/ditto "${SOURCE_APP}" "${DESTINATION_APP}"

echo "installed: ${DESTINATION_APP}"
echo "Finder: Home → Applications → Music Agent"
