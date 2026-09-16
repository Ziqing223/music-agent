#!/bin/zsh
# Music Agent — development bundle assembler (P20 Slice A).
#
# Single compile path: SwiftPM release build, then dist/"Music Agent.app"
# with the usual Contents/{MacOS,Info.plist} layout. Build failures exit
# non-zero; there is no fallback compiler path.
#
# Signing, notarization, DMG and /Applications installation are intentionally
# out of scope.
set -euo pipefail

cd "$(dirname "$0")"

BIN="MusicAgent"
APP="dist/Music Agent.app"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

swift build -c release
cp "$(swift build -c release --show-bin-path)/${BIN}" "$STAGING/${BIN}"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$STAGING/${BIN}" "$APP/Contents/MacOS/${BIN}"
cp Resources/Info.plist "$APP/Contents/Info.plist"

echo "built (swiftpm): ${APP}"