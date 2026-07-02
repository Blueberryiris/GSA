#!/usr/bin/env bash
# Double-click this in Finder to build (if needed) and start the GSA Engine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/gsa-engine"

export PATH="/opt/homebrew/opt/rustup/bin:$PATH"

BIN="target/release/gsa-engine"
if [[ ! -x "$BIN" || "${1:-}" == "--build" ]]; then
  echo "Building GSA Engine (release)..."
  cargo build --release
fi

exec "$BIN"
