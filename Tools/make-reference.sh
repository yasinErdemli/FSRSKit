#!/bin/sh
# Writes the reference values the tests compare against, from fsrs-rs at the
# commit pinned in Tools/reference/Cargo.toml. Needs Rust (cargo).
#
#     Tools/make-reference.sh
#
# Cargo's build folder lives in ~/Library/Caches, outside the repo.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/Library/Caches/fsrskit-reference}"
"${CARGO:-cargo}" run --release --quiet \
    --manifest-path "$root/Tools/reference/Cargo.toml" \
    -- "$root/Tests/FSRSKitTests/Reference"
