#!/usr/bin/env bash
# Re-create the LOCAL-ONLY symlinks that let the rendered deck show the replay frames.
#
# The replay SVGs are NOT committed to this website (see .gitignore). They live in the
# bandits checkout and are surfaced through a gitignored `replay/` symlink next to the
# rendered HTML. Run this once after `quarto render` (Quarto does not create symlinks).
#
#   research/presentations/bandits_many_arms/link-replay.sh
#
# Override the frame source with: REPLAY_SRC=/path/to/bandits/replay ./link-replay.sh
set -euo pipefail

REPLAY_SRC="${REPLAY_SRC:-$HOME/bandits/replay}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DECK_REL="research/presentations/bandits_many_arms"

if [[ ! -d "$REPLAY_SRC" ]]; then
  echo "✗ frame source not found: $REPLAY_SRC" >&2
  echo "  set REPLAY_SRC=/path/to/bandits/replay and re-run." >&2
  exit 1
fi

# Link ONLY into the rendered output (where the HTML resolves `replay/`). We must NOT
# put a symlink in the source deck dir: Quarto would follow it and copy all 52 MB into
# docs/ on every render. Run this AFTER `quarto render`.
target="$REPO_ROOT/docs/$DECK_REL/replay"
if [[ ! -d "$REPO_ROOT/docs/$DECK_REL" ]]; then
  echo "✗ render the deck first: $REPO_ROOT/docs/$DECK_REL does not exist" >&2
  exit 1
fi
rm -f "$target"
ln -s "$REPLAY_SRC" "$target"
echo "✓ linked docs/$DECK_REL/replay -> $REPLAY_SRC"
