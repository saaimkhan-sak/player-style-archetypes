#!/usr/bin/env bash
set -euo pipefail

# Always run from the repo root
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# Optional: only auto-commit specific directories (safer)
# Adjust these paths to what you actually want to track.
TRACK_PATHS=(
  "app"
  "src"
  "config"
  ".streamlit"
  "requirements.txt"
  "data/app"
  "reports"
)


# Make sure we're on main
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$BRANCH" != "main" ]]; then
  echo "Not on main (currently: $BRANCH). Refusing to auto-push."
  exit 1
fi

# Stage only the tracked paths (prevents accidentally committing secrets/junk)
git add -A "${TRACK_PATHS[@]}"

# If nothing changed, exit quietly
if git diff --cached --quiet; then
  echo "No changes to commit."
  exit 0
fi

# Commit message with timestamp
TS="$(date +"%Y-%m-%d %H:%M:%S")"
git commit -m "Auto-update generated artifacts (${TS})"

# Push
git push origin main

echo "Pushed updates."

