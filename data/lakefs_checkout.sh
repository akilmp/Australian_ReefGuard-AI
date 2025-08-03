#!/usr/bin/env bash
set -euo pipefail

# Usage: lakefs_checkout.sh <repository> <branch> [<source_branch>]
# Create a branch in the specified repository if it doesn't exist and print its path.

REPO="${1:?repository name required}"
BRANCH="${2:?branch name required}"
SOURCE_BRANCH="${3:-main}"

LAKEFS_PATH="lakefs://${REPO}/${BRANCH}"
SOURCE_PATH="lakefs://${REPO}/${SOURCE_BRANCH}"

# Create branch if it doesn't already exist
lakectl branch create "${LAKEFS_PATH}" --source "${SOURCE_PATH}" >/dev/null 2>&1 || true

echo "Checked out ${LAKEFS_PATH}"
