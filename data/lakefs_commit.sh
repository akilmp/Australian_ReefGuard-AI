#!/usr/bin/env bash
set -euo pipefail

# Usage: lakefs_commit.sh <repository> <branch> <message>
# Commit staged changes in the specified repository branch with a message.

REPO="${1:?repository name required}"
BRANCH="${2:?branch name required}"
MESSAGE="${3:?commit message required}"

lakectl commit "lakefs://${REPO}/${BRANCH}" -m "${MESSAGE}"
