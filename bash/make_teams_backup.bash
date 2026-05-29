#!/usr/bin/env bash 

# Stage everything
git add -A
# Commit changes or fail silently if nothing to commit
if ! git diff --cached --quiet; then
  git commit -m "Commit before backup" || true
fi
TAG=backup-$(date +%Y%m%d-%H%M%S)
git tag -a $TAG -m "Weekly backup"
git push origin $TAG