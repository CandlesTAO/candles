#!/bin/bash
# Script to run pre-commit hooks manually

set -e

echo "Running pre-commit hooks..."

uv run pre-commit run --all-files

if [ "$1" == "--with-tests" ]; then
    echo ""
    echo "Running pre-push hooks (includes tests)..."
    uv run pre-commit run --hook-stage pre-push --all-files
fi

echo ""
echo "All pre-commit checks passed"
