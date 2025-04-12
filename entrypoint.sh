#!/bin/bash
# Install editable packages at container start
pip install -e d4rl
pip install -e .
# Execute the passed command (default: bash)
exec "$@"
