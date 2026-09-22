#!/bin/bash
set -e

CODE_DIR="/workspace/third_parties/alphafold3"
SETUP_FILE="${CODE_DIR}/pyproject.toml"

if [ ! -f "${SETUP_FILE}" ]; then
    echo "Error: Source code not found in ${CODE_DIR}. Please mount the source code correctly."
    exit 1
fi

pip show alphafold3 &> /dev/null || pip3 install --no-deps -e "${CODE_DIR}"

exec "$@"
