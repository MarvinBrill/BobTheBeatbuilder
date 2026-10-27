#!/bin/sh
set -eu
test -c /dev/dxg || { echo 'Missing /dev/dxg: use a Windows 11 WSL2 GPU-enabled environment.'; exit 1; }
test -d /usr/lib/wsl/lib || { echo 'WSL GPU libraries missing.'; exit 1; }
test -d /usr/lib/wsl/drivers || { echo 'WSL host drivers missing.'; exit 1; }
echo 'WSL paths exist locally. The Docker daemon must see the same paths.'
docker compose -f compose.directml.yml config --quiet
echo 'Configuration validated. Run: docker compose -f compose.directml.yml up --build'
