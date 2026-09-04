#!/usr/bin/env bash
# =============================================================================
# QRC environment setup for macOS / Linux.
# Windows users: run setup_venv.bat instead.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${REPO}/.venv"

echo
echo "=== QRC Environment Setup ==="
echo "Repo : ${REPO}"
echo "Venv : ${VENV}"
echo

# Find a supported Python (3.13, 3.12, or 3.11). 3.14 is not supported yet
# (numpy 2.2.6 has no 3.14 wheel).
PY=""
for v in 3.13 3.12 3.11; do
    if command -v "python${v}" >/dev/null 2>&1; then
        PY="python${v}"
        break
    fi
done
if [ -z "${PY}" ]; then
    echo "ERROR: No supported Python found."
    echo "Install Python 3.11, 3.12, or 3.13 (Python 3.14 is not supported yet)."
    exit 1
fi

echo "Using: ${PY} ($(${PY} --version))"

# Recreate the venv from scratch.
rm -rf "${VENV}"
"${PY}" -m venv "${VENV}"

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

echo "Installing packages..."
python -m pip install -r "${REPO}/requirements.txt"

echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name qrc_venv --display-name "Python (QRC)"

echo
echo "=== Done! ==="
echo "Activate with:  source ${VENV}/bin/activate"
echo "Start JupyterLab:  jupyter lab"
echo
