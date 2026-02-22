#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}

$PYTHON_BIN -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

# Install CUDA 12.1 wheels for NVIDIA GPUs by default.
# Override TORCH_INDEX_URL if needed.
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

# Install project dependencies in editable mode.
pip install -e .

echo "Setup complete. Activate with: source $VENV_DIR/bin/activate"
