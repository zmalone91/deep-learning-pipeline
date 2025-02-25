#!/usr/bin/env bash
set -euo pipefail
# ^ "set -euo pipefail" is common in production scripts to exit on errors, unset vars, etc.

# Name of your virtual environment folder
VENV_NAME="tf_env"

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_NAME" ]; then
  echo "Creating virtual environment: $VENV_NAME"
  python3 -m venv "$VENV_NAME"
fi

# 2. Activate the environment
echo "Activating virtual environment: $VENV_NAME"
# macOS/Linux activation:
source "$VENV_NAME/bin/activate"

# 3. Upgrade pip and install requirements
echo "Installing requirements from requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

echo "Done! Your environment is set up."
 