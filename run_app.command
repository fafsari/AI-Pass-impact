#!/bin/bash
# Start the Ki-67 app. Double-click this file (first time: right-click -> Open).
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed yet. Open Terminal and run:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "Then double-click run_app.command again."
  read -n 1 -s -r -p "Press any key to close."
  exit 1
fi
echo "Starting the app. The first start installs packages and takes a few minutes."
echo "Your browser will open http://localhost:8501 - keep this window open while you use the app."
uv run streamlit run app.py
