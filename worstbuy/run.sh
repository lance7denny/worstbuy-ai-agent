#!/usr/bin/env bash
# One-command start for the Worstbuy demo. Runs entirely on your machine.
set -e
cd "$(dirname "$0")/.."

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Get it from https://ollama.com/download, start it, then run this script again."
  exit 1
fi
if ! ollama list >/dev/null 2>&1; then
  echo "Ollama is installed but not running. Open the Ollama app (or run: ollama serve) and try again."
  exit 1
fi

PY="${PYTHON:-python3}"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Python 3.10 or newer is required (found: $($PY --version 2>&1)). Set PYTHON=/path/to/python3.12 to use another one."
  exit 1
}

[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install -q -r requirements.txt

missing=()
for m in qwen3:4b-instruct nomic-embed-text; do
  ollama list | awk 'NR>1 {print $1}' | grep -q "^$m" || missing+=("$m")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Missing models: ${missing[*]}  (about 2.8 GB download in total, from the Ollama registry)"
  read -r -p "Download them now with 'ollama pull'? [y/N] " ans
  if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
    for m in "${missing[@]}"; do ollama pull "$m"; done
  else
    echo "Okay. Run these yourself, then start again:"
    for m in "${missing[@]}"; do echo "  ollama pull $m"; done
    exit 1
  fi
fi

echo "Starting Worstbuy at http://127.0.0.1:8000  (Ctrl+C to stop)"
exec python worstbuy/server.py
