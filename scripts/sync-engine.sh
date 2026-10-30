#!/bin/bash
# Build a fresh venv in Stitch app Resources (venvs aren't relocatable — create in place).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${PYTHON:-/opt/homebrew/bin/python3}"
if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi

DEST="$ROOT/app/Stitch/Resources/engine"
rm -rf "$DEST"
mkdir -p "$DEST"

cp "$ROOT/engine/stitch.py" "$DEST/stitch.py"
cp "$ROOT/engine/requirements.txt" "$DEST/requirements.txt"

"$PYTHON" -m venv "$DEST/.venv"
"$DEST/.venv/bin/pip" install -q --upgrade pip
"$DEST/.venv/bin/pip" install -q -r "$DEST/requirements.txt"

cat > "$DEST/stitch-engine" << 'INNER'
#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" "$ROOT/stitch.py" "$@"
INNER
chmod +x "$DEST/stitch-engine" "$DEST/stitch.py"

# Dev CLI convenience (local venv at engine/.venv)
"$ROOT/engine/setup.sh"

echo "Synced engine -> app/Stitch/Resources/engine/"
