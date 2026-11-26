# Recreate the experiment environment (pinned stack, RTX 4090 / sm_89).
#
#   VENV=.venv bash scripts/setup_env.sh
#
# The pins are the ones validated locally (see docs/4090-runbook.md); they are
# the *only* supported stack for phase-1 artifacts. A different torch version
# invalidates `compile_key.json` and `runtime.CandidateStore` refuses to load
# old candidates on purpose.
set -euo pipefail

VENV=${VENV:-.venv}
PY=${PY:-3.12}
# Optional PyPI mirror, e.g. INDEX=https://mirrors.aliyun.com/pypi/simple/
INDEX=${INDEX:-}
ROOT=$(cd "$(dirname "$0")/.." && pwd)

command -v uv >/dev/null || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }

idx=()
[[ -n "$INDEX" ]] && idx=(--index-url "$INDEX")

uv venv --python "$PY" "$VENV" --allow-existing
uv pip install --python "$VENV/bin/python" "${idx[@]}" -r "$ROOT/requirements.lock"

"$VENV/bin/python" -c "import torch, triton, sys; \
print('python', sys.version.split()[0]); \
print('torch', torch.__version__, 'cuda', torch.version.cuda); \
print('triton', triton.__version__); \
print('cuda available', torch.cuda.is_available()); \
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

uv pip freeze --python "$VENV/bin/python" > "$ROOT/pip-freeze.txt"
echo "wrote $ROOT/pip-freeze.txt"
