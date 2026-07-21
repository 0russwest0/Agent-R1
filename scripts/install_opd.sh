#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_VERL_REPO="https://github.com/fishsure/verl.git"
DEFAULT_VERL_REF="5779c7c6782733f77ef640f557bea572dfeacc12"
VERL_REPO="${VERL_REPO:-$DEFAULT_VERL_REPO}"
VERL_REF="${VERL_REF:-$DEFAULT_VERL_REF}"
VERL_SPEC="${VERL_SPEC:-verl @ git+${VERL_REPO}@${VERL_REF}}"

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  echo "pip is not available for $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$VERL_REPO" == "$DEFAULT_VERL_REPO" && "$VERL_REF" == "$DEFAULT_VERL_REF" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade --no-deps -r requirements-opd.txt
else
  echo "Installing OPD verl from $VERL_REPO at $VERL_REF"
  "$PYTHON_BIN" -m pip install --upgrade --no-deps "$VERL_SPEC"
fi

PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
import sys

try:
    import verl
    from verl.experimental.teacher_loop import MultiTeacherModelManager  # noqa: F401
    from verl.trainer.distillation import distillation_ppo_loss  # noqa: F401
    from verl.trainer.ppo.utils import Role
except (ImportError, AttributeError) as exc:
    raise SystemExit(f"Installed verl does not provide Agent-R1 OPD support: {exc}") from exc

if not hasattr(Role, "TeacherModel"):
    raise SystemExit("Installed verl does not define Role.TeacherModel.")

version = getattr(verl, "__version__", "unknown")
if "agentr1.opd" not in version:
    raise SystemExit(f"Unexpected verl version: {version}")

print(f"Agent-R1 OPD dependencies are ready (Python {sys.version.split()[0]}, verl {version}).")
PY

cat <<'EOF'
Next, prepare a dataset and launch an OPD example, for example:
  python3 -m recipes.gsm8k.data_preprocess.process_gsm8k --local_save_dir "$HOME/data/gsm8k"
  bash examples/gsm8k/run_opd.sh
EOF
