#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PHYSMEND_PYTHON:-python}" "$PROJECT/infer_t2v.py" "$@"
