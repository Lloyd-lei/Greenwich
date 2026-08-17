#!/usr/bin/env bash
set -euo pipefail

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/env.sh"
exec "$ALPHAMOTION_ENV/bin/alphamotion" doctor
