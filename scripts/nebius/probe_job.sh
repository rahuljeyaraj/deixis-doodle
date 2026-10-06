#!/usr/bin/env bash
# The 24-hour Token Factory probe as a Nebius AI Cloud Serverless Job on CPU (measure list M35).
#
# PREPARED, NOT RUN YET (6 Oct 2026). Run inside WSL (the Nebius CLI has no Windows build).
# One Nemotron call a minute for 24 hours (1,440 calls), the plotter's real request
# (probe_request.py). The key comes from Nebius SecretStash (the CLI calls it `mysterybox`),
# passed with --env-secret, never --env. probe.py prints only timings, status codes and token
# counts; probe_summary.py checks the fetched log for the key before any number is quoted.
#
# Usage:
#   bash scripts/nebius/probe_job.sh <command> [arg]
#
# Commands:
#   settings        the resolved settings and the create command (free, no cloud call)
#   request         write out/probe/probe_request.json (needs the repo's uv env; run on Windows:
#                   uv run python scripts/nebius/probe_request.py)
#   secret          store NEBIUS_API_KEY from .env in SecretStash as $SECRET_NAME (free in
#                   preview; never printed). Needs NEBIUS_PARENT_ID (the project).
#   dry-run         the CLI validates the create request (free, creates nothing)
#   create          create the Job (BILLING STARTS: a small CPU VM for about 24 h, about $2)
#   status          job id and state
#   logs [N]        last N log lines (default 20)
#   fetch-logs [F]  the whole log to F (default out/probe/job-log.txt), in hourly slices
#   cancel          stop the job now
#   delete          delete the finished job's record
#   list            jobs AND endpoints: check that nothing is left running
#   watchdog [H]    sleep H hours (default 26), then cancel. Run it in the background
#
# The day, step by step (after the Fri 9 Oct region check):
#   (Windows)  uv run python scripts/nebius/probe_request.py
#   (WSL)      export NEBIUS_PARENT_ID=<the eu-north1 project id>   # else the CLI default
#              bash scripts/nebius/probe_job.sh settings
#              bash scripts/nebius/probe_job.sh secret
#              bash scripts/nebius/probe_job.sh dry-run
#              bash scripts/nebius/probe_job.sh create
#              bash scripts/nebius/probe_job.sh logs        # a few {"i": ..., "status": 200} lines
#   (next day) bash scripts/nebius/probe_job.sh fetch-logs
#   (Windows)  uv run python scripts/probe_summary.py out/probe/job-log.txt --out <results>
#              bash scripts/nebius/probe_job.sh list ; ... delete
#
# Settings (environment variables override): JOB_NAME, IMAGE, PLATFORM, PRESET, DISK_SIZE,
#   TIMEOUT, CALLS, EVERY, SECRET_NAME, NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID, NEBIUS_BIN, ENV_FILE.
#
# NOT CONFIRMED (check with `dry-run` on the day):
#   - the CPU platform and preset names for Jobs in eu-north1 (cpu-d3 / 2vcpu-8gb here; the
#     jobs docs only show GPU examples; `nebius compute platform list` shows what exists);
#   - that `--env-secret KEY=<secret name>` finds the secret by name in the same project (the
#     docs say a name, an ID, or secret@version all work);
#   - that python:3.12-slim pulls from Docker Hub without registry credentials.
#
# Sources (read 6 Oct 2026):
#   https://docs.nebius.com/serverless/jobs/manage          --env-secret <key=secret_selector>;
#       the key must match a payload key in the secret; --timeout min 1h, max 168h
#   https://docs.nebius.com/cli/reference/mysterybox/secret/create   --secret-version-payload
#       '[{"key": "<key>", "string_value": "<value>"}]', --parent-id required
#   https://docs.nebius.com/mysterybox/resources/pricing    "in preview and provided free"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
NB="${NEBIUS_BIN:-$HOME/.nebius/bin/nebius}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

NAME="${JOB_NAME:-deixis-probe-24h}"
IMAGE="${IMAGE:-python:3.12-slim}"
PLATFORM="${PLATFORM:-cpu-d3}"
PRESET="${PRESET:-2vcpu-8gb}"
DISK_SIZE="${DISK_SIZE:-32Gi}"
TIMEOUT="${TIMEOUT:-26h}"  # 1,440 calls a minute apart take 24 h; the hard stop is 2 h later
CALLS="${CALLS:-1440}"
EVERY="${EVERY:-60}"
SECRET_NAME="${SECRET_NAME:-deixis-token-factory}"
REQUEST="$REPO_DIR/out/probe/probe_request.json"

RUN="python3 -u /job/probe.py --request /job/probe_request.json --calls $CALLS --every $EVERY"
ARGS="-c \"$RUN\""

env_value() {
  [ -f "$ENV_FILE" ] || return 0
  grep -E "^[[:space:]]*$1[[:space:]]*=" "$ENV_FILE" | tail -n 1 | cut -d= -f2- \
    | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' \
    -e "s/^'\(.*\)'$/\1/" || true
}

KEY_VALUE="$(env_value NEBIUS_API_KEY)"

redact() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$KEY_VALUE" ] && line="${line//"$KEY_VALUE"/***REDACTED***}"
    printf '%s\n' "$line"
  done
}

need_cli() {
  if [ ! -x "$NB" ]; then
    echo "Nebius CLI not found at $NB. Run this script inside WSL, or set NEBIUS_BIN." >&2
    exit 1
  fi
}

scope_flags=()
[ -n "${NEBIUS_PARENT_ID:-}" ] && scope_flags+=(--parent-id "$NEBIUS_PARENT_ID")

print_settings() {
  cat <<EOF
job name:    $NAME
image:       $IMAGE
platform:    $PLATFORM / $PRESET  (CPU, not preemptible)
disk:        $DISK_SIZE
timeout:     $TIMEOUT  ($CALLS calls, one every $EVERY s)
key:         from SecretStash secret "$SECRET_NAME" (--env-secret NEBIUS_API_KEY=$SECRET_NAME)
request:     $REQUEST $([ -f "$REQUEST" ] && echo "($(wc -c < "$REQUEST") bytes)" || echo "(MISSING: run 'request')")
parent id:   ${NEBIUS_PARENT_ID:-<CLI profile default>}
subnet id:   ${NEBIUS_SUBNET_ID:-<default>}
.env key:    $([ -n "$KEY_VALUE" ] && echo "found (never printed)" || echo "NOT FOUND in $ENV_FILE")
command:     bash $ARGS
EOF
}

job_id() {
  "$NB" ai job get-by-name ${scope_flags[@]+"${scope_flags[@]}"} --name "$NAME" \
    --format jsonpath='{.metadata.id}' 2>/dev/null || true
}

create() {
  need_cli
  [ -f "$REQUEST" ] || { echo "No $REQUEST: run (Windows) uv run python scripts/nebius/probe_request.py" >&2; exit 1; }
  if [ "${1:-}" != "--dry-run" ] && [ -n "$(job_id)" ]; then
    echo "A job named $NAME already exists: check it with 'status', then 'delete' it." >&2
    exit 1
  fi
  local flags=(
    --name "$NAME"
    --image "$IMAGE"
    --platform "$PLATFORM"
    --preset "$PRESET"
    --disk-size "$DISK_SIZE"
    --timeout "$TIMEOUT"
    --restart-policy never
    --inject-file "$SCRIPT_DIR/probe.py:/job/probe.py"
    --inject-file "$REQUEST:/job/probe_request.json"
    --env PYTHONUNBUFFERED=1
    --env-secret "NEBIUS_API_KEY=$SECRET_NAME"
    --container-command bash
    --args "$ARGS"
  )
  [ -n "${NEBIUS_SUBNET_ID:-}" ] && flags+=(--subnet-id "$NEBIUS_SUBNET_ID")
  print_settings
  echo
  "$NB" ai job create ${scope_flags[@]+"${scope_flags[@]}"} "${flags[@]}" --async "$@" 2>&1 \
    | redact
}

case "${1:-}" in
  settings)
    print_settings
    ;;
  request)
    echo "Run this on Windows, in the repo:  uv run python scripts/nebius/probe_request.py"
    ;;
  secret)
    need_cli
    [ -n "$KEY_VALUE" ] || { echo "No NEBIUS_API_KEY in $ENV_FILE" >&2; exit 1; }
    [ -n "${NEBIUS_PARENT_ID:-}" ] || { echo "Set NEBIUS_PARENT_ID to the project id first." >&2; exit 1; }
    if "$NB" mysterybox secret get-by-name --parent-id "$NEBIUS_PARENT_ID" --name "$SECRET_NAME" \
        --format jsonpath='{.metadata.id}' >/dev/null 2>&1; then
      echo "secret $SECRET_NAME already exists in this project (not changed)"
      exit 0
    fi
    # The value goes on this one command line (seen only by this WSL user, for a moment);
    # the output is redacted and nothing is written to a file.
    payload="[{\"key\": \"NEBIUS_API_KEY\", \"string_value\": \"$KEY_VALUE\"}]"
    "$NB" mysterybox secret create --parent-id "$NEBIUS_PARENT_ID" --name "$SECRET_NAME" \
      --description "Token Factory key for the deixis probe job" \
      --secret-version-payload "$payload" 2>&1 | redact | grep -v -i "string_value" || true
    unset payload
    echo "secret $SECRET_NAME created (value not shown)"
    ;;
  dry-run)
    create --dry-run
    ;;
  create)
    create
    cat <<EOF

Created (async). It bills while it runs: a small CPU VM for about 24 h (about \$2).
Check in a few minutes:   bash scripts/nebius/probe_job.sh logs
Stop it early:            bash scripts/nebius/probe_job.sh cancel
Safety net:               setsid nohup bash scripts/nebius/probe_job.sh watchdog 26 > /tmp/probe_watchdog.log 2>&1 &
EOF
    ;;
  status)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 0; }
    "$NB" ai job get "$id" --format json 2>&1 | redact | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    print(raw); sys.exit(0)
print("id:", d.get("metadata", {}).get("id"))
print("status:", json.dumps(d.get("status", {}), indent=1))'
    ;;
  logs)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 0; }
    "$NB" ai job logs "$id" --tail "${2:-20}" --timestamps 2>&1 | redact
    ;;
  fetch-logs)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 1; }
    OUT="${2:-$REPO_DIR/out/probe/job-log.txt}"
    mkdir -p "$(dirname "$OUT")"
    created="$("$NB" ai job get "$id" --format jsonpath='{.metadata.created_at}')"
    python3 - "$NB" "$id" "$created" "$OUT" "$SCRIPT_DIR" <<'PY'
import sys
from datetime import datetime, UTC
nb, job, created, out, here = sys.argv[1:6]
sys.path.insert(0, here)
from logs_to_files import cli_query, fetch_lines
start = int(datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp())
end = int(datetime.now(UTC).timestamp())
lines = fetch_lines(cli_query(nb, job), start, end, step=3600)
open(out, "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"{len(lines)} lines -> {out}")
PY
    redact < "$OUT" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"  # belt and braces: probe.py prints no key
    echo "Next, on Windows: uv run python scripts/probe_summary.py out/probe/job-log.txt"
    ;;
  cancel)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 0; }
    "$NB" ai job cancel "$id" 2>&1 | redact
    echo "cancelled $id. Double-check: bash scripts/nebius/probe_job.sh list"
    ;;
  delete)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME (already deleted)"; exit 0; }
    "$NB" ai job delete "$id" 2>&1 | redact
    echo "deleted $id"
    ;;
  list)
    need_cli
    echo "== jobs =="
    "$NB" ai job list ${scope_flags[@]+"${scope_flags[@]}"} --format table 2>&1 | redact
    echo "== endpoints =="
    "$NB" ai endpoint list ${scope_flags[@]+"${scope_flags[@]}"} 2>&1 | redact
    ;;
  watchdog)
    H="${2:-26}"
    echo "watchdog: will cancel $NAME in $H h (started $(date +%H:%M:%S))"
    sleep "$((H * 3600))"
    echo "watchdog: cancelling after $H h"
    bash "$SELF" cancel
    ;;
  *)
    sed -n '2,40p' "$SELF" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
