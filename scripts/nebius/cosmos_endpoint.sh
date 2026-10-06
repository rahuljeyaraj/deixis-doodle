#!/usr/bin/env bash
# Cosmos vision-language endpoint on Nebius AI Cloud Serverless (vLLM, OpenAI-compatible).
#
# Run inside WSL (the Nebius CLI has no native Windows build). See scripts/nebius/README.md.
# Ported from rahul's working 22 Sep script (C:\work\test_nebius_nvidia\cosmos_endpoint.sh),
# itself based on github.com/nebius/serverless-ai-cookbook templates/endpoint-cosmos3-reasoner.
#
# Usage:
#   bash scripts/nebius/cosmos_endpoint.sh <command> [arg]
#
# Commands:
#   dry-run            validate the create request (free, creates nothing)
#   create             create the endpoint (BILLING STARTS; ~15 min cold start for Cosmos3-Nano)
#   status             print id, state and URLs
#   url                print only the https URL (for COSMOS_BASE_URL)
#   wait [N]           poll every 30 s, up to N times (default 60), until /v1/models answers 200
#   logs [N]           last N log lines (default 100)
#   stop | start       stop or restart the endpoint (stop may still bill for the disk)
#   delete             delete the endpoint (ALWAYS do this when you are finished)
#   watchdog [MIN]     sleep MIN minutes (default 60), then delete. Run it in the background
#   settings           print the resolved settings (token redacted) and exit
#
# Profiles (COSMOS_PROFILE, default "nano"):
#   nano     nvidia/Cosmos3-Nano on gpu-rtx6000-a / 1gpu-24vcpu-218gb, uk-south2 (worked 22 Sep)
#   reason2  nvidia/Cosmos-Reason2-8B on gpu-l40s-a / 1gpu-8vcpu-32gb (eu-north1 only: set
#            NEBIUS_PARENT_ID to a project in eu-north1; may need HF_TOKEN in .env)
#   edge     nvidia/Cosmos3-Edge on gpu-rtx6000-a / 1gpu-24vcpu-218gb, uk-south2 (untested)
#
# Any single setting can be overridden with an environment variable:
#   ENDPOINT_NAME, MODEL, IMAGE, PLATFORM, PRESET, DISK_SIZE, SHM_SIZE, PREEMPTIBLE (true|false),
#   VLLM_ARGS (replaces the whole vLLM argument string), NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID,
#   NEBIUS_BIN (default ~/.nebius/bin/nebius), ENV_FILE (default <repo>/.env).
#
# The auth token comes from the repo's .env: COSMOS_API_KEY, falling back to COSMOS_ENDPOINT_TOKEN.
# It is redacted from everything this script prints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
NB="${NEBIUS_BIN:-$HOME/.nebius/bin/nebius}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"
PROFILE="${COSMOS_PROFILE:-nano}"

# ---------------------------------------------------------------------------------------------
# Profiles. The "nano" defaults are exactly the 22 Sep settings that worked.
# ---------------------------------------------------------------------------------------------
# Session times for the measure list (M21: cold start, and the idle share of billed time with
# the per-call log out/cosmos/calls.jsonl that CosmosClient writes): one line per event.
stamp() {
  mkdir -p "$REPO_DIR/out/cosmos"
  printf '{"utc": "%s", "event": "%s", "endpoint": "%s", "profile": "%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$NAME" "$PROFILE" >> "$REPO_DIR/out/cosmos/sessions.jsonl"
}

case "$PROFILE" in
  nano)
    D_NAME=cosmos3-nano-reasoner
    D_MODEL=nvidia/Cosmos3-Nano
    D_IMAGE=vllm/vllm-openai:v0.29.0
    D_PLATFORM=gpu-rtx6000-a
    D_PRESET=1gpu-24vcpu-218gb
    D_DISK=500Gi
    D_ARGS_EXTRA="--tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling"
    ;;
  nano-video)
    # nano, plus video allowed explicitly: use it only if the Week 1 video check
    # (scripts/cosmos_video_check.py ask) is refused on "nano". Not yet tried (6 Oct).
    D_NAME=cosmos3-nano-reasoner
    D_MODEL=nvidia/Cosmos3-Nano
    D_IMAGE=vllm/vllm-openai:v0.29.0
    D_PLATFORM=gpu-rtx6000-a
    D_PRESET=1gpu-24vcpu-218gb
    D_DISK=500Gi
    D_ARGS_EXTRA="--tensor-parallel-size 1 --mm-encoder-tp-mode data --async-scheduling --limit-mm-per-prompt.image 2 --limit-mm-per-prompt.video 1"
    ;;
  reason2)
    D_NAME=cosmos-reason2-8b
    D_MODEL=nvidia/Cosmos-Reason2-8B
    D_IMAGE=vllm/vllm-openai:v0.29.0
    D_PLATFORM=gpu-l40s-a
    D_PRESET=1gpu-8vcpu-32gb
    D_DISK=250Gi
    D_ARGS_EXTRA="--tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.9 --reasoning-parser qwen3"
    ;;
  edge)
    D_NAME=cosmos3-edge-reasoner
    D_MODEL=nvidia/Cosmos3-Edge
    D_IMAGE=vllm/vllm-openai:v0.29.0
    D_PLATFORM=gpu-rtx6000-a
    D_PRESET=1gpu-24vcpu-218gb
    D_DISK=250Gi
    D_ARGS_EXTRA="--tensor-parallel-size 1 --max-model-len 32768"
    ;;
  *)
    echo "unknown COSMOS_PROFILE '$PROFILE' (use nano, nano-video, reason2 or edge)" >&2
    exit 2
    ;;
esac

NAME="${ENDPOINT_NAME:-$D_NAME}"
MODEL="${MODEL:-$D_MODEL}"
IMAGE="${IMAGE:-$D_IMAGE}"
PLATFORM="${PLATFORM:-$D_PLATFORM}"
PRESET="${PRESET:-$D_PRESET}"
DISK_SIZE="${DISK_SIZE:-$D_DISK}"
SHM_SIZE="${SHM_SIZE:-16Gi}"
PREEMPTIBLE="${PREEMPTIBLE:-true}"
VLLM_ARGS="${VLLM_ARGS:-serve $MODEL $D_ARGS_EXTRA --host 0.0.0.0 --port 8000}"

# ---------------------------------------------------------------------------------------------
# Secrets: read from .env without sourcing it, and never print them.
# ---------------------------------------------------------------------------------------------
env_value() {
  # Last assignment of $1 in the env file, with CR, surrounding quotes and spaces stripped.
  [ -f "$ENV_FILE" ] || return 0
  grep -E "^[[:space:]]*$1[[:space:]]*=" "$ENV_FILE" | tail -n 1 | cut -d= -f2- \
    | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' \
    -e "s/^'\(.*\)'$/\1/" || true
}

TOKEN="$(env_value COSMOS_API_KEY)"
[ -n "$TOKEN" ] || TOKEN="$(env_value COSMOS_ENDPOINT_TOKEN)"
HF_TOKEN_VALUE="${HF_TOKEN:-$(env_value HF_TOKEN)}"

redact() {
  # Replace every occurrence of the secrets with a marker. Pure bash, so any character is safe.
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$TOKEN" ] && line="${line//"$TOKEN"/***REDACTED***}"
    [ -n "$HF_TOKEN_VALUE" ] && line="${line//"$HF_TOKEN_VALUE"/***REDACTED***}"
    printf '%s\n' "$line"
  done
}

need_token() {
  if [ -z "$TOKEN" ]; then
    echo "No token: set COSMOS_API_KEY (or COSMOS_ENDPOINT_TOKEN) in $ENV_FILE." >&2
    echo "Generate one with: openssl rand -hex 32" >&2
    exit 1
  fi
}

need_cli() {
  if [ ! -x "$NB" ]; then
    echo "Nebius CLI not found at $NB. Run this script inside WSL, or set NEBIUS_BIN." >&2
    exit 1
  fi
}

scope_flags=()
[ -n "${NEBIUS_PARENT_ID:-}" ] && scope_flags+=(--parent-id "$NEBIUS_PARENT_ID")

warn_delete() {
  cat <<EOF

  !!! The endpoint bills per second while it exists ($PLATFORM: about \$1.55-1.80/h).
  !!! When finished:   bash scripts/nebius/cosmos_endpoint.sh delete
  !!! Safety net:      setsid nohup bash scripts/nebius/cosmos_endpoint.sh watchdog 90 \\
  !!!                    > /tmp/cosmos_watchdog.log 2>&1 &
EOF
}

print_settings() {
  cat <<EOF
profile:     $PROFILE
name:        $NAME
model:       $MODEL
image:       $IMAGE
platform:    $PLATFORM / $PRESET  (preemptible=$PREEMPTIBLE)
disk / shm:  $DISK_SIZE / $SHM_SIZE
vllm args:   $VLLM_ARGS
parent id:   ${NEBIUS_PARENT_ID:-<CLI profile default>}
subnet id:   ${NEBIUS_SUBNET_ID:-<default>}
env file:    $ENV_FILE
token:       $([ -n "$TOKEN" ] && echo "set (redacted)" || echo MISSING)
HF token:    $([ -n "$HF_TOKEN_VALUE" ] && echo "set (redacted)" || echo "not set")
EOF
}

endpoint_id() {
  "$NB" ai endpoint get-by-name ${scope_flags[@]+"${scope_flags[@]}"} --name "$NAME" \
    --format jsonpath='{.metadata.id}' 2>/dev/null || true
}

endpoint_url() {
  local id="$1"
  "$NB" ai endpoint get --id "$id" --format json 2>/dev/null \
    | grep -oE 'https://[A-Za-z0-9._:/-]+' | head -n 1 | sed 's:/*$::' || true
}

create() {
  need_cli
  need_token
  local flags=(
    --name "$NAME"
    --image "$IMAGE"
    --platform "$PLATFORM"
    --preset "$PRESET"
    --container-port 8000
    --shm-size "$SHM_SIZE"
    --disk-size "$DISK_SIZE"
    --auth token --token "$TOKEN"
    --container-command vllm
    --args "$VLLM_ARGS"
  )
  [ "$PREEMPTIBLE" = "true" ] && flags+=(--preemptible)
  [ -n "${NEBIUS_SUBNET_ID:-}" ] && flags+=(--subnet-id "$NEBIUS_SUBNET_ID")
  [ -n "$HF_TOKEN_VALUE" ] && flags+=(--env "HF_TOKEN=$HF_TOKEN_VALUE")
  print_settings
  echo
  "$NB" ai endpoint create ${scope_flags[@]+"${scope_flags[@]}"} "${flags[@]}" --async "$@" 2>&1 \
    | redact
}

status() {
  need_cli
  local id
  id="$(endpoint_id)"
  if [ -z "$id" ]; then
    echo "no endpoint named $NAME"
    return 0
  fi
  "$NB" ai endpoint get --id "$id" --format json 2>&1 | redact | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    print(raw); sys.exit(0)
s = d.get("status", {})
print("id:", d.get("metadata", {}).get("id"))
print("state:", s.get("state"))
print("urls:", s.get("public_endpoints", []))'
}

case "${1:-}" in
  settings)
    print_settings
    ;;
  dry-run)
    create --dry-run
    ;;
  create)
    create
    stamp created
    echo
    echo "Created (async). Next: bash scripts/nebius/cosmos_endpoint.sh wait 40"
    warn_delete
    ;;
  status)
    status
    ;;
  url)
    need_cli
    id="$(endpoint_id)"
    [ -z "$id" ] && { echo "no endpoint named $NAME" >&2; exit 1; }
    endpoint_url "$id"
    ;;
  wait)
    need_cli
    need_token
    for _ in $(seq 1 "${2:-60}"); do
      S="$(status 2>&1 || true)"
      STATE="$(printf '%s\n' "$S" | awk '/^state:/{print $2}' || true)"
      URL="$(printf '%s\n' "$S" | grep -oE 'https://[A-Za-z0-9._:-]+' | head -n 1 || true)"
      CODE=000
      if [ -n "$URL" ]; then
        CODE="$(curl -s -o /dev/null -w '%{http_code}' -m 10 \
          -H "Authorization: Bearer $TOKEN" "$URL/v1/models" || true)"
      fi
      echo "$(date +%H:%M:%S) state=${STATE:-?} http=$CODE url=${URL:-none}"
      if [ "$CODE" = 200 ]; then
        stamp ready
        echo "READY"
        echo "Put this in .env:  COSMOS_BASE_URL=$URL"
        warn_delete
        exit 0
      fi
      case "$STATE" in
        *ERROR*|*FAIL*|*STOPPED*|*DELET*) echo "BAD STATE"; printf '%s\n' "$S"; exit 1 ;;
      esac
      sleep 30
    done
    echo "TIMEOUT waiting (the endpoint is still billing: delete it if you are giving up)"
    exit 1
    ;;
  logs)
    need_cli
    id="$(endpoint_id)"
    [ -z "$id" ] && { echo "no endpoint named $NAME"; exit 0; }
    "$NB" ai endpoint logs "$id" --tail "${2:-100}" --timestamps 2>&1 | redact
    ;;
  stop|start)
    need_cli
    id="$(endpoint_id)"
    [ -z "$id" ] && { echo "no endpoint named $NAME"; exit 0; }
    "$NB" ai endpoint "$1" --id "$id" 2>&1 | redact
    [ "$1" = start ] && warn_delete
    ;;
  delete)
    need_cli
    id="$(endpoint_id)"
    [ -z "$id" ] && { echo "no endpoint named $NAME (already deleted)"; exit 0; }
    "$NB" ai endpoint delete --id "$id" 2>&1 | redact
    stamp deleted
    echo "deleted $id. Double-check: $NB ai endpoint list; $NB compute disk list"
    ;;
  watchdog)
    MIN="${2:-60}"
    echo "watchdog: will delete $NAME in $MIN min (started $(date +%H:%M:%S))"
    sleep "$((MIN * 60))"
    echo "watchdog: deleting after $MIN min"
    bash "$SELF" delete
    ;;
  *)
    sed -n '2,33p' "$SELF" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
