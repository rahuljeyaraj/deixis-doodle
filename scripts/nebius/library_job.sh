#!/usr/bin/env bash
# Line-art library generation as a Nebius AI Cloud Serverless Job (SANA -> PNGs in a bucket).
#
# PREPARED, NOT RUN YET (30 Sep 2026). Plan: docs/virtual-test-plan.md sections 6 and 7.
# Run inside WSL (the Nebius CLI has no native Windows build). See scripts/nebius/README.md.
#
# Usage:
#   bash scripts/nebius/library_job.sh <command> [arg]
#
# Commands:
#   settings           print the resolved settings and the create command (free, no cloud call)
#   dry-run            the CLI validates the create request (free, creates nothing)
#   bucket             create the output bucket if missing, print its ID (storage: pennies)
#   create             create the Job (BILLING STARTS; it stops by itself, see "Time limits")
#   status             job id and state
#   logs [N]           last N log lines (default 100)
#   follow             stream the logs (Ctrl-C stops watching, not the job)
#   fetch [DIR]        download images + manifest.json from the bucket (default DIR: out/sana_<set>)
#   fetch-logs [DIR]   rebuild them from the job's log instead (checked by sha256); the way that
#                      worked on 1 Oct 2026, when every write to the bucket mount failed (EIO)
#   cancel             stop the job now (the VM stops and its disk is deleted)
#   delete             delete the finished job's record (needed before re-creating the same set)
#   list               list jobs AND endpoints: check that nothing is left running
#   watchdog [MIN]     sleep MIN minutes (default 30), then cancel. Run it in the background
#
# The pilot, step by step (about 10 minutes, about $0.10 to $0.40; see "Cost"):
#   bash scripts/nebius/library_job.sh settings
#   bash scripts/nebius/library_job.sh bucket          # once; see "Output bucket" for aws setup
#   bash scripts/nebius/library_job.sh dry-run
#   bash scripts/nebius/library_job.sh create
#   setsid nohup bash scripts/nebius/library_job.sh watchdog 30 > /tmp/library_watchdog.log 2>&1 &
#   bash scripts/nebius/library_job.sh follow          # until "done: 20 images"
#   bash scripts/nebius/library_job.sh fetch
#   bash scripts/nebius/library_job.sh list            # nothing RUNNING; then delete the record
#   # back on Windows:
#   uv run python scripts/trace_library.py --job out/sana_pilot --out-dir out/traced_pilot
#   # then write the run in docs/gpu-usage-log.md
#
# Settings (environment variables override the defaults below):
#   LIBRARY_SET (pilot|style_test|full), JOB_NAME, IMAGE, PLATFORM, PRESET, DISK_SIZE, TIMEOUT, INNER_MIN,
#   PREEMPTIBLE (true|false), MODEL (sana|sana-4.8b|flux), BUCKET, S3_ENDPOINT,
#   NEBIUS_PARENT_ID, NEBIUS_SUBNET_ID, NEBIUS_BIN (default ~/.nebius/bin/nebius),
#   ENV_FILE (default <repo>/.env; HF_TOKEN is read from it and redacted in all output)
#
# Time limits, three layers:
#   1. library_job.py stops starting images after the set's max_minutes (pilot 15, full 40) and
#      a watchdog thread in it exits 3 minutes after that;
#   2. the container command runs it under `timeout ${INNER_MIN}m` (pip install + download + run);
#   3. the Job's own --timeout (hard stop by Nebius). Its MINIMUM is 1h (docs), so 1h it is.
#   Plus the optional `watchdog` above, which cancels from your side.
#
# Cost (prices from https://docs.nebius.com/compute/resources/pricing, 30 Sep 2026; billed per
# second while the job runs): RTX PRO 6000 (gpu-rtx6000-a, uk-south2) $1.80/GPU-hour on demand,
# $0.95 preemptible. Pilot estimate: 3-6 min VM start and image pull (4.3 GB image), 1 min pip,
# 1-3 min model download (10 GB fp16), 1 min load, 20 images x 2-4 s: about 7-12 min, so
# $0.21-0.36 on demand or $0.11-0.19 preemptible. Worst case with the inner timeout (25 min):
# $0.75 on demand; if even that fails, the 1h Job timeout: $1.80 on demand, $0.95 preemptible
# (the default here, so the pilot's worst case stays under the $1 cap). Full run (41 x 10):
# about 30 min, $0.90 on demand or $0.48 preemptible; cap $3.
#
# Output bucket (one-time): the job writes to a bucket mounted at /out, because the job's own
# disk is deleted when it ends. `bucket` creates it. `fetch` needs the AWS CLI in WSL with a
# Nebius access key (https://docs.nebius.com/object-storage/quickstart):
#   export SA_ID=$(nebius iam service-account create --name object-storage-sa \
#     --format jsonpath='{.metadata.id}')
#   (add it to the editors group: nebius iam group get-by-name --name editors --parent-id
#    $TENANT_ID ...; nebius iam group-membership create --parent-id $EDITORS_GROUP_ID \
#    --member-id $SA_ID)
#   nebius iam v2 access-key create --account-service-account-id $SA_ID \
#     --description 'AWS CLI' --format json      # -> key id; then `nebius iam v2 access-key get`
#   aws configure set aws_access_key_id ...; aws configure set aws_secret_access_key ...
#   aws configure set region uk-south2; aws configure set endpoint_url $S3_ENDPOINT
#
# Sources (read 30 Sep 2026):
#   https://docs.nebius.com/serverless/jobs                    jobs overview, volumes, inject files
#   https://docs.nebius.com/serverless/jobs/manage             create, list, logs, cancel, delete;
#                                                              "billed only while the job is running"
#   https://docs.nebius.com/cli/reference/ai/job               subcommands
#   https://docs.nebius.com/cli/reference/ai/job/create        --timeout "min 1h, max 168h, default
#       24h"; --volume SOURCE:CONTAINER_PATH[:MODE]; --inject-file LOCAL:CONTAINER (64 KiB, ro);
#       --restart-policy never|on-failure; --preemptible; --dry-run; --async
#   https://docs.nebius.com/cli/reference/ai/job/logs          logs <id> --follow --tail --since
#   https://docs.nebius.com/cli/reference/ai/job/cancel        cancel <id>
#   https://docs.nebius.com/cli/reference/ai/job/get-by-name   get-by-name --name
#   https://docs.nebius.com/cli/reference/ai/job/list          list --format
#   https://docs.nebius.com/serverless/tutorials/fine-tuning   bucket volume "<bucket_ID>:/path",
#       nebius storage bucket create / get-by-name, aws s3 cp to download
#   https://docs.nebius.com/object-storage/quickstart          access keys, endpoint
#       https://storage.<region>.nebius.cloud
#
# NOT CONFIRMED (check with `dry-run` and on the day):
#   - that Serverless Jobs offer gpu-rtx6000-a / 1gpu-24vcpu-218gb in uk-south2 (it worked for
#     the Cosmos *endpoint* on 22 Sep; the jobs docs only show gpu-l40s-a examples). Fallback:
#     PLATFORM=gpu-l40s-a PRESET=1gpu-8vcpu-32gb with NEBIUS_PARENT_ID of an eu-north1 project;
#   - whether --preemptible alone is enough for jobs (the create docs also list
#     --follows-spot-price and --spot-pricing-policy-id); if the dry run complains, try
#     PREEMPTIBLE=false (worst case then $1.80, still inside the plan's $8 total);
#   - the exact quoting of --args: copied from the fine-tuning tutorial's pattern
#     (--container-command bash --args '-c "..."');
#   - the JSON field names of a job's status (`status` prints the raw status block);
#   - the S3 endpoint for uk-south2 (the docs give the pattern, with eu-north1 as the example);
#   - DISK_SIZE 100Gi (the default is 250Gi; 100Gi is plenty for a 10 GB model and a 9 GB image).
#
# Models (Hugging Face model cards, read 2026-09-30): SANA 1600M and SANA1.5 4.8B are Apache-2.0,
# not gated, text encoder Gemma-2-2B-IT under Gemma terms; FLUX.1 [schnell] is Apache-2.0 but
# GATED, so MODEL=flux (or the automatic fallback) needs HF_TOKEN in .env from an account that
# accepted its terms.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
SELF="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
NB="${NEBIUS_BIN:-$HOME/.nebius/bin/nebius}"
ENV_FILE="${ENV_FILE:-$REPO_DIR/.env}"

SET="${LIBRARY_SET:-pilot}"
case "$SET" in
  pilot) D_INNER=25 ;;
  style_test) D_INNER=25 ;;
  full) D_INNER=50 ;;
  *) echo "unknown LIBRARY_SET '$SET' (use pilot, style_test or full)" >&2; exit 2 ;;
esac

NAME="${JOB_NAME:-deixis-library-$SET}"
IMAGE="${IMAGE:-pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime}"  # Blackwell needs CUDA >= 12.8
PLATFORM="${PLATFORM:-gpu-rtx6000-a}"
PRESET="${PRESET:-1gpu-24vcpu-218gb}"
DISK_SIZE="${DISK_SIZE:-100Gi}"
TIMEOUT="${TIMEOUT:-1h}"  # the Job's hard limit; 1h is the minimum Nebius accepts
INNER_MIN="${INNER_MIN:-$D_INNER}"
PREEMPTIBLE="${PREEMPTIBLE:-true}"
MODEL="${MODEL:-sana}"
BUCKET="${BUCKET:-deixis-library}"
S3_ENDPOINT="${S3_ENDPOINT:-https://storage.uk-south2.nebius.cloud}"

# The container: install diffusers, then run the injected script under a timeout. The PyTorch
# image already has torch, Pillow and CUDA. PYTHONUNBUFFERED makes the log lines show at once.
RUN="pip install -q diffusers transformers accelerate sentencepiece pyyaml protobuf && \
timeout ${INNER_MIN}m python /job/library_job.py --prompts /job/library_prompts.yaml \
--set $SET --out /out/$SET --model $MODEL"
ARGS="-c \"$RUN\""

# ---------------------------------------------------------------------------------------------
# Secrets: read from .env without sourcing it, and never print them.
# ---------------------------------------------------------------------------------------------
env_value() {
  [ -f "$ENV_FILE" ] || return 0
  grep -E "^[[:space:]]*$1[[:space:]]*=" "$ENV_FILE" | tail -n 1 | cut -d= -f2- \
    | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' \
    -e "s/^'\(.*\)'$/\1/" || true
}

HF_TOKEN_VALUE="${HF_TOKEN:-$(env_value HF_TOKEN)}"

redact() {
  local line
  while IFS= read -r line || [ -n "$line" ]; do
    [ -n "$HF_TOKEN_VALUE" ] && line="${line//"$HF_TOKEN_VALUE"/***REDACTED***}"
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

warn_cancel() {
  cat <<EOF

  !!! The job bills per second while it runs ($PLATFORM: \$1.80/h on demand, \$0.95 preemptible).
  !!! It stops by itself (script limit, then timeout ${INNER_MIN}m, then the Job's $TIMEOUT).
  !!! To stop it now:   bash scripts/nebius/library_job.sh cancel
  !!! Safety net:       setsid nohup bash scripts/nebius/library_job.sh watchdog 30 \\
  !!!                     > /tmp/library_watchdog.log 2>&1 &
  !!! Afterwards:       bash scripts/nebius/library_job.sh list   (nothing RUNNING)
EOF
}

print_settings() {
  cat <<EOF
set:         $SET  (prompts: scripts/nebius/library_prompts.yaml)
job name:    $NAME
model:       $MODEL
image:       $IMAGE
platform:    $PLATFORM / $PRESET  (preemptible=$PREEMPTIBLE)
disk:        $DISK_SIZE
limits:      script max_minutes (yaml) < timeout ${INNER_MIN}m < Job --timeout $TIMEOUT
bucket:      $BUCKET  -> mounted at /out, images in /out/$SET
s3 endpoint: $S3_ENDPOINT
parent id:   ${NEBIUS_PARENT_ID:-<CLI profile default>}
subnet id:   ${NEBIUS_SUBNET_ID:-<default>}
HF token:    $([ -n "$HF_TOKEN_VALUE" ] && echo "set (redacted)" || echo "not set (not needed for SANA)")
command:     bash $ARGS
EOF
}

job_id() {
  "$NB" ai job get-by-name ${scope_flags[@]+"${scope_flags[@]}"} --name "$NAME" \
    --format jsonpath='{.metadata.id}' 2>/dev/null || true
}

bucket_id() {
  "$NB" storage bucket get-by-name ${scope_flags[@]+"${scope_flags[@]}"} --name "$BUCKET" \
    --format jsonpath='{.metadata.id}' 2>/dev/null || true
}

create() {
  need_cli
  local bid
  bid="$(bucket_id)"
  if [ -z "$bid" ]; then
    if [ "${1:-}" = "--dry-run" ]; then
      bid="<bucket-id>"
      echo "(no bucket named $BUCKET yet: run 'bucket' before 'create')"
    else
      echo "No bucket named $BUCKET: run 'bash scripts/nebius/library_job.sh bucket' first." >&2
      exit 1
    fi
  fi
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
    --volume "$bid:/out:rw"
    --inject-file "$SCRIPT_DIR/library_job.py:/job/library_job.py"
    --inject-file "$SCRIPT_DIR/library_prompts.yaml:/job/library_prompts.yaml"
    --env PYTHONUNBUFFERED=1
    --env PIP_ROOT_USER_ACTION=ignore
    --env PIP_DISABLE_PIP_VERSION_CHECK=1
    --container-command bash
    --args "$ARGS"
  )
  [ "$PREEMPTIBLE" = "true" ] && flags+=(--preemptible)
  [ -n "${NEBIUS_SUBNET_ID:-}" ] && flags+=(--subnet-id "$NEBIUS_SUBNET_ID")
  [ -n "$HF_TOKEN_VALUE" ] && flags+=(--env "HF_TOKEN=$HF_TOKEN_VALUE")
  print_settings
  echo
  "$NB" ai job create ${scope_flags[@]+"${scope_flags[@]}"} "${flags[@]}" --async "$@" 2>&1 \
    | redact
}

status() {
  need_cli
  local id
  id="$(job_id)"
  if [ -z "$id" ]; then
    echo "no job named $NAME"
    return 0
  fi
  "$NB" ai job get "$id" --format json 2>&1 | redact | python3 -c '
import json, sys
raw = sys.stdin.read()
try:
    d = json.loads(raw)
except ValueError:
    print(raw); sys.exit(0)
print("id:", d.get("metadata", {}).get("id"))
print("status:", json.dumps(d.get("status", {}), indent=1))'
}

case "${1:-}" in
  settings)
    print_settings
    ;;
  dry-run)
    create --dry-run
    ;;
  bucket)
    need_cli
    bid="$(bucket_id)"
    if [ -z "$bid" ]; then
      "$NB" storage bucket create ${scope_flags[@]+"${scope_flags[@]}"} --name "$BUCKET" 2>&1 \
        | redact
      bid="$(bucket_id)"
    fi
    echo "bucket $BUCKET: ${bid:-<not found>}"
    ;;
  create)
    create
    echo
    echo "Created (async). Next: bash scripts/nebius/library_job.sh follow"
    warn_cancel
    ;;
  status)
    status
    ;;
  logs)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 0; }
    "$NB" ai job logs "$id" --tail "${2:-100}" --timestamps 2>&1 | redact
    ;;
  follow)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 0; }
    "$NB" ai job logs "$id" --follow --timestamps 2>&1 | redact
    ;;
  fetch)
    DIR="${2:-$REPO_DIR/out/sana_$SET}"
    AWS="$(command -v aws || echo "$HOME/.local/bin/aws")"
    [ -x "$AWS" ] || { echo "aws CLI not found (see 'Output bucket' above)" >&2; exit 1; }
    mkdir -p "$DIR"
    AWS_PROFILE="${AWS_PROFILE:-nebius}" "$AWS" s3 sync "s3://$BUCKET/$SET/" "$DIR" \
      --endpoint-url "$S3_ENDPOINT"
    echo "fetched to $DIR. Next, on Windows:"
    echo "  uv run python scripts/trace_library.py --job out/sana_$SET --out-dir out/traced_$SET"
    ;;
  fetch-logs)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 1; }
    DIR="${2:-$REPO_DIR/out/sana_$SET}"
    created="$("$NB" ai job get "$id" --format jsonpath='{.metadata.created_at}')"
    # the logs API gives at most 1000 lines per request: logs_to_files.py reads in time slices
    python3 "$SCRIPT_DIR/logs_to_files.py" --fetch "$NB" "$id" "$created" "$DIR"
    echo "Next, on Windows:"
    echo "  uv run python scripts/trace_library.py --job out/sana_$SET --out-dir out/traced_$SET"
    ;;
  cancel)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME"; exit 0; }
    "$NB" ai job cancel "$id" 2>&1 | redact
    echo "cancelled $id. Double-check: bash scripts/nebius/library_job.sh list"
    ;;
  delete)
    need_cli
    id="$(job_id)"
    [ -z "$id" ] && { echo "no job named $NAME (already deleted)"; exit 0; }
    "$NB" ai job delete "$id" 2>&1 | redact
    echo "deleted $id (the bucket and its images stay)"
    ;;
  list)
    need_cli
    echo "== jobs =="
    "$NB" ai job list ${scope_flags[@]+"${scope_flags[@]}"} --format table 2>&1 | redact
    echo "== endpoints =="
    "$NB" ai endpoint list ${scope_flags[@]+"${scope_flags[@]}"} 2>&1 | redact
    ;;
  watchdog)
    MIN="${2:-30}"
    echo "watchdog: will cancel $NAME in $MIN min (started $(date +%H:%M:%S))"
    sleep "$((MIN * 60))"
    echo "watchdog: cancelling after $MIN min"
    bash "$SELF" cancel
    ;;
  *)
    sed -n '2,35p' "$SELF" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
