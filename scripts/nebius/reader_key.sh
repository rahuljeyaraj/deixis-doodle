#!/usr/bin/env bash
# One-time setup so `library_job.sh fetch` can download from the bucket (run by a person, in WSL).
#
#   bash scripts/nebius/reader_key.sh
#
# 1. A service account "deixis-library-reader" in the "viewers" group (read-only, not "editors").
# 2. An access key for it that expires by itself (KEY_EXPIRES, default 2026-10-15).
# 3. The key goes into the AWS CLI profile "nebius" (~/.aws/credentials, mode 600). The secret
#    is never printed. The raw reply is kept in ~/.nebius-keys/ (mode 600) and nowhere else.
#
# Needs: the Nebius CLI signed in, and the AWS CLI (`~/.local/bin/aws`). Changes the account:
# ask rahul first. Undo: nebius iam v2 access-key delete --id <key id>, and delete the account.

set -euo pipefail

NB="${NEBIUS_BIN:-$HOME/.nebius/bin/nebius}"
PROJECT="${NEBIUS_PARENT_ID:-project-e05adzyfln00btmrn4h880}"
TENANT="${NEBIUS_TENANT_ID:-tenant-e00bhxnxd3r9vbrg2t}"
NAME=deixis-library-reader
KEY_EXPIRES="${KEY_EXPIRES:-2026-10-15T00:00:00Z}"
PROFILE="${AWS_PROFILE_NAME:-nebius}"
REGION=uk-south2
S3_ENDPOINT="${S3_ENDPOINT:-https://storage.uk-south2.nebius.cloud}"

umask 077

sa_id="$("$NB" iam service-account get-by-name --parent-id "$PROJECT" --name "$NAME" \
  --format jsonpath='{.metadata.id}' 2>/dev/null || true)"
if [ -z "$sa_id" ]; then
  sa_id="$("$NB" iam service-account create --parent-id "$PROJECT" --name "$NAME" \
    --format jsonpath='{.metadata.id}')"
fi
echo "service account: $sa_id"

viewers="$("$NB" iam group get-by-name --parent-id "$TENANT" --name viewers \
  --format jsonpath='{.metadata.id}')"
"$NB" iam group-membership create --parent-id "$viewers" --member-id "$sa_id" \
  --format jsonpath='{.metadata.id}' >/dev/null 2>&1 || echo "(already in viewers, or see the console)"
echo "in group viewers: $viewers"

mkdir -p ~/.nebius-keys
raw=~/.nebius-keys/$NAME.json
"$NB" iam v2 access-key create --parent-id "$PROJECT" --account-service-account-id "$sa_id" \
  --name "$NAME-$(date +%Y%m%d%H%M)" --description "download SANA library images (read-only)" \
  --expires-at "$KEY_EXPIRES" --secret-delivery-mode inline --format json > "$raw"

# Find the key id and the secret in the reply without printing either.
python3 - "$raw" "$PROFILE" "$REGION" "$S3_ENDPOINT" <<'PY'
import configparser, json, os, sys

raw, profile, region, endpoint = sys.argv[1:]
d = json.load(open(raw))

def find(x, want):
    if isinstance(x, dict):
        for k, v in x.items():
            if want(k) and isinstance(v, str) and v:
                return v
            r = find(v, want)
            if r:
                return r
    if isinstance(x, list):
        for v in x:
            r = find(v, want)
            if r:
                return r
    return None

key_id = find(d, lambda k: "access_key_id" in k.lower())
secret = find(d, lambda k: "secret" in k.lower() and "mode" not in k.lower())
if not key_id or not secret:
    def shape(x):
        if isinstance(x, dict):
            return {k: shape(v) for k, v in x.items()}
        if isinstance(x, list):
            return [shape(v) for v in x]
        return type(x).__name__
    print("could not find the key id and secret; reply shape:", json.dumps(shape(d)))
    sys.exit(1)

aws = os.path.expanduser("~/.aws")
os.makedirs(aws, mode=0o700, exist_ok=True)
for name, section, values in (
    ("credentials", profile, {"aws_access_key_id": key_id, "aws_secret_access_key": secret}),
    ("config", f"profile {profile}", {"region": region, "endpoint_url": endpoint}),
):
    path = os.path.join(aws, name)
    cp = configparser.ConfigParser()
    cp.read(path)
    cp[section] = values
    with open(path, "w") as fh:
        cp.write(fh)
    os.chmod(path, 0o600)
print(f"AWS profile '{profile}' written (key id {key_id[:4]}..., secret not shown)")
PY

echo "expires: $KEY_EXPIRES"
echo "test:    AWS_PROFILE=$PROFILE ~/.local/bin/aws s3 ls s3://deixis-library/"
