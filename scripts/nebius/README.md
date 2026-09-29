# Nebius scripts (run from WSL)

The Nebius CLI has no native Windows build, so these scripts run inside WSL Ubuntu, where the CLI is
already installed at `~/.nebius/bin/nebius` (profile `default`, AI Cloud project in uk-south2).

**Everything here except `settings` and `dry-run` can cost money.** A GPU endpoint bills per second
while it exists: about $1.80/h on `gpu-rtx6000-a` ($43 a day if forgotten). Always finish with
`delete`, and start a `watchdog` as a safety net.

## One-time setup

1. The repo `.env` (at `C:\work\deixis-doodle\.env`, gitignored) must hold the endpoint token as
   `COSMOS_API_KEY=` (or the older name `COSMOS_ENDPOINT_TOKEN=`). Any long random string works:
   `openssl rand -hex 32`. The script redacts it from everything it prints.
2. Optional, for gated Hugging Face models (Cosmos-Reason2): add `HF_TOKEN=` to `.env`. It is passed
   to the container with `--env` and redacted too.
3. Open WSL in the repo: `wsl`, then `cd /mnt/c/work/deixis-doodle`.

## Cosmos endpoint: `cosmos_endpoint.sh`

```bash
bash scripts/nebius/cosmos_endpoint.sh settings   # what would be created (free)
bash scripts/nebius/cosmos_endpoint.sh dry-run    # the CLI validates the request (free)
bash scripts/nebius/cosmos_endpoint.sh create     # BILLING STARTS; about 15 min cold start
setsid nohup bash scripts/nebius/cosmos_endpoint.sh watchdog 90 > /tmp/cosmos_watchdog.log 2>&1 &
bash scripts/nebius/cosmos_endpoint.sh wait 40    # polls every 30 s until /v1/models answers 200
bash scripts/nebius/cosmos_endpoint.sh url        # the https URL; put it in .env as COSMOS_BASE_URL
# ... run the R1 eval or the app from Windows (uv run python scripts/r1_eval.py ...) ...
bash scripts/nebius/cosmos_endpoint.sh delete     # ALWAYS
~/.nebius/bin/nebius ai endpoint list             # double-check that nothing is left
~/.nebius/bin/nebius compute disk list
```

Other commands: `status`, `logs [N]`, `stop`, `start`. A stopped endpoint may still bill for its disk,
so prefer `delete`. Don't `pkill -f` the watchdog from a shell whose own command line matches it.

### Profiles

Pick one with `COSMOS_PROFILE=<name>` in front of the command.

| Profile | Model | Platform / preset | Region | Status |
|---|---|---|---|---|
| `nano` (default) | `nvidia/Cosmos3-Nano` | `gpu-rtx6000-a` / `1gpu-24vcpu-218gb` | uk-south2 | Worked 22 Sep: precise boxes, 1.15 s median |
| `reason2` | `nvidia/Cosmos-Reason2-8B` | `gpu-l40s-a` / `1gpu-8vcpu-32gb` | **eu-north1 only**: set `NEBIUS_PARENT_ID` to a project there | Untested |
| `edge` | `nvidia/Cosmos3-Edge` | `gpu-rtx6000-a` / `1gpu-24vcpu-218gb` | uk-south2 | Untested; reasons by default |

Any setting can be overridden with an environment variable: `ENDPOINT_NAME`, `MODEL`, `IMAGE`,
`PLATFORM`, `PRESET`, `DISK_SIZE`, `SHM_SIZE`, `PREEMPTIBLE`, `VLLM_ARGS`, `NEBIUS_PARENT_ID`,
`NEBIUS_SUBNET_ID`, `NEBIUS_BIN`, `ENV_FILE`. For example, Cosmos-Reason2-8B on the uk-south2 GPU:

```bash
COSMOS_PROFILE=reason2 PLATFORM=gpu-rtx6000-a PRESET=1gpu-24vcpu-218gb \
  bash scripts/nebius/cosmos_endpoint.sh dry-run
```

Details, costs and prompt formats for each option are in `docs/spikes/r1-cosmos.md`.

## Line endings

`.gitattributes` in this folder keeps `*.sh` files LF-only. If WSL reports `$'\r': command not found`,
run `sed -i 's/\r$//' scripts/nebius/*.sh`.
