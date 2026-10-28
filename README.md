# deixis-doodle

**Tell any robot where.** Point and say "here", "that one", "this big", and a robot acts on
the place you meant.

This repo is the app for two real robots:

- **The pen plotter**, a converted Ender 3. Say "draw a cat here, this big" with your
  fingertip on the paper. It shows its guess with a pen-up hover first, then draws what you
  asked, where you pointed, at the size you showed.
- **The robot vacuum**, an off-the-shelf ECOVACS DEEBOT. Stand on a spot, say "clean here",
  say "yes", step out. It cleans a box round where you stood.

Both share one app: the camera, hand and feet tracking (MediaPipe), speech to words
(Parakeet, on this computer), and the language model (Nemotron, on Nebius Token Factory).
The place itself is worked out by the [deixis](https://github.com/rahuljeyaraj/deixis) core,
over MCP.

Entry for the Nebius x NVIDIA Global AI Hackathon, Physical AI track. To get everything in
one clone, use [deixis-robots](https://github.com/rahuljeyaraj/deixis-robots).

## Set-up

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12. The core must sit next to
this folder, at `../deixis`. The vacuum also needs
[deixis-vacuum](https://github.com/rahuljeyaraj/deixis-vacuum) at `../deixis-vacuum`, set up
as its README says.

```powershell
uv sync --all-extras
uv run pytest
copy .env.example .env                          # then put a Nebius Token Factory key in .env
uv run python scripts/asr_fetch_models.py       # the speech model, into .cache/models/
```

The Python package inside is called `pointdraw` (in `src/pointdraw`).

## Run the plotter

```powershell
uv run python scripts/live_run.py --port COM9 --backend msmf       # the real plotter
uv run python scripts/live_run.py --dry-run --backend msmf --type  # no plotter: type, print G-code
```

Before the first run, fit the camera to the plotter with `scripts/calibrate.py`: the plotter
draws a grid of crosses, parks, photographs them and saves the fit. At start the app homes
all three axes, draws the frame of the drawing area and parks.

What to buy for the plotter: [hardware/BOM.md](hardware/BOM.md).

## Run the vacuum

```powershell
uv run python scripts/vacuum_run.py --camera 1                 # the real robot
uv run python scripts/vacuum_run.py --camera 1 --robot fake    # a pretend robot, nothing moves
```

Before the first run, map the floor from the robot's own drive with `scripts/vacuum_map.py`
(a marker on the robot's lid, the camera high up). After that the camera must not move.

Each script's header lists all its options and keys.

## NVIDIA models

| Model | What it does | Where it runs |
|---|---|---|
| Nemotron 3.5 Lightning | Turns the sentence into tool calls that point at words, not coordinates | Nebius Token Factory |
| Parakeet TDT 0.6B v2 | Speech to words, with the time of each word | This computer (CPU) |
| Cosmos 3 Nano | Reads what is already drawn on the paper (off unless `--cosmos live`) | Nebius AI Cloud Serverless Endpoint |
| SANA 1600M | Made the plotter's drawings ahead of time, never live | Nebius AI Cloud Serverless Job |

The Nebius scripts are in [scripts/nebius/](scripts/nebius/README.md).

## Licence

This repo: MIT, see [LICENSE](LICENSE).

Others' work keeps its own licence:

| What | Licence | Here |
|---|---|---|
| [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) drawings, by Google Creative Lab | CC BY 4.0 | Included, in `library/` |
| Drawings made with [SANA 1600M](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers) (text encoder Gemma-2-2B-IT) | Apache-2.0 (Gemma Terms of Use) | Included, in `library/` |
| Hershey fonts, as packaged by [vpype](https://github.com/abey79/vpype) | Free to use; vpype is MIT | Used for lettering |
| [MediaPipe](https://ai.google.dev/edge/mediapipe) hand and pose models | Apache-2.0 | In the `mediapipe` package; the pose model is downloaded |
| [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), as exported by [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | CC-BY-4.0; sherpa-onnx is Apache-2.0 | Downloaded, not in this repo |
| NVIDIA Nemotron and Cosmos | NVIDIA Open Model License | Hosted APIs |
| [deixis-vacuum](https://github.com/rahuljeyaraj/deixis-vacuum) | GPL-3.0 | A separate program, reached over MCP |

How the drawings were chosen and changed: [library/ATTRIBUTION.md](library/ATTRIBUTION.md).
