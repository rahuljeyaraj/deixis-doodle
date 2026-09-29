# R1 test photos: how to add them

R1 checks what Cosmos adds on top of our own ink segmentation (plan 5.1 and 4.3). The three tasks:

- **(a) Label a pointed crop:** "what is drawn here?" This matters most.
- **(b) Semantic place:** "empty space to the right of the house".
- **(c) Find a named mark:** "find the circle".

The synthetic set (`uv run python scripts/r1_make_synthetic.py`) runs with no photos at all. Real
photos tell us whether Cosmos copes with real pens, paper and light. About 20 photos is enough.

**Photos never get committed.** `data/r1/.gitignore` ignores images and results in this folder.

## 1. Take the photos (about 30 minutes)

- Use A5 or A4 white paper. Draw 2 to 5 separate marks per sheet, in the style the demo will use:
  - rough shapes: a box, a circle, a triangle, a star, an arrow
  - quick doodles: a house, a tree, a cat, a sun, a smiley face
  - a few with pencil, a few with a black pen, a few with crayon
- Leave some empty space on every sheet, so the "empty space next to X" questions have answers.
- Photograph each sheet **from directly above**, with the whole sheet in view, in even light and
  with no hands in the shot. Plain JPEG from the phone is fine. Around 1920 px on the long side is
  plenty; larger photos are downscaled before sending.
- Include 3 or 4 photos from the real overhead webcam at the plotter if it is mounted by then.
- Copy them into `data/r1/photos/` (e.g. `img01.jpg` … `img20.jpg`).

## 2. Label them (about 20 minutes)

```powershell
uv run python scripts/r1_label.py data/r1/photos
```

For each photo a window opens. Drag a box around one mark, press SPACE or ENTER, repeat for every
mark, then press ESC. The terminal asks for each box's **label** (the most natural name, e.g.
`smiley face`) and optional **synonyms** that also count as correct (e.g. `face, smiley`). Results go
to `data/r1/photos/labels.json`; re-running skips photos that are already labelled (`--redo` to
change them).

Then add 2 or 3 `point_queries` per photo by hand in `labels.json`: the question, and the pixel
box where a correct answer may land. Any image viewer that shows pixel coordinates works (Paint
shows them in the status bar). A point on any labelled mark counts as wrong automatically.

The format is in `labels.example.json` in this folder:

```json
{"version": 1,
 "images": [{"file": "img01.jpg",
             "marks": [{"label": "circle", "synonyms": ["ring"], "box_px": [812, 402, 1105, 690]}],
             "point_queries": [{"query": "empty space below the circle",
                                "accept_box_px": [812, 700, 1105, 980]}]}]}
```

- `box_px` is `[x1, y1, x2, y2]` in pixels of the photo as stored: x to the right, y **down**.
- `file` is relative to the `labels.json` file.
- "Above" and "below" mean as the photo is viewed (away from and towards the person drawing).

## 3. Run R1 (needs the endpoint up: about $1 for the whole run)

From WSL: `bash scripts/nebius/cosmos_endpoint.sh create`, then `wait 40`, then `url`. Put the URL
in `.env` as `COSMOS_BASE_URL`. Then from Windows:

```powershell
uv run python scripts/r1_eval.py --labels .cache/r1/synthetic/labels.json   # synthetic first
uv run python scripts/r1_eval.py --labels data/r1/photos/labels.json         # then the photos
```

Each run prints every trial and a markdown table, and writes `.cache/r1/results-*.json`. Paste the
table into `docs/spikes/r1-cosmos.md`. **Then delete the endpoint** (`cosmos_endpoint.sh delete`).
