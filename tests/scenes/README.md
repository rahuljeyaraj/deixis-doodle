# Virtual-world scenes

One YAML file per scene (`S<nn>-<name>.yaml`). The runner is `scripts/virtual_run.py`; the
tests are `uv run pytest -m virtual`. The plan is `docs/virtual-test-plan.md`.

All positions are **paper mm**: x to the user's right, y away from the user, origin at the
sheet's bottom-left corner as the user sees it (A5: 210 × 148). The drawable area is the
paper minus 8 mm. All times are seconds from the start of the scene.

```yaml
id: S2
title: "A cat here, this big"
tests: "what this scene proves"
voice: text            # text (default), tts or both: which voice modes it is meant for

lines:                 # what is said
  - at: 1.6            # speech starts here; words are timed at a natural rate
    text: "A cat here, this big."
  - at: 5.0
    text: "Yes."
    words: [[yes, 5.0, 5.3]]   # optional: exact word times [word, t0, t1]

hand:                  # the fingertip, as keyframes, in order
  - {at: 0.3, enter: [150, -40]}   # the hand appears (pointing) at this point
  - {move: [60, 90], dur: 0.6}     # moves there (eased) over dur seconds
  - {hold: 1.0}                    # stays still
  - {pinch: 45}                    # shows a size: thumb 45 mm from the index tip
  - {two_hands: 80}                # a second index finger 80 mm to the left
  - {one_hand: true}
  - {pose: open}                   # point | pinch | open | fist (open and fist don't point)
  - {height: 20}                   # finger this high above the paper (parallax)
  - {leave: true, dur: 0.4}        # moves off towards the user and disappears
  - {wait: 2.0}                    # nothing for a while

sheet:                 # what is on the paper at the start
  robot:               # earlier robot drawings (the runtime knows them exactly)
    - {object: house, at: [60, 80], size: 45}
    - {label: "ASR", at: [40, 20], size: 5}
  user:                # the user's own pen marks (only the camera sees them)
    - {shape: square, at: [140, 70], size: 40, ink: black pen, label: box}

events:                # things that happen during the scene
  - {at: 2.0, user_mark: {shape: circle, at: [100, 70], size: 30}}

expect:
  calls: [place_drawing, adjust, confirm]   # tool names, in this order (others may be between)
  drawn:                                    # every drawing, in the order it was drawn
    - {kind: object, label: cat, center: [60, 90], tol_mm: 5, size: 45, size_tol: 0.2}
    - {kind: label, text: nemotron, below: brain}
    - {kind: arrow, from: ear, to: brain}
    - {kind: object, label: hand, inside: box}
  asks: 0                                   # exact number of questions
  ask_contains: ["house"]
  ask_options: [house, tree]
  states: [PREVIEW, DRAW]                   # visited in this order
  look_contains: [circle]
  say_contains: ["Stopped"]
  stop: false        # a stop must happen within the limits
  pause: false       # the interlock must pause and resume a drawing
  scene_empty: false # the scene is empty at the end (new sheet)

variations:          # defaults shown; each adds a run
  seeds: [1, 2]      # other jitter and dropout seeds
  hover_mm: 8        # the finger this high above the paper
  shifts: [-0.2, 0.2]  # speech moved this much against the gesture

plotter: {emergency_parser: false}
max_time: 90
```

Every scene is also checked for: no motion while a hand is over the paper, pause and
stop within 1.7 s and 50 mm, all ink inside the drawable area, and whitelisted G-code only.

**Shapes for user marks:** see `SHAPES` in `src/pointdraw/scene/synth.py` (circle, square,
triangle, star, heart, arrow, house, tree, ...). **Inks:** pencil, black pen, blue crayon,
red crayon, green crayon, orange crayon.

**Agent replies in mock mode** come from `agent_cache.json` (recorded by
`virtual_run.py --ai live --record`) and otherwise from the rule-based fallback parser.
