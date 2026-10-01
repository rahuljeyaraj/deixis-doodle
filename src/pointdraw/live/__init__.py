"""Glue for running the app on the real plotter, camera and microphone (`scripts/live_run.py`).

- `view`: camera pixels to bed mm with the bed's travel since calibration, and the top-down
  paper image the runtime looks at.
- `agent`: asks the agent in a background thread, so frames keep flowing while it thinks.
- `speech`: microphone to words, in a background thread.

The plotter side is `pointdraw.runtime.live_plotter.LivePlotter`.
"""
