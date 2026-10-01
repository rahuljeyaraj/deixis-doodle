# Attribution

## Drawings: Quick, Draw!

The doodles in `quickdraw/` and `candidates/` come from the
[Quick, Draw! dataset](https://github.com/googlecreativelab/quickdraw-dataset) by
Google Creative Lab, made available under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

Changes made: we selected drawings from the "simplified" files (recognised drawings only),
flipped the y axis, moved each drawing's bounding box to the origin, and dropped the
country and timestamp fields. Each variant keeps its original `key_id`. At draw time the
strokes are smoothed (Chaikin corner cutting) and scaled.

## Drawings: generated with SANA

The line drawings in `sana/` were generated ahead of time (never during a demo) with NVIDIA's
[SANA 1600M](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px_diffusers)
(`Efficient-Large-Model/Sana_1600M_1024px_diffusers`), released under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (model card checked on
1 Oct 2026). Its text encoder, Gemma-2-2B-IT, is used under Google's
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
[Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy). The images were
made on a Nebius AI Cloud Serverless Job, then traced to single pen lines (centre-line
tracing, `scripts/trace_library.py`); Cosmos3-Nano labels ranked them and a person picked
the variants. Each variant keeps its image name (`<object>_s<seed>`) as its `key_id`, with
its prompt and seed.

## Lettering: Hershey fonts

Labels are drawn with the Hershey vector fonts (Dr A. V. Hershey, US National Bureau of
Standards, 1967), as packaged by [vpype](https://github.com/abey79/vpype) (MIT licence),
whose font data comes from Michael Fogleman's [axi](https://github.com/fogleman/axi) (MIT).
The Hershey fonts are free to use; this credit is given as their usage terms ask.
