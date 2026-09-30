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

## Lettering: Hershey fonts

Labels are drawn with the Hershey vector fonts (Dr A. V. Hershey, US National Bureau of
Standards, 1967), as packaged by [vpype](https://github.com/abey79/vpype) (MIT licence),
whose font data comes from Michael Fogleman's [axi](https://github.com/fogleman/axi) (MIT).
The Hershey fonts are free to use; this credit is given as their usage terms ask.
