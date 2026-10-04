"""Physical constants that printed sheets, vision and the controller must agree on.

Marker *positions* are deliberately not fixed here. Markers are stuck on the bed anywhere
around the paper, and calibration learns where they are from crosses the plotter draws
(plan section 4.3). Only the dictionary, IDs and printed size are fixed.

Frames: the bed frame is machine XY in mm (the paper is fixed to the bed). The paper frame
is the bed frame as the user sees the sheet: +x to the user's right, +y away from the user.
"""

ARUCO_DICTIONARY = "DICT_4X4_50"

BED_MARKER_IDS = tuple(range(0, 8))  # small stickers on the plotter bed
BED_MARKER_SIZE_MM = 20.0  # edge of the black square, as printed

FLOOR_MARKER_IDS = tuple(range(10, 14))  # A4 sheets on the floor (room scale, T2.9)
FLOOR_MARKER_SIZE_MM = 150.0

ROBOT_MARKER_ID = 14  # on the vacuum's lid, printed top toward the robot's front (calibration)
ROBOT_MARKER_SIZE_MM = 100.0

PAPER_WIDTH_MM = 210.0  # A5 landscape
PAPER_HEIGHT_MM = 148.0
