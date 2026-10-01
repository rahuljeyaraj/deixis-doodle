"""Prompts for the three R1 tasks (plan 5.1) and for the app's `look` tool.

- (a) label_crop: "what is drawn here?" on a crop around the pointed spot. Matters most, because
  precise geometry comes from ink segmentation; Cosmos supplies the meaning.
- (b) point_to: a semantic place, e.g. "empty space right of the house".
- (c) find: boxes for a named mark, e.g. "the circle".

Grounding wording follows the Cosmos 3 Reasoner cookbook example ("Locate ... Return a json.")
and the 0-1000 `bbox_2d` / `point_2d` keys used by Cosmos3 and Qwen3-VL-family models. Image
blocks go before the text in the message, as the cookbook recommends.
"""

from __future__ import annotations

PROMPT_VERSION = "r1-v0"

LABEL_CROP = (
    "This is a close-up photo, taken from directly above, of a small hand-drawn mark on white "
    "paper. What does the mark depict? Answer with the most likely everyday object or shape "
    "name, one to three words (for example: cat, house, circle, arrow, star, smiley face). "
    "Ignore the paper, pen strokes that are cut off at the edge, and any printed text. "
    'Return a json: {"label": "<name>", "alternatives": ["<name>", "<name>"]}.'
)

POINT_TO = (
    "This photo shows a sheet of paper from directly above, with hand-drawn marks on it. "
    "Up in the image is away from the viewer. {query} "
    "Point to one spot on the paper that satisfies this, not on top of any existing mark. "
    'Return a json: [{{"point_2d": [x, y], "label": "<short description>"}}] with coordinates '
    "normalised to 0-1000."
)

FIND = (
    "This photo shows a sheet of paper from directly above, with hand-drawn marks on it. "
    "Locate the accurate bounding box of {query} as a whole. If there are several, return each "
    "one. If there is none, return []. "
    'Return a json: [{{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}}] with coordinates '
    "normalised to 0-1000."
)

FIND_ALL = (
    "This photo shows a sheet of paper from directly above. Locate every separate hand-drawn "
    "mark or drawing on the paper, but not the paper itself, the desk or anything off the paper. "
    'Return a json: [{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}] with coordinates '
    "normalised to 0-1000."
)

NO_THINK_SUFFIX = " Answer directly, without explaining your reasoning."

LOOK = (
    "This is a sheet of paper seen from directly above. {robot}"
    'The person at the table asks: "{query}" Answer them in one short spoken sentence.'
)


def look_prompt(query: str, robot_drew: list[str] | None = None) -> str:
    """The app's `look` tool: a free question about the paper, answered in one sentence.
    `robot_drew` names the robot's own drawings, so "what did I draw?" can leave them out."""
    robot = ""
    if robot_drew:
        names = [f"the {x}" for x in robot_drew]
        listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        robot = f"The robot drew {listed}; the person drew everything else. "
    q = query.strip()
    if q and q[-1] not in ".?!":
        q += "?"
    return LOOK.format(robot=robot, query=q)


def label_crop_prompt() -> str:
    return LABEL_CROP


def point_to_prompt(query: str) -> str:
    q = query.strip()
    if q and q[-1] not in ".?!":
        q += "."
    return POINT_TO.format(query=q[:1].upper() + q[1:])


def find_prompt(query: str) -> str:
    q = query.strip().rstrip(".?!")
    if not q:
        return FIND_ALL
    if not q.lower().startswith(("the ", "a ", "an ", "every ", "all ")):
        q = "the " + q
    return FIND.format(query=q)
