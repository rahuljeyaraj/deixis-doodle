"""Short snippets of real Cosmos3-Nano replies from rahul's 22 Sep test.

Source: C:\\work\\test_nebius_nvidia\\runs\\grounding_cosmos3nano.txt and vision_cosmos3nano.txt
(nvidia/Cosmos3-Nano on a Nebius Serverless Endpoint, images 1920x1080).
"""

# Grounding prompt, page 1: fenced JSON, 0-1000 boxes.
GROUNDING_PAGE1 = """```json
[
    {"label": "paper", "bbox_2d": [166, 131, 726, 902]},
    {"label": "elephant", "bbox_2d": [251, 264, 540, 512]}
]
```"""

# Grounding prompt, page 10: two objects.
GROUNDING_PAGE10 = """```json
[
    {"label": "paper", "bbox_2d": [232, 88, 772, 844]},
    {"label": "flower", "bbox_2d": [339, 271, 439, 455]},
    {"label": "flowerpot", "bbox_2d": [380, 679, 463, 800]}
]
```"""

# Describe prompt, page 10: boxes spontaneously given as corner pairs under "position".
DESCRIBE_PAGE10 = """```json
{
  "objects": [
    {
      "name": "child's drawing",
      "position": [
        [
          235,
          90
        ],
        [
          768,
          841
        ]
      ],
      "colours": ["white", "blue", "orange", "red"]
    },
    {
      "name": "table",
      "position": [[0, 0], [998, 995]],
      "colours": ["brown"]
    }
  ],
  "largest_empty_region": [[542, 198], [688, 763]],
  "stayed_inside_lines": true
}
```"""

# Describe prompt, page 3: positions as words, no boxes.
DESCRIBE_PAGE3 = """```json
{
  "objects": [
    {"name": "car drawing", "position": "center-left", "colours": ["purple", "yellow", "blue"]},
    {"name": "paper", "position": "center", "colours": ["white"]}
  ],
  "largest_empty_region": "top-right",
  "stayed_inside_lines": null
}
```"""
