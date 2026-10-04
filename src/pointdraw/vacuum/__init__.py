"""The vacuum app: "clean here" with an unmodified robot vacuum (plan U2).

A camera high up sees the floor (four floor markers) and the person's feet; speech gives
timed words; the agent turns them into the vacuum's tool calls; the deixis core (over MCP)
turns "here" into a box on the floor; the app previews it, waits for "yes" and for the
person to step out, maps the box onto the robot's own map and sends it to the robot's MCP
server (`deixis-vacuum-mcp`, a separate GPL repo).
"""
