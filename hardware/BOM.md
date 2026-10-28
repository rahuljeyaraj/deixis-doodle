# Bill of materials

What we used to turn an Ender 3 into the pen plotter, and to film the plotter and the
vacuum app. "Must" means the app needs it; "Should" and "Optional" are extras.

## The plotter and the vacuum app

| Item | Notes | Must? |
|---|---|---|
| USB webcam, 1080p, manual exposure and focus | C920/C922 class; its clip has a 1/4"-20 socket | Must |
| Desk clamp arm or boom for the camera | Separate from the printer; 1/4"-20 screw; lens 50–60 cm above the paper | Must |
| Diffuse LED panel | Even light, no glare | Must |
| USB or wireless clip-on (lavalier) mic | Bluetooth earbuds hurt speech recognition; also for the voice-over | Must |
| Latching mushroom switch (DC rated) | Emergency stop: NC contact on the +24 V line; ≥ 30 V DC, ≥ 5 A DC | Must |
| USB big button | Software stop (`M410`) | Must |
| Black fineliners, 0.5 to 0.8 mm, several brands | Test three; body 6–11 mm for the pltr pen holder | Must |
| A5 paper, 120 to 160 gsm, matte | Doesn't cockle; also for the accuracy target sheet | Must |
| Tape corners or magnets | Hold the paper; no binder clips | Must |
| 2 × 100 kΩ resistors | Stand in for the hot end's and bed's thermistors, so the firmware runs without them; 1/4 W through-hole | Must |
| Matte sticker paper | For the vacuum's lid marker (`print/robot-marker.pdf`) | Vacuum only |
| A second camera (a phone works) with a tall tripod or shelf mount | Looks down at the floor for the vacuum app | Vacuum only |
| SG90 servo and lead | A servo pen lift, instead of the Z axis | Optional |
| Phone tripod | Angled shots for the film | Should |

## Small parts for the pen holder and set-up

| Item | Notes | Must? |
|---|---|---|
| 3 × M3×10 countersunk screw | pltr base to hot-end bracket | Must |
| 2 × M3×40 socket-head bolt, 2 × M3 nyloc nut | pltr stamp guides | Must |
| 4 × M4×15 button-head screw | pltr pen clamp | Must |
| 2 × light compression spring, OD ≤ 6.5 mm, ID ≥ 3.3 mm, 15–20 mm | Ballpoint click-pen springs work | Must |
| 1–2 rubber bands | Phone tripod adapter | Should |
| microSD card ≤ 32 GB, FAT32 | Firmware flashing | Must |
| Hookup wire 18 AWG, ferrules or crimps, heat-shrink | Emergency stop in the 24 V line; insulating the bed power leads | Must |
| Backup holder parts: 2 × 3 mm × 40 mm rods, 2 × M3 heat-set inserts, M3 screws, 2 × springs (4 mm OD) | Only if the pltr holder doesn't fit | Optional |
| Ruler or calipers | Scale checks, square test, accuracy errors | Must |
