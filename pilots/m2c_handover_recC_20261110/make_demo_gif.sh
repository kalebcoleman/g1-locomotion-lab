#!/bin/bash
# M2c hero post-processing: crop the fixed wide camera to the robot's path, trim, add phase labels, build the GIF (<= 5.5 MB, about 1 MB in practice).
# usage: make_demo_gif.sh [source_mp4]    (default media/portfolio_hero1_s5_0000.mp4, written by run_demo.sh)
#   crop 640x560 at (440,160) of the 1280x720 frame: keeps the robot and its feet in view from the push (at x~660, y~400) to the walk toward the camera; checked on
#   a contact sheet of the committed GIF (media/hero_s5_sheet.png). Trim 0.8-12.0 s (the robot leaves the bottom of the frame at about 13 s).
#   Phase labels (video time, from the result JSON): push 1.50 s, fall detected 2.14 s, hand-over 10.14 s; GIF time = video time - 0.8 s.
#   encode: 10 fps, 560 px wide, 64-colour palette.
set -e
P=$(cd "$(dirname "$0")" && pwd); SRC=${1:-$P/media/portfolio_hero1_s5_0000.mp4}
F=${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}; T=$(mktemp -d)
printf 'pushed' > $T/1; printf 'fall detected' > $T/2; printf 'getting up (learned policy)' > $T/3; printf 'walking policy takes over' > $T/4
dt(){ echo "drawtext=textfile=$T/$1:expansion=none:fontfile=$F:fontcolor=white:fontsize=17:box=1:boxcolor=0x14181dcc:boxborderw=6:x=10:y=10:enable='$2'"; }
ffmpeg -hide_banner -loglevel error -y -ss 0.8 -to 12.0 -i "$SRC" \
  -vf "crop=640:560:440:160,scale=560:-1:flags=lanczos,$(dt 1 "between(t,0.65,1.34)"),$(dt 2 "between(t,1.34,2.2)"),$(dt 3 "between(t,2.2,9.34)"),$(dt 4 "gte(t,9.34)"),fps=10,split[s0][s1];[s0]palettegen=max_colors=64:stats_mode=diff[p];[s1][p]paletteuse=dither=none" \
  -loop 0 $P/media/demo_fall_getup_walk.gif
ls -la $P/media/demo_fall_getup_walk.gif
