#!/bin/bash
# M2c demo post-processing: crop the fixed camera to the env-0 robot, trim to the sequence, add phase labels, and build the GIF (<= 5.5 MB).
# usage: make_demo_gif.sh [source_mp4]    (default media/demo_blend_try1_0000.mp4, the recording behind the shipped clip)
#   crop 780x520 at (270,170): lower than the first crop (760x510 at 280,30), which cut off the robot's feet. NOT yet re-checked against the source
#   recording (it is no longer kept); check a contact sheet of the output before using it.
#   trim 2.6-14.0 s: push at 3.0 s, fall detected at 3.84 s, hand-over at 10.84 s, first 3 s of the walk after it. Labels use GIF time = sim time - 2.6 s.
#   encode: 10 fps, 560 px wide, 64-colour palette.
set -e
P=$(cd "$(dirname "$0")" && pwd); SRC=${1:-$P/media/demo_blend_try1_0000.mp4}
F=${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}; T=$(mktemp -d)
printf 'pushed' > $T/1; printf 'fall detected' > $T/2; printf 'getting up (learned policy)' > $T/3; printf 'walking policy takes over' > $T/4
dt(){ echo "drawtext=textfile=$T/$1:expansion=none:fontfile=$F:fontcolor=white:fontsize=22:box=1:boxcolor=0x14181dcc:boxborderw=6:x=14:y=14:enable='between(t,$2,$3)'"; }
ffmpeg -hide_banner -loglevel error -y -ss 2.6 -to 14.0 -i "$SRC" -vf "crop=780:520:270:170" -c:v libx264 -crf 20 -pix_fmt yuv420p -an $T/clip.mp4
ffmpeg -hide_banner -loglevel error -y -i $T/clip.mp4 \
  -vf "scale=560:-1:flags=lanczos,$(dt 1 0 1.2),$(dt 2 1.2 2.4),$(dt 3 2.4 8.2),$(dt 4 8.2 12),fps=10,split[s0][s1];[s0]palettegen=max_colors=64:stats_mode=diff[p];[s1][p]paletteuse=dither=none" -loop 0 $P/media/demo_fall_getup_walk.gif
ls -la $P/media/demo_fall_getup_walk.gif
