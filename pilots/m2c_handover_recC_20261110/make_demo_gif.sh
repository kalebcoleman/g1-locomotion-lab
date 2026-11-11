#!/bin/bash
# M2c demo post-processing: crop the fixed camera to the env-0 robot, trim to the sequence, and build the GIF (< 8 MB) plus the matching mp4.
# usage: make_demo_gif.sh [source_mp4]    (default media/demo_blend_try1_0000.mp4, the recording behind the shipped clip)
#   crop 760x510 at (280,30): frames the origin robot at ~40-50% of the crop height, falls and walk-away stay inside.
#   trim 2.6-14.0 s: push at 3.0 s, fall detected at 3.84 s, hand-over at 10.84 s, first 3 s of the walk after it.
set -e
P=$(cd "$(dirname "$0")" && pwd); SRC=${1:-$P/media/demo_blend_try1_0000.mp4}
ffmpeg -hide_banner -loglevel error -y -ss 2.6 -to 14.0 -i "$SRC" -vf "crop=760:510:280:30" -c:v libx264 -crf 20 -pix_fmt yuv420p -an $P/media/demo_fall_getup_walk.mp4
ffmpeg -hide_banner -loglevel error -y -i $P/media/demo_fall_getup_walk.mp4 \
  -vf "fps=12,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64:stats_mode=diff[p];[s1][p]paletteuse=dither=none" -loop 0 $P/media/demo_fall_getup_walk.gif
ls -la $P/media/demo_fall_getup_walk.mp4 $P/media/demo_fall_getup_walk.gif
