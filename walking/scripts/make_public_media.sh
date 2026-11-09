#!/bin/bash
# Build the small public media in media/ from the raw MP4s in media-raw/ (not committed). Needs ffmpeg.
# media-raw/progression/prog_0000..0008.mp4 map to iterations 0 50 100 200 350 500 700 1000 1499.
# media-raw/newton_best/newton_best_0002.mp4 is the three-quarter single-robot clip (final policy).
set -e
cd "$(dirname "$0")/.."
FONT=${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}
IT=(0 50 100 200 350 500 700 1000 1499); T=$(mktemp -d); mkdir -p media
lab() { # in out text
  ffmpeg -v error -y -i "$1" -vf "drawtext=fontfile=$FONT:text='$3':x=24:y=24:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=10" -an -c:v libx264 -pix_fmt yuv420p "$2"; }
for i in 0 3 4 8; do lab media-raw/progression/prog_000$i.mp4 $T/it${IT[$i]}.mp4 "Newton PPO iteration ${IT[$i]}"; done
# hero: iteration 0 vs 1499, side by side, phase caption underneath
PH="drawtext=fontfile=$FONT:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8:x=(w-text_w)/2:y=h-70"
ffmpeg -v error -y -i $T/it0.mp4 -i $T/it1499.mp4 -filter_complex "[0][1]hstack=2,\
${PH}:text='stand  (0-2.5 s)':enable='lt(t,2.5)',\
${PH}:text='walk 0.8 m/s  (2.5-5 s)':enable='between(t,2.5,5)',\
${PH}:text='turn 0.8 rad/s  (5-7.5 s)':enable='between(t,5,7.5)',\
${PH}:text='stop  (7.5-10 s)':enable='gte(t,7.5)',\
fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer" media/hero_before_after.gif
# 2x2 frame montage (t = 6 s, turn phase) of iterations 0, 200, 350, 1499
for n in 0 200 350 1499; do ffmpeg -v error -y -ss 6 -i $T/it$n.mp4 -frames:v 1 -vf scale=640:-1 $T/f$n.png; done
ffmpeg -v error -y -i $T/f0.png -i $T/f200.png -i $T/f350.png -i $T/f1499.png -filter_complex "[0][1]hstack[t];[2][3]hstack[b];[t][b]vstack" media/progression_frames.png
# single-robot clips (400 px wide GIF, 8 fps)
for L in newton_best; do
  ffmpeg -v error -y -i media-raw/$L/${L}_0002.mp4 -vf "fps=8,scale=400:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4" -loop 0 media/${L}_single_robot.gif
done
ls -l media | awk '{print $5, $9}'
