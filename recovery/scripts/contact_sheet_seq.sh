#!/bin/bash
# contact_sheet_seq.sh <video.mp4> <out.png> <t1> <t2> ... : frames of an UNCUT 50 fps clip at the given times (seconds), labelled with the time computed from the source frame number.
V=$1; O=$2; shift 2
SEL=$(for t in "$@"; do printf "eq(n,%d)+" "$(python3 -c "print(round($t*50))")"; done); SEL=${SEL%+}
N=$#; COLS=4; ROWS=$(( (N + COLS - 1) / COLS ))
LABEL="t = %{eif\:n/50\:d}.%{eif\:mod(n,50)*2\:d\:2} s"
ffmpeg -v error -y -i "$V" -vf "drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='$LABEL':x=20:y=20:fontsize=44:fontcolor=white:box=1:boxcolor=black@0.6,select='$SEL',scale=640:-1,tile=${COLS}x${ROWS}" -fps_mode vfr -frames:v 1 "$O"
