#!/bin/bash
# contact_sheet.sh <video.mp4> <out.png> : frames at t = 0.5, 1, 2, 4, 6, 9 s from an UNCUT 10 s clip, labelled with the time
F="fps=50,select='eq(n,25)+eq(n,50)+eq(n,100)+eq(n,200)+eq(n,300)+eq(n,450)',scale=640:-1,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='t = %{pts\:flt\:2} s':x=10:y=10:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6,tile=3x2"
ffmpeg -v error -y -i "$1" -vf "$F" -fps_mode vfr -frames:v 1 "$2"
