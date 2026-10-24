#!/bin/bash
# usage: run.sh <outdir> <adapter> <scenes> [laptop]; proxy on, play pressed 4 s after the first picture
W=${W:-$REPO}
PY=python
OUT=$1; AD=${2:-default}; SC=${3:-cube}
mkdir -p $OUT
cd $W
for s in ${SC//,/ }; do
  if [ -n "$LAPTOP" ]; then PRE="taskset -c 0-3 env QT_SCALE_FACTOR=2"; else PRE="env"; fi
  PYTHONPATH=$W DISPLAY=:171 QT_QPA_PLATFORM=xcb timeout 600 flock -w 1800 /tmp/nb-gpu.lock \
    timeout 400 $PRE $PY tools/benchmark_3d_playback.py --scene $s --size 1920x1080 --seconds ${SECS:-4} --first-seconds 3 --play-at first --play-delay ${PD:-4} --adapter $AD --settle-wait 5 --proxy on --out $OUT/$s-${RUN:-1}.json ${EXTRA} > $OUT/$s-${RUN:-1}.log 2>&1
  echo "$s exit $?" >> $OUT/done.txt
done
