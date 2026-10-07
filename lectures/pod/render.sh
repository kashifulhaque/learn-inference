#!/usr/bin/env bash
# Renders lectures on a Runpod pod set up by setup.sh, at 1080p, as many at
# once as the pod's cores allow. Run it from the repository copy that the Mac
# rsyncs to /workspace/<course>; the course id comes from lectures/course.json.
#
#   bash lectures/pod/render.sh --all
#   bash lectures/pod/render.sh 02-memory-arithmetic 03-tokens-and-embeddings
#   bash lectures/pod/render.sh --all --force
#
# Output goes to /workspace/<course>/build/<slug>/<version>/dist, the narration
# cache to /workspace/cache/<course>/tts, and the log to
# /workspace/logs/<course>/render-<UTC time>.log (latest.log points at it).
# Copy the build directory back and publish from the Mac with
# `build.py --publish ... --from DIR`; the pod never uploads.
#
# Environment:
#   LECTURE_TTS_URL, LECTURE_TTS_TOKEN  The speech server. Set them in the
#                                       pod's environment, never in a file
#                                       in the repository.
#   LECTURE_TTS=estimate                Silent narration, for a smoke test.
#   LECTURE_JOBS                        Renders at once (default: cores / 2).
#   LECTURE_ENCODER=h264_nvenc          Use NVENC if FFmpeg has it.
set -euo pipefail

LECTURES=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(dirname "$LECTURES")
VENV=/workspace/venv
COURSE=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["course"])' "$LECTURES/course.json")
OUT=/workspace/$COURSE/build
LOGS=/workspace/logs/$COURSE
CACHE=/workspace/cache/$COURSE/tts

if [ $# -eq 0 ]; then
  sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
fi
for arg in "$@"; do
  case $arg in
    --publish|--upload|--from|--dry-run)
      echo "render.sh renders only; publish from the Mac with build.py --publish." >&2
      exit 2 ;;
  esac
done
if [ "${LECTURE_TTS:-remote}" != estimate ] && [ -z "${LECTURE_TTS_URL:-}" ]; then
  echo "LECTURE_TTS_URL is not set. Export it (and LECTURE_TTS_TOKEN) in the pod's environment," >&2
  echo "or set LECTURE_TTS=estimate for a silent test render." >&2
  exit 2
fi
if ! "$VENV/bin/python" -c 'import manim' 2>/dev/null; then
  echo "$VENV has no Manim; run bash $LECTURES/pod/setup.sh first." >&2
  exit 2
fi

mkdir -p "$OUT" "$LOGS" "$CACHE"
export LECTURE_NATIVE=1 LECTURE_TTS_CACHE="$CACHE" PATH="$VENV/bin:$PATH"
jobs=()
[ -n "${LECTURE_JOBS:-}" ] && jobs=(--jobs "$LECTURE_JOBS")
log="$LOGS/render-$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "$log" "$LOGS/latest.log"
cd "$REPO"
echo "rendering $* into $OUT; log $log"
set +e
python lectures/build.py "$@" --quality h --out "$OUT" ${jobs[@]+"${jobs[@]}"} 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
echo "exit $rc" | tee -a "$log"
exit "$rc"
