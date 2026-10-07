#!/usr/bin/env bash
# Installs the lecture toolchain on a Runpod pod (Ubuntu, running as root), so
# that lectures/build.py can render with LECTURE_NATIVE=1. It installs the
# same pieces as lectures/Dockerfile: TeX Live, FFmpeg, WebP tools, the fonts
# (through lectures/fonts.sh, checked by SHA-256), and Manim Community 0.21 in
# a virtual environment on the network volume at /workspace/venv.
#
# Safe to re-run. A stamp on the container disk records a finished setup of
# this exact script; a pod reset wipes the container disk (and the apt
# packages with it), so the next run reinstalls them and reuses the venv.
#
# Usage: bash lectures/pod/setup.sh
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

LECTURES=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV=/workspace/venv
STAMP=/root/.lecture-setup-$(cat "$LECTURES/pod/setup.sh" "$LECTURES/fonts.sh" | sha256sum | cut -c1-12)
if [ -f "$STAMP" ]; then
  echo "setup already done ($STAMP)"
  exit 0
fi

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl unzip fontconfig ffmpeg webp rsync \
  build-essential pkg-config libcairo2-dev libpango1.0-dev python3-dev python3-venv \
  texlive texlive-xetex texlive-latex-extra texlive-fonts-extra texlive-science texlive-pictures dvisvgm

# Fonts: the same releases and checksums as the Dockerfile.
fonts=$(mktemp -d)
fetch() {  # fetch NAME SHA256 URL
  curl -fsSL --retry 3 -o "$fonts/$1" "$3"
  echo "$2  $fonts/$1" | sha256sum -c -
}
fetch serif.zip 549fdb8f9a682bd06944298621404969f6de77c2e422ff3b8244a1dcd6a0c425 \
  https://github.com/adobe-fonts/source-serif/releases/download/4.005R/source-serif-4.005_Desktop.zip
fetch plex-sans.zip fb365d910566e6d199cc2c15579a7dd9a267128e18431a394ed81f1970c69200 \
  https://github.com/IBM/plex/releases/download/%40ibm/plex-sans%401.1.0/ibm-plex-sans.zip
fetch plex-mono.zip 6d23f01257663d8cc49a0d64c22ced630b79e0e2a0ac08a0da86e9a38bbc481c \
  https://github.com/IBM/plex/releases/download/%40ibm/plex-mono%402.5.0/ibm-plex-mono.zip
fetch inter.zip 9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e \
  https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip
fetch jbmono.zip 6f6376c6ed2960ea8a963cd7387ec9d76e3f629125bc33d1fdcd7eb7012f7bbf \
  https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip
bash "$LECTURES/fonts.sh" "$fonts" /usr/share/fonts/truetype/lectures
rm -rf "$fonts"

# Manim 0.21 needs Python 3.11 or later. Ubuntu 24.04 ships 3.12; on 22.04,
# install 3.11 from the deadsnakes archive.
py_ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; }
PY=python3
if ! py_ok python3; then
  apt-get install -y --no-install-recommends software-properties-common gpg-agent
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update
  apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev
  PY=python3.11
fi

# The venv lives on the volume, so it survives a pod reset. Rebuild it if it
# was made with an older Python or its interpreter went with the old disk.
if [ -e "$VENV" ] && ! py_ok "$VENV/bin/python"; then
  rm -rf "$VENV"
fi
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet manim==0.21.0
"$VENV/bin/python" -c "import manim, numpy, PIL, pygments; print('manim', manim.__version__, 'ok')"
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc && echo "ffmpeg has h264_nvenc" \
  || echo "ffmpeg has no h264_nvenc; renders encode with libx264"

rm -f /root/.lecture-setup-*
touch "$STAMP"
echo "setup done"
