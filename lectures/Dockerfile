# syntax=docker/dockerfile:1.7
#
# The lecture render image: Manim Community 0.21, TeX Live, FFmpeg, WebP
# tools, and the site's fonts. Narration comes from a remote speech server
# (or LECTURE_TTS=estimate), so the image holds no speech model. Build it
# from the repository root:
#
#   docker build -t learn-inference-lectures:dev lectures
#
# The base image publishes a native linux/arm64 variant, so renders on Apple
# silicon never run under emulation. Every download is pinned by checksum,
# so the image is reproducible. lectures/pod/setup.sh installs the same
# pieces on a Runpod pod; keep the two in step.

FROM manimcommunity/manim:v0.21.0

USER root

RUN apt-get update \
 && apt-get install -y --no-install-recommends unzip fontconfig ffmpeg webp \
 && rm -rf /var/lib/apt/lists/*

# Fonts ----------------------------------------------------------------------
# learn-inference sets prose in Source Serif 4, interface text in IBM Plex
# Sans, and code in IBM Plex Mono (frontend/src/index.css). Inter and
# JetBrains Mono are here for courses whose sites use them. Static TTF cuts,
# because Pango picks weights from named instances more reliably than from
# variable axes.

ADD --checksum=sha256:549fdb8f9a682bd06944298621404969f6de77c2e422ff3b8244a1dcd6a0c425 \
    https://github.com/adobe-fonts/source-serif/releases/download/4.005R/source-serif-4.005_Desktop.zip /tmp/serif.zip
ADD --checksum=sha256:fb365d910566e6d199cc2c15579a7dd9a267128e18431a394ed81f1970c69200 \
    https://github.com/IBM/plex/releases/download/%40ibm/plex-sans%401.1.0/ibm-plex-sans.zip /tmp/plex-sans.zip
ADD --checksum=sha256:6d23f01257663d8cc49a0d64c22ced630b79e0e2a0ac08a0da86e9a38bbc481c \
    https://github.com/IBM/plex/releases/download/%40ibm/plex-mono%402.5.0/ibm-plex-mono.zip /tmp/plex-mono.zip
ADD --checksum=sha256:9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e \
    https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip /tmp/inter.zip
ADD --checksum=sha256:6f6376c6ed2960ea8a963cd7387ec9d76e3f629125bc33d1fdcd7eb7012f7bbf \
    https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip /tmp/jbmono.zip

COPY fonts.sh /tmp/fonts.sh
RUN bash /tmp/fonts.sh /tmp /usr/share/fonts/truetype/lectures && rm -f /tmp/*.zip /tmp/fonts.sh

ENV PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2

# The repository is mounted at /work; the kit is importable from there.
ENV PYTHONPATH=/work/lectures
WORKDIR /work

USER manimuser
