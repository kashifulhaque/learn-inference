#!/usr/bin/env bash
# Unpacks the lecture fonts into a font directory and refreshes fontconfig.
# Shared by the Dockerfile and pod/setup.sh, so both install the same cuts.
#
# Usage: fonts.sh SRC_DIR DEST_DIR
#   SRC_DIR holds serif.zip, plex-sans.zip, plex-mono.zip, inter.zip, and
#   jbmono.zip, each already checked against its SHA-256 (see FONT_SUMS).
set -euo pipefail
src=$1
dest=$2
mkdir -p "$dest"
unzip -j -o "$src/serif.zip" \
  'source-serif-4.005_Desktop/TTF/SourceSerif4-Regular.ttf' \
  'source-serif-4.005_Desktop/TTF/SourceSerif4-It.ttf' \
  'source-serif-4.005_Desktop/TTF/SourceSerif4-Semibold.ttf' \
  'source-serif-4.005_Desktop/TTF/SourceSerif4-SemiboldIt.ttf' \
  'source-serif-4.005_Desktop/TTF/SourceSerif4-Bold.ttf' \
  'source-serif-4.005_Desktop/TTF/SourceSerif4Display-Regular.ttf' \
  'source-serif-4.005_Desktop/TTF/SourceSerif4Display-It.ttf' \
  'source-serif-4.005_Desktop/TTF/SourceSerif4Display-Semibold.ttf' -d "$dest"
unzip -j -o "$src/plex-sans.zip" \
  'ibm-plex-sans/fonts/complete/ttf/IBMPlexSans-Regular.ttf' \
  'ibm-plex-sans/fonts/complete/ttf/IBMPlexSans-Italic.ttf' \
  'ibm-plex-sans/fonts/complete/ttf/IBMPlexSans-Medium.ttf' \
  'ibm-plex-sans/fonts/complete/ttf/IBMPlexSans-SemiBold.ttf' \
  'ibm-plex-sans/fonts/complete/ttf/IBMPlexSans-Bold.ttf' -d "$dest"
unzip -j -o "$src/plex-mono.zip" \
  'ibm-plex-mono/fonts/complete/ttf/IBMPlexMono-Regular.ttf' \
  'ibm-plex-mono/fonts/complete/ttf/IBMPlexMono-Italic.ttf' \
  'ibm-plex-mono/fonts/complete/ttf/IBMPlexMono-Medium.ttf' \
  'ibm-plex-mono/fonts/complete/ttf/IBMPlexMono-SemiBold.ttf' -d "$dest"
unzip -j -o "$src/inter.zip" 'extras/ttf/Inter-Regular.ttf' 'extras/ttf/Inter-Italic.ttf' \
  'extras/ttf/Inter-Medium.ttf' 'extras/ttf/Inter-SemiBold.ttf' 'extras/ttf/Inter-Bold.ttf' -d "$dest"
unzip -j -o "$src/jbmono.zip" 'fonts/ttf/JetBrainsMono-Regular.ttf' 'fonts/ttf/JetBrainsMono-Medium.ttf' \
  'fonts/ttf/JetBrainsMono-Bold.ttf' -d "$dest"
fc-cache -f
installed=$(fc-list : family)
for family in 'Source Serif 4' 'IBM Plex Sans' 'IBM Plex Mono' 'Inter' 'JetBrains Mono'; do
  grep -qF "$family" <<<"$installed" || { echo "font family missing after install: $family" >&2; exit 1; }
done
