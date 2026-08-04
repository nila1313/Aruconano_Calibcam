#!/usr/bin/env bash

set -euo pipefail

PROJECT="$(
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    pwd
)"

SOURCE="$PROJECT/src/native/aruco_nano_detector.cpp"
NANO_INCLUDE="$PROJECT/external/aruco_nano"
OUTPUT="$PROJECT/build/native/aruco_nano_detector"

CXX="/usr/bin/clang++"

if [ ! -x "$CXX" ]; then
    echo "ERROR: Compiler was not found: $CXX" >&2
    exit 1
fi

if ! command -v pkg-config >/dev/null 2>&1; then
    echo "ERROR: pkg-config was not found." >&2
    exit 1
fi

if ! pkg-config --exists opencv4; then
    echo "ERROR: opencv4 was not found through pkg-config." >&2
    exit 1
fi

if [ ! -f "$SOURCE" ]; then
    echo "ERROR: Detector source is missing: $SOURCE" >&2
    exit 1
fi

if [ ! -f "$NANO_INCLUDE/aruco_nano.h" ]; then
    echo "ERROR: Pinned ArUco Nano header is missing." >&2
    exit 1
fi

if [ -e "$OUTPUT" ]; then
    echo "ERROR: Build output already exists: $OUTPUT" >&2
    exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

echo "Compiler:"
"$CXX" --version | head -n 1

echo
echo "OpenCV:"
pkg-config --modversion opencv4

echo
echo "Building:"
echo "$OUTPUT"

# pkg-config intentionally supplies the Homebrew OpenCV 4.13
# include path and linked libraries.
# shellcheck disable=SC2046
"$CXX" \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -Wall \
    -Wextra \
    -Wpedantic \
    "$SOURCE" \
    -I"$NANO_INCLUDE" \
    $(pkg-config --cflags opencv4) \
    $(pkg-config --libs opencv4) \
    -o "$OUTPUT"

echo
echo "Build completed:"
ls -lh "$OUTPUT"
