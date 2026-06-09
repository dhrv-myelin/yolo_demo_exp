#!/usr/bin/env bash

set -e

echo "================================="
echo "Processing Vid2"
echo "================================="

uv run test.py \
  --video ./data/Vid2.mp4 \
  --camera-name Vid2 \
  --roi "566,408;745,410;740,175;558,165;558,399"

echo
echo "================================="
echo "Processing Vid3"
echo "================================="

uv run test.py \
  --video ./data/Vid3.mp4 \
  --camera-name Vid3 \
  --roi "734,396;724,207;476,199;468,257;621,272;625,399;735,396"

echo
echo "================================="
echo "Processing Vid4"
echo "================================="

uv run test.py \
  --video ./data/Vid4.mp4 \
  --camera-name Vid4 \
  --roi "648,481;858,404;721,223;546,292;647,475"

echo
echo "All videos processed."
