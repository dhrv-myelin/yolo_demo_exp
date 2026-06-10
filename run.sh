#!/usr/bin/env bash

set -e

echo "================================="
echo "Processing Vid2"
echo "================================="

uv run test.py \
  --video ./data/D03_deliside_cashbelowphone_001_720.mp4 \
  --camera-name D03 \
  --roi "566,408;745,410;740,175;558,165;558,399"

echo
echo "================================="
echo "Processing Vid3"
echo "================================="

uv run test.py \
  --video ./data/D03_deliside_cashinpocket_002_720.mp4 \
  --camera-name D03 \
  --roi "734,396;724,207;476,199;468,257;621,272;625,399;735,396"

echo
echo "================================="
echo "Processing Vid4"
echo "================================="

uv run test.py \
  --video ./data/D02_glassside_cashbelowphone_002_720.mp4 \
  --camera-name D02 \
  --roi "648,481;858,404;721,223;546,292;647,475"

echo
echo "All videos processed."
