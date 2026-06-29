#!/bin/bash

# Ensure images directory exists
mkdir -p ./data/images/

echo "Starting extraction of all 12 image archives (This may take a while)..."
for f in ./data/archives/*.tar.gz; do
    echo "Extracting $f..."
    # -k flag prevents overwriting existing images from previous runs to save time
    tar -kxzf "$f" -C ./data/images/ 2>/dev/null || true
done
echo "Extraction complete!"

echo "Starting training on the full dataset for 10 epochs..."
python3 train.py --epochs 10
