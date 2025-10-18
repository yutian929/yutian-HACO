#!/bin/bash

TARGET_DIR="."
FILE_URL="https://huggingface.co/datasets/dqj5182/haco-data/resolve/main/demo/data/base_data.tar.gz"
ARCHIVE_NAME="$TARGET_DIR/base_data.tar.gz"

mkdir -p "$TARGET_DIR"

echo "Downloading base_data.tar.gz..."
wget -c "$FILE_URL" -O "$ARCHIVE_NAME"

echo "Decompressing into $TARGET_DIR..."
tar -xvzf "$ARCHIVE_NAME" -C "$TARGET_DIR"

echo "Removing archive..."
rm "$ARCHIVE_NAME"

echo "Done. Extracted to $TARGET_DIR"

TARGET_DIR="base_data/release_checkpoint"
mkdir -p "$TARGET_DIR"

# Base URL of the Hugging Face dataset repo (using 'resolve/main')
BASE_URL="https://huggingface.co/datasets/dqj5182/haco-checkpoints/resolve/main"

# List of files to download (add more as needed)
FILES=(
  "haco_final_hamer_checkpoint.ckpt"
)

# Download each file directly to the target directory
for file in "${FILES[@]}"; do
  echo "Downloading $file to $TARGET_DIR..."
  wget -c "$BASE_URL/$file" -O "$TARGET_DIR/$file"
done

echo "All files downloaded to $TARGET_DIR"