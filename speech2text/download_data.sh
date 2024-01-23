#!/bin/bash

# Directory to check or create
DIR="data"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    # If it doesn't exist, create the directory
    mkdir "$DIR"
fi

# Base URL
BASE_URL="https://huggingface.co/datasets/javyduck/MMCBench/resolve/main/speech2text"

# Arrays of options
types=("heavy" "light")
kinds=("hard_1k" "random_1k")

# Loop over types and kinds to download each file
for type in "${types[@]}"; do
    for kind in "${kinds[@]}"; do
        FILE_NAME="${type}_corrupted_${kind}.parquet"
        URL="${BASE_URL}/${FILE_NAME}"

        # Use wget to download the file into the 'data' directory
        wget -O "${DIR}/${FILE_NAME}" "$URL"
    done
done
