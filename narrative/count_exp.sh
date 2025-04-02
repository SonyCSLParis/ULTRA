#!/bin/bash

# Check if a path argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Use the provided path argument
path="$1"

# Count text files in the specified path
find "$path" -type f -name "*.txt" | wc -l