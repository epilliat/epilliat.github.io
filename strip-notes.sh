#!/bin/bash

# Script to strip speaker notes from .draft.qmd files
# Recursively processes all subfolders from root

echo "========================================="
echo "Searching for .draft.qmd files..."
echo "========================================="
echo ""

# Counter for processed files
TOTAL_COUNT=0

# Find all .draft.qmd files recursively
while IFS= read -r -d '' file; do
    # Get the directory and filename
    dir=$(dirname "$file")
    filename=$(basename "$file")
    
    # Create output filename by removing .draft
    output="${file%.draft.qmd}.qmd"
    
    echo "Processing: $file"
    
    # Check if file contains speaker notes
    if grep -q "^::: {\.notes}" "$file"; then
        NOTES_COUNT=$(grep -c "^::: {\.notes}" "$file")
        echo "  Found $NOTES_COUNT note section(s)"
    else
        echo "  No speaker notes found"
    fi
    
    # Remove speaker notes and create output file
    sed '/^::: {\.notes}/,/^:::$/d' "$file" > "$output"
    
    echo "  ✓ Created: $output"
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    echo ""
    
done < <(find . -type f -name "*.draft.qmd" -print0)

echo "========================================="
if [ $TOTAL_COUNT -eq 0 ]; then
    echo "No .draft.qmd files found"
else
    echo "✓ Processed $TOTAL_COUNT file(s) total"
fi
echo "========================================="