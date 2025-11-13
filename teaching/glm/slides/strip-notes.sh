#!/bin/bash

# Script to strip speaker notes from .draft.qmd files
# Creates .qmd versions (without .draft) in the same folder

echo "Looking for .draft.qmd files..."

# Counter for processed files
COUNT=0

# Find all .draft.qmd files in current directory
for file in *.draft.qmd; do
    # Check if any .draft.qmd files exist
    if [ ! -f "$file" ]; then
        echo "No .draft.qmd files found in current directory"
        exit 0
    fi
    
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
    COUNT=$((COUNT + 1))
    echo ""
done

echo "========================================="
echo "✓ Processed $COUNT file(s)"
echo "========================================="