#!/bin/bash

# Add frontmatter to markdown files that don't have it
find /workspace/Learning/ai-course-site/src/content/docs -name "*.md" | while read file; do
  # Check if file already has frontmatter
  if ! head -1 "$file" | grep -q "^---$"; then
    # Extract title from first H1 heading or use filename
    title=$(grep -m1 "^# " "$file" | sed 's/^# //')
    if [ -z "$title" ]; then
      # Use filename as title
      title=$(basename "$file" .md | sed 's/-/ /g' | sed 's/^[0-9]* //')
    fi
    
    # Create temp file with frontmatter
    {
      echo "---"
      echo "title: \"$title\""
      echo "---"
      echo ""
      cat "$file"
    } > "${file}.tmp"
    mv "${file}.tmp" "$file"
    echo "Added frontmatter to: $file"
  fi
done
