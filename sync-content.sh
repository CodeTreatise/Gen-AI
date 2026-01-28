#!/bin/bash

# Sync content from AI_ML_Web_Integration_Course to ai-course-site
# This syncs the expanded lesson content to the Starlight documentation site

SOURCE_DIR="/workspace/Learning/AI_ML_Web_Integration_Course"
DEST_DIR="/workspace/Learning/ai-course-site/src/content/docs"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Syncing Course Content to Starlight Site ===${NC}"
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"
echo ""

# Function to add frontmatter to a file if missing
add_frontmatter() {
    local file="$1"
    
    # Check if file already has frontmatter
    if head -1 "$file" | grep -q "^---$"; then
        return 0
    fi
    
    # Extract title from first H1 heading or use filename
    title=$(grep -m1 "^# " "$file" | sed 's/^# //')
    if [ -z "$title" ]; then
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
}

# Sync a specific unit
sync_unit() {
    local unit_num="$1"
    local unit_dir=$(find "$SOURCE_DIR" -maxdepth 1 -type d -name "${unit_num}-*" | head -1)
    
    if [ -z "$unit_dir" ]; then
        echo "Unit $unit_num not found"
        return 1
    fi
    
    local unit_name=$(basename "$unit_dir")
    local dest_unit_dir="$DEST_DIR/$unit_name"
    
    echo -e "${GREEN}Syncing: $unit_name${NC}"
    
    # Remove old content in destination
    rm -rf "$dest_unit_dir"
    mkdir -p "$dest_unit_dir"
    
    # Copy all markdown files, preserving structure
    find "$unit_dir" -name "*.md" | while read src_file; do
        # Skip _complete-unit.md (master outline)
        if [[ "$src_file" == *"_complete-unit.md"* ]]; then
            continue
        fi
        
        # Get relative path from unit dir
        rel_path="${src_file#$unit_dir/}"
        dest_file="$dest_unit_dir/$rel_path"
        
        # Create destination directory if needed
        mkdir -p "$(dirname "$dest_file")"
        
        # Copy and add frontmatter
        cp "$src_file" "$dest_file"
        add_frontmatter "$dest_file"
    done
    
    # Count synced files
    count=$(find "$dest_unit_dir" -name "*.md" | wc -l)
    echo "  → Synced $count files"
}

# If unit number provided, sync only that unit
if [ -n "$1" ]; then
    # Pad to 2 digits
    unit_num=$(printf "%02d" "$1")
    sync_unit "$unit_num"
else
    # Sync all units
    for i in $(seq 0 25); do
        unit_num=$(printf "%02d" "$i")
        sync_unit "$unit_num"
    done
fi

echo ""
echo -e "${GREEN}=== Sync Complete ===${NC}"
echo "Run 'npm run build' in ai-course-site to rebuild the site"
