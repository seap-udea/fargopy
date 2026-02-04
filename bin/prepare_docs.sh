#!/bin/bash

# Exit on error
set -e

# Get the directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"

echo "Running examples_doc.py..."
python3 "$DIR/examples_doc.py"

echo "Generating README.md from README.ipynb..."
make readme 

echo "Sanitizing README.md..."
# 1. Replace version number "X.X.X" with "X.Y.Z" in "Running FARGOpy version ..." context
sed -i '' 's/Running FARGOpy version [0-9]*\.[0-9]*\.[0-9]*/Running FARGOpy version X.Y.Z/g' "$ROOT_DIR/README.md"

# 2. Replace local paths ending in /fargo3d/
perl -i -pe 's|[\w/.~-]+/fargo3d/|/local_directory/fargo3d/|g' "$ROOT_DIR/README.md"

echo "Removing footer from README.md..."
# Remove lines from "Powered by fargopy" to the end of the file
# We use sed to delete from the line matching "Powered by fargopy" to the end ($)
# Note: macOS sed requires empty string for -i argument
sed -i '' '/Powered by fargopy/,$d' "$ROOT_DIR/README.md"
# Remove the usage of --- that was before the footer if it exists
sed -i '' '/^---$/d' "$ROOT_DIR/README.md"

echo "Documentation preparation complete."
