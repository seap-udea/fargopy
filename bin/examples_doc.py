#!/usr/bin/env python3
import os
import glob
import shutil

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root_dir, 'examples')
    legacy_dir = os.path.join(examples_dir, 'legacy')
    docs_examples_dir = os.path.join(root_dir, 'docs', 'examples')
    rst_file = os.path.join(root_dir, 'docs', 'examples.rst')

    # Ensure docs/examples exists
    if os.path.exists(docs_examples_dir):
        shutil.rmtree(docs_examples_dir)
    os.makedirs(docs_examples_dir)

    # Collect notebooks
    notebooks = []
    
    # helper to process directory
    def process_dir(directory):
        files = sorted(glob.glob(os.path.join(directory, "*.ipynb")))
        for f in files:
            basename = os.path.basename(f)
            dest = os.path.join(docs_examples_dir, basename)
            shutil.copy2(f, dest)
            notebooks.append(basename)
            print(f"Copied {basename} to docs/examples/")

    print(f"Scanning {examples_dir}...")
    process_dir(examples_dir)
    
    if os.path.isdir(legacy_dir):
        print(f"Scanning {legacy_dir}...")
        process_dir(legacy_dir)

    # Generate examples.rst
    print(f"Updating {rst_file}...")
    with open(rst_file, 'w') as rst:
        rst.write("\n")
        rst.write("Tutorials and Examples\n")
        rst.write("======================\n\n")
        rst.write(".. toctree::\n")
        rst.write("   :maxdepth: 2\n")
        rst.write("   :caption: Examples\n\n")
        
        for nb in notebooks:
            name = os.path.splitext(nb)[0]
            rst.write(f"   examples/{name}\n")

    print("Done.")

if __name__ == "__main__":
    main()
