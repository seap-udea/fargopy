#!/usr/bin/env python3
import os
import glob
import shutil


def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    examples_dir = os.path.join(root_dir, "examples")
    docs_examples_dir = os.path.join(root_dir, "docs", "examples")
    rst_file = os.path.join(root_dir, "docs", "examples.rst")

    # Ensure docs/examples exists
    if os.path.exists(docs_examples_dir):
        shutil.rmtree(docs_examples_dir)
    os.makedirs(docs_examples_dir)

    # Read index.rst to get priority order
    index_file = os.path.join(examples_dir, "index.rst")
    priority_order = []
    if os.path.exists(index_file):
        with open(index_file, "r") as f:
            priority_order = [line.strip() for line in f if line.strip()]

    # Collect all notebooks
    all_notebooks = glob.glob(os.path.join(examples_dir, "*.ipynb"))
    notebook_map = {os.path.basename(f): f for f in all_notebooks}

    # Determine final order
    ordered_files = []

    # 1. Add files from index.rst if they exist
    for name in priority_order:
        if name in notebook_map:
            ordered_files.append(notebook_map[name])
            del notebook_map[name]
        else:
            print(f"Warning: {name} listed in index.rst but not found in examples/")

    # 2. Add remaining files alphabetically
    for name in sorted(notebook_map.keys()):
        ordered_files.append(notebook_map[name])

    # Copy files and build list for RST
    notebooks = []
    for f in ordered_files:
        basename = os.path.basename(f)
        dest = os.path.join(docs_examples_dir, basename)
        shutil.copy2(f, dest)
        notebooks.append(basename)
        print(f"Copied {basename} to docs/examples/")

    # Generate examples.rst
    print(f"Updating {rst_file}...")
    with open(rst_file, "w") as rst:
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
