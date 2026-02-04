#!/usr/bin/env python3
"""
Script to convert README.ipynb to README.md and sanitize image links for GitHub.
"""

import os
import re
import subprocess
import sys

import shutil

# Configuration
NOTEBOOK_FILE = "README.ipynb"
MARKDOWN_FILE = "README.md"
GITHUB_REPO_URL = "https://raw.githubusercontent.com/seap-udea/fargopy/refactor"
GALLERY_DIR = "gallery"
NB_OUTPUT_DIR = "README_files"


def convert_notebook():
    """Converts the notebook to markdown using jupyter nbconvert."""
    print(f"Converting {NOTEBOOK_FILE} to {MARKDOWN_FILE}...")
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "markdown",
                "--output",
                MARKDOWN_FILE,
                NOTEBOOK_FILE,
            ]
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


def move_images_to_gallery():
    """Moves images from nbconvert output directory to gallery."""
    if not os.path.exists(NB_OUTPUT_DIR):
        return

    print(f"Moving images from {NB_OUTPUT_DIR} to {GALLERY_DIR}...")
    if not os.path.exists(GALLERY_DIR):
        os.makedirs(GALLERY_DIR)

    for filename in os.listdir(NB_OUTPUT_DIR):
        src = os.path.join(NB_OUTPUT_DIR, filename)
        dst = os.path.join(GALLERY_DIR, filename)
        if os.path.isfile(src):
            shutil.move(src, dst)
            print(f"  Moved {filename} to {GALLERY_DIR}/")

    # Remove empty output dir
    try:
        os.rmdir(NB_OUTPUT_DIR)
    except OSError:
        pass


def fix_image_links():
    """Replaces local image links with GitHub URLs using HTML tags and savefig filenames."""
    print(f"Fixing image links in {MARKDOWN_FILE}...")

    with open(MARKDOWN_FILE, "r") as f:
        lines = f.readlines()

    last_savefig = None
    new_lines = []

    # Regex to capture savefig.
    # Matches: plt.savefig('path') or fig.savefig("path")
    # Group 2 is the path/filename.
    savefig_pattern = re.compile(r"(plt|fig)\.savefig\(['\"](.*?)['\"]\)")

    # Pattern to match Markdown images: ![alt](path)
    md_img_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

    # Pattern to match HTML images: <img ... src="path" ...>
    # We want to capture the whole tag to replace it, and extract src.
    html_img_pattern = re.compile(r'<img\s+.*?src=["\'](.*?)["\'].*?>')

    for line in lines:
        # 1. Search for savefig calls in valid code/text lines
        # We assume they appear in code blocks or text before the image
        sf_match = savefig_pattern.search(line)
        if sf_match:
            raw_path = sf_match.group(2)
            last_savefig = os.path.basename(raw_path)
            # print(f"Found savefig: {last_savefig}") # Debug

        # 2. Check for Markdown image
        md_match = md_img_pattern.search(line)
        if md_match:
            alt_text = md_match.group(1)
            path = md_match.group(2)

            # Skip absolute URLs
            if path.startswith("http") or path.startswith("https"):
                new_lines.append(line)
                continue

            # Determine filename to use
            if last_savefig:
                filename = last_savefig
                # Reset after using? Usually yes, to avoid applying to unrelated images.
                last_savefig = None
            else:
                filename = os.path.basename(path)

            # Determine new path logic
            if "README" in filename and filename.endswith(".png"):  # Generated
                new_path = f"{GALLERY_DIR}/{filename}"
            elif path.startswith(f"{NB_OUTPUT_DIR}/"):
                new_path = f"{GALLERY_DIR}/{filename}"
            elif path.startswith(f"{GALLERY_DIR}/"):
                new_path = f"{GALLERY_DIR}/{filename}"
            else:
                new_path = f"{GALLERY_DIR}/{filename}"

            full_url = f"{GITHUB_REPO_URL}/{new_path}"
            print(f"  Replacing '{path}' -> '{full_url}'")
            new_line = f'<img src="{full_url}" alt="{alt_text}">'

            new_lines.append(new_line + "\n")
            continue

        # 3. Check for HTML image
        # Note: Previous steps might have already converted md to html or simple html tags exists.
        html_match = html_img_pattern.search(line)
        if html_match:
            src = html_match.group(1)
            if src.startswith("http") or src.startswith("https"):
                new_lines.append(line)
                continue

            if last_savefig:
                filename = last_savefig
                last_savefig = None
            else:
                filename = os.path.basename(src)

            new_path = f"{GALLERY_DIR}/{filename}"
            full_url = f"{GITHUB_REPO_URL}/{new_path}"
            print(f"  Replacing HTML src '{src}' -> '{full_url}'")

            # Simple replace of the match in the line:
            new_line = line.replace(src, full_url)
            new_lines.append(new_line)
            continue

        new_lines.append(line)

    with open(MARKDOWN_FILE, "w") as f:
        f.writelines(new_lines)


def remove_extra_spaces():
    """Removes extra spaces (newlines) from the markdown file."""
    print(f"Cleaning extra spaces in {MARKDOWN_FILE}...")
    with open(MARKDOWN_FILE, "r") as f:
        content = f.read()

    # 1. Compact indented blank lines (likely in code/output blocks).
    prev_content = None
    while content != prev_content:
        prev_content = content
        # Remove lines that are just indentation (4+ spaces) inside the text
        content = re.sub(r"\n[ \t]{4,}\n", "\n", content)

    # 2. Reduce multiple blank lines (3+ newlines) to a single blank line (2 newlines) in normal text
    content = re.sub(r"\n([ \t]*\n){2,}", "\n\n", content)

    with open(MARKDOWN_FILE, "w") as f:
        f.write(content)


def ensure_list_spacing():
    """Ensures there is a blank line before a list if it follows a paragraph ending in colon."""
    print(f"Fixing list spacing in {MARKDOWN_FILE}...")
    with open(MARKDOWN_FILE, "r") as f:
        content = f.read()

    # Regex to find a line ending in ':' followed immediately by a list item.
    # We want to insert a newline in between.
    # Group 1: The line ending in : (and possibly trailing whitespace)
    # Group 2: The newline and the list item start (\n - ...)
    # We replace with \1\n\2 to add an extra newline.

    # Matches: "Text:  \n- Item" -> "Text:  \n\n- Item"
    # Matches: "Text:\n- Item" -> "Text:\n\n- Item"
    # Note: we use non-greedy matching on the first part if needed, but here structure is simple.
    # We match strictly a colon, optional spaces, newline, optional spaces, hyphen/star.
    new_content = re.sub(r"(:[ \t]*\n)([ \t]*[-*])", r"\1\n\2", content)

    with open(MARKDOWN_FILE, "w") as f:
        f.write(new_content)


def main():
    if not os.path.exists(NOTEBOOK_FILE):
        print(f"Error: {NOTEBOOK_FILE} not found.")
        sys.exit(1)

    convert_notebook()
    move_images_to_gallery()
    fix_image_links()
    remove_extra_spaces()
    ensure_list_spacing()
    print("Done.")


if __name__ == "__main__":
    main()
