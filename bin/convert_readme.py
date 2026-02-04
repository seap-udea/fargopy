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
    """Replaces local image links with GitHub URLs."""
    print(f"Fixing image links in {MARKDOWN_FILE}...")

    with open(MARKDOWN_FILE, "r") as f:
        content = f.read()

    def replacer(match):
        alt_text = match.group(1)
        path = match.group(2)

        # Skip absolute URLs
        if path.startswith("http") or path.startswith("https"):
            return match.group(0)

        # If path is in README_files, point to gallery
        if path.startswith(f"{NB_OUTPUT_DIR}/"):
            filename = os.path.basename(path)
            new_path = f"{GALLERY_DIR}/{filename}"
        elif path.startswith(f"{GALLERY_DIR}/"):
            new_path = path
        else:
            # Assume other relative paths are also in gallery if they look like generated images
            # or keep them as is if they are not standard outputs.
            # For this task, we assume the user wants typical outputs in gallery.
            # But let's be safe: only touch recognized directories or explicit requests.
            new_path = path

        full_url = f"{GITHUB_REPO_URL}/{new_path}"
        print(f"  Replacing '{path}' -> '{full_url}'")
        return f"![{alt_text}]({full_url})"

    new_content = re.sub(r"!\[(.*?)\]\((.*?)\)", replacer, content)

    def img_tag_replacer(match):
        full_tag = match.group(0)
        src = match.group(2)
        if src.startswith("http") or src.startswith("https"):
            return full_tag

        # Same logic for HTML tags
        if src.startswith(f"{NB_OUTPUT_DIR}/"):
            filename = os.path.basename(src)
            new_path = f"{GALLERY_DIR}/{filename}"
        elif src.startswith(f"{GALLERY_DIR}/"):
            new_path = src
        else:
            new_path = src

        full_url = f"{GITHUB_REPO_URL}/{new_path}"
        print(f"  Replacing HTML src '{src}' -> '{full_url}'")
        return full_tag.replace(f'src="{src}"', f'src="{full_url}"')

    new_content = re.sub(
        r'<img\s+(.*?)src=["\'](.*?)["\']', img_tag_replacer, new_content
    )

    with open(MARKDOWN_FILE, "w") as f:
        f.write(new_content)


def main():
    if not os.path.exists(NOTEBOOK_FILE):
        print(f"Error: {NOTEBOOK_FILE} not found.")
        sys.exit(1)

    convert_notebook()
    move_images_to_gallery()
    fix_image_links()
    print("Done.")


if __name__ == "__main__":
    main()
