import os
import json


def update_links(directory):
    for root, dirs, files in os.walk(directory):
        # Exclude hidden directories, _build, and checkpoints
        dirs[:] = [
            d
            for d in dirs
            if not d.startswith(".") and not d.startswith("_") and d != "checkpoints"
        ]

        for file in files:
            if file.endswith(".ipynb"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Perform replacements
                    # 1. Open in Colab links: github/seap-udea/fargopy/blob/refactor/ -> github/seap-udea/fargopy/blob/main/
                    # 2. Raw content links: raw.githubusercontent.com/seap-udea/fargopy/refactor/ -> raw.githubusercontent.com/seap-udea/fargopy/main/

                    new_content = content.replace(
                        "github/seap-udea/fargopy/blob/refactor/",
                        "github/seap-udea/fargopy/blob/main/",
                    )
                    new_content = new_content.replace(
                        "raw.githubusercontent.com/seap-udea/fargopy/refactor/",
                        "raw.githubusercontent.com/seap-udea/fargopy/main/",
                    )
                    new_content = new_content.replace(
                        "github.com/seap-udea/fargopy/blob/refactor/",
                        "github.com/seap-udea/fargopy/blob/main/",
                    )  # Catch direct github links too

                    if new_content != content:
                        print(f"Updating links in {filepath}")
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(new_content)
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")


if __name__ == "__main__":
    update_links(".")
