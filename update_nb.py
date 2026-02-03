
import json

files = [
    'examples/legacy/fargopy-tutorial-zoom_in.ipynb',
    'examples/legacy/fargopy-tutorial-plotly.ipynb'
]

for filepath in files:
    try:
        with open(filepath, 'r') as f:
            nb = json.load(f)
        
        modified = False
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                new_source = []
                for line in cell['source']:
                    if "sim.load_field" in line and "interpolate=True" not in line:
                        parts = line.rsplit(')', 1)
                        if len(parts) == 2:
                            new_line = parts[0] + ",interpolate=True)" + parts[1]
                            new_source.append(new_line)
                            modified = True
                        else:
                            new_source.append(line)
                    else:
                        new_source.append(line)
                cell['source'] = new_source
        
        if modified:
            with open(filepath, 'w') as f:
                json.dump(nb, f, indent=1)
            print(f"Updated {filepath}")
        else:
            print(f"No changes needed for {filepath}")

    except FileNotFoundError:
        print(f"File not found: {filepath}")
