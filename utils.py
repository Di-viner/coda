import json
import os

def read_json_file(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)

    elif filepath.endswith('.jsonl'):
        data = []
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")


def save_jsonl(data, path):
    if data is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w', encoding='utf-8') as w:
        for unit in data:
            line = json.dumps(unit)
            w.write(line + "\n")