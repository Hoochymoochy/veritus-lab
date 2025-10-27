# webscraper/brazil/utils/save.py
import json
import os

import json

def save_json(data, filename, pretty=False):
    with open(filename, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            json.dump(data, f, ensure_ascii=False)
    print(f"💾 Saved {filename}")
