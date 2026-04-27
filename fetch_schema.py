import requests
import json

url = "UPLOAD YOUR TARGET ADDRESS HERE"

headers = {
    "Authorization": "Bearer supersecretkey"
}

response = requests.get(url, headers=headers)

data = response.json()
with open("cube_meta.json", "w") as f:
    json.dump(data, f, indent=4)

print("✅ JSON saved successfully!")

import json

with open("cube_meta.json") as f:
    meta = json.load(f)

for cube in meta["cubes"]:
    print(f"\n📦 Cube: {cube['name']}")
    
    print("  Measures:")
    for m in cube["measures"]:
        print(f"    - {m['name']} ({m['type']})")
    
    print("  Dimensions:")
    for d in cube["dimensions"]:
        print(f"    - {d['name']} ({d['type']})")
