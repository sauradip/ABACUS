import json, os
root = "/data/amondal/UniCount/UniCount_data"
ann_path = os.path.join(root, "annotations.json")
with open(ann_path) as f:
    data = json.load(f)
print("Top-level keys:", list(data.keys()))
if data.get("images"):
    img = data["images"][0]
    print("\nFirst image:", img)
if data.get("annotations"):
    ann = data["annotations"][0]
    print("\nFirst annotation:", ann)
if data.get("categories"):
    cat = data["categories"][0]
    print("\nFirst category:", cat)
