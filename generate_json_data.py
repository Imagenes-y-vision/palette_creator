import pandas as pd
import json

df = pd.read_csv("beatriz_gonzalez.csv")
data = {}
for i, row in df.iterrows():
    id_ = i+1
    palettes_by_method = {}
    for column_name in row.index:
        if "palette" in column_name:
            method = column_name.split("_")[0]
            palettes_by_method[method] = json.loads(row[column_name])
    image_data = {"id": id_, "title": f"Image {id_}", "filename": f"{row['filename']}", "description": f"Description {id_}",
                  "palettes_by_method": palettes_by_method }
    data[id_] = image_data

json_data = json.dumps(data)

with open("palette_app/backend/data.json", "w") as f:
    f.write(json_data)
    
