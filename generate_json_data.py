import pandas as pd
import json

df = pd.read_csv("beatriz_gonzalez.csv")
data = {}
for i, row in df.iterrows():
    id_ = i+1
    image_data = {"id": id_, "title": f"Image {id_}", "filename": f"{row['filename']}", "description": f"Description {id_}",
                  "palette": json.loads(row["kmeans+bitcutting_palette"])}
    data[id_] = image_data

json_data = json.dumps(data)

with open("palette_app/backend/data.json", "w") as f:
    f.write(json_data)
    
