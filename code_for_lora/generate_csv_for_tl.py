import pandas as pd
import json
import os

csv_path = 'data/train.csv'
img_dir = 'data/train'
output_jsonl = os.path.join(img_dir, 'metadata.jsonl')

df = pd.read_csv(csv_path)

print(f"Processing {len(df)} images...")

with open(output_jsonl, 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        species = str(row['label']).strip()
        
        prompt = f"a macro photo of a {species} butterfly, natural background"
        
        line = {
            "file_name": row['filename'],
            "text": prompt
        }
        f.write(json.dumps(line) + '\n')

print(f"Success! Created file in: {output_jsonl}")