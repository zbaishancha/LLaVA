import json
import pandas as pd
import jsonlines
import copy

"""
playground/data/DRAMA/val.json
playground/data/LingoQA/val.json
"""
pre_json_file = 'results/llava-v1.5-7b-task-drama-baseline.jsonl'
ori_val_file = 'playground/data/DRAMA/val_filtered.json'
csv_file_path = 'llava-v1.5-7b-task-drama-baseline.csv'


data = []
with open(pre_json_file) as file:
    for sample in jsonlines.Reader(file):
        data.append(sample)
        data.append(copy.deepcopy(sample)) # for repeat answer
        

with open(ori_val_file, 'r') as file:
    ori_data = json.load(file) 


ori_data_dict = {item['id']: item for item in ori_data}

for item in data:
    item2 = ori_data_dict.get(item['question_id'])
    item['answer'] = item.pop('text')
    item['question'] = item.pop('prompt')
    item['image'] = item2['image']
    item['gt'] = item2["conversations"][1]["value"]

df = pd.DataFrame(data)

df.to_csv(csv_file_path, index=False)

print('JSON has been converted to CSV successfully!')