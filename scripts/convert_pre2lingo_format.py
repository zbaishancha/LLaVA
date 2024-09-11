import json
import pandas as pd
import jsonlines
import copy
pre_json_file = 'results/llava-v1.5-7b-task-img-token-reweight.jsonl'
ori_val_file = 'playground/data/LingoQA/val.json'
csv_file_path = 'llava-v1.5-7b-task-img-token-reweight.csv'


data = []
with open(pre_json_file) as file:
    for sample in jsonlines.Reader(file):
        data.append(sample)
        data.append(copy.deepcopy(sample)) # for repeat answer
        

with open(ori_val_file, 'r') as file:
    ori_data = json.load(file) 

for item, ori_item in zip(data, ori_data):
    assert item['question_id'] == ori_item['id']
    item['answer'] = item.pop('text')
    item['question'] = item.pop('prompt')
    item['segment_id'] = ori_item['segment_id']
    item['image'] = ori_item['image']
    item['gt'] = ori_item["conversations"][1]["value"]


df = pd.DataFrame(data)

df.to_csv(csv_file_path, index=False)

print('JSON has been converted to CSV successfully!')