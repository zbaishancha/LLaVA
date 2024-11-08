import json
from collections import defaultdict
import jsonlines

def transform_data(source_data, label_data):
    # 第一步：将source_data按照source_id分组并合并text到answer列表中
    grouped_data = defaultdict(list)
    for item in source_data:
        grouped_data[item["source_id"]].append(item["text"])
    
    # 第二步：构建新的数据结构
    transformed_data = []
    for source_id, answers in grouped_data.items():
        new_id = f"{source_id}_0"  # 添加 "_0" 后缀
        # 从label_data中查找匹配的label
        label_item = next((item["label"] for item in label_data if item["id"] == new_id), None)
        
        if label_item is not None:  # 只有在找到对应的label时，才添加到结果中
            new_entry = {
                "id": new_id,
                "answer": answers,
                "label": label_item
            }
            transformed_data.append(new_entry)
    
    return transformed_data


with open("playground/data/BDD-X/DriveGPT4_output.json", 'r') as file:
    label_data = json.load(file)
    
source_data = []
with open("results/checkpoints/llava-v1.5-7b-task-bdd-x.jsonl",) as file:
    for sample in jsonlines.Reader(file):
        source_data.append(sample)

# 转换数据
transformed_data = transform_data(source_data, label_data)

# 保存到新的 JSON 文件
with open("bdd_output.json", "w") as f:
    json.dump(transformed_data, f, indent=4)