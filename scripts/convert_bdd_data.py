import json
import uuid
import os


base_img_path = "playground/data/BDD-X/BDD_X_imgs_select/"

def transform_data(data):
    transformed_data = []
    for entry in data:
        base_image = entry["image"].split(".")[0]  # 提取基础图像名称
        idx_list = entry["idx_list"]
        time_length = entry["time_length"] if "time_length" in entry.keys() else 0.0
        source_id = entry["id"]

        image_paths = [f"{base_img_path}{base_image}_{idx}.png" for idx in idx_list]
        if not all(os.path.exists(path) for path in image_paths):
            continue
        
        for i, convo_pair in enumerate(zip(entry["conversations"][::2], entry["conversations"][1::2])):
            human_convo, gpt_convo = convo_pair
            
            # 删除 human 文本中的 "\n<video>"
            human_text = human_convo["value"].replace("\n<video>", "")
            human_text = f"<image>\n{human_text}"  # 在 human 文本前添加 "<image>\n"
            
            new_entry = {
                "source_id": source_id,
                "id": str(uuid.uuid4()),
                "image": image_paths,
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": gpt_convo["value"]}
                ],
                "time_length": time_length
            }
            transformed_data.append(new_entry)
    return transformed_data

with open("playground/data/BDD-X/BDD_X_training_label.json", 'r') as file:
    data = json.load(file)

transformed_data = transform_data(data)

with open("playground/data/BDD-X/train.json", 'w') as json_file:
    json.dump(transformed_data, json_file, indent=4)