import json

with open('playground/data/llava_v1_5_mix665k.json', 'r') as f:
    data = json.load(f)

new_data = []

"""
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWhat are the colors of the bus in the image?"
      },
      {
        "from": "gpt",
        "value": "The bus in the image is white and red."
      },
"""
num = 0
# 遍历每个元素
for item in data:
    base_id = item['id']
    try:
        image_path = item['image']
    except:
        print(base_id)
        num += 1
        
    if "textvqa" in image_path or "VG_100K" in image_path or "ocr_vqa" in image_path:
        continue
    conversations = item['conversations']
    assert len(conversations) % 2 == 0
    for i in range(0, len(conversations), 2):
        human = conversations[i]
        gpt = conversations[i + 1]

        if human['from'] == 'human' and gpt['from'] == 'gpt':
            if not human['value'].startswith('<image>\n'):
                human['value'] = "<image>\n" + human['value']

            new_item = {
                "id": f"{base_id}_{i // 2 + 1}",
                "image": image_path,
                "conversations": [ {"from": "human", "value": human['value']},
                                   {"from": "gpt", "value": gpt['value']}]
            }

            new_data.append(new_item)

with open('playground/data/llava_v1_5_mix665k_single_conversation.json', 'w') as f:
    json.dump(new_data, f, indent=4)

print(len(new_data), num)
