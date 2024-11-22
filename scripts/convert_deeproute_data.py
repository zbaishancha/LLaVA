import json
import os
import uuid
import random

"""
[
  {
    "id": "997bb945-628d-4724-b370-b84de974a19f",
    "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
      },
      {
        "from": "gpt",
        "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
      },
    ]
  },
  ...
]
{
        "id": "1",
        "image": [
            "images/french_toast/278421.jpg",
            "images/onion_rings/1074382.jpg",
            "images/spring_rolls/419151.jpg",
            "images/carrot_cake/263995.jpg",
            "images/eggs_benedict/44135.jpg"
        ],
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<image>\n<image>\n<image>\n<image>\nWhat food are these five pictures about?"
            },
            {
                "from": "gpt",
                "value": "french toast, onion rings, spring rolls, carrot cake, eggs benedict"
            }
        ]
}
"""


def generate_data(image_path_list, question_set):
    """
    生成符合指定格式的数据集。

    :param image_path_list: 包含多组 image path 的列表，每组是一个 list。
    :param question_set: 包含 human value 的 question 集合。
    :return: 生成的数据列表。
    """
    data = []

    for image_group in image_path_list:
        # 随机生成唯一 ID
        unique_id = str(uuid.uuid4())

        # 从 question set 中随机抽取一个 human value
        question = random.choice(question_set)

        # 构造单个数据字典
        entry = {
            "id": unique_id,
            "image": image_group,
            "conversations": [
                {
                    "from": "human",
                    "value": question
                },
                {
                    "from": "gpt",
                    "value": ""
                }
            ]
        }

        data.append(entry)

    return data
  

with open("playground/data/LingoQA/val.json", 'r') as file:
    lingo_val_data = json.load(file)

with open("playground/data/deeproute/img_infos.json", 'r') as file:
    deeproute_img_path = json.load(file)


question_set = []
for item in lingo_val_data:
    if "conversations" in item:
        for conversation in item["conversations"]:
            if conversation["from"] == "human":
                question_set.append(conversation["value"])
                
question_set = question_set[::2]

image_path_list = []
for bag in deeproute_img_path:
    image_paths = [entry['image_path']['camera_1'] for entry in bag if 'camera_1' in entry['image_path']]
    i = 0
    while i + 5 <= len(image_paths):
        image_path_list.append(image_paths[i:i + 5])
        i += 5


result_data = generate_data(image_path_list, question_set)


print(f"total data length: {len(result_data)}")

with open("playground/data/deeproute/val.json", 'w') as json_file:
    json.dump(result_data, json_file, indent=4)

# 打印部分结果
for entry in result_data[:2]:  # 打印前两个数据
    print(json.dumps(entry, indent=4))