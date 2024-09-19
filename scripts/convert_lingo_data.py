import yaml
import json
import cv2
import os
import copy
import re
import sys
import argparse
import pandas as pd
import pyarrow.parquet as pq
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

data_path_list = ["playground/data/LingoQA/action/train.parquet",
                  "playground/data/LingoQA/scenery/train.parquet"]

conversations = []

IMAGE_STR = "<image>\n"

for data_path in data_path_list:
    dirname = os.path.dirname(data_path)
    parquet_file = pq.ParquetFile(data_path)
    data = parquet_file.read().to_pandas()
    data = data.to_dict(orient='records')
    for sample in data:
        conversation = dict()
        conversation["id"] = sample['question_id']
        conversation["segment_id"] = sample['segment_id']
        conversation["image"] = [os.path.join(dirname, img_path) for img_path in sample["images"]]
        question, answer = dict(), dict()
        question["from"] = "human"
        question["value"] = IMAGE_STR * len(conversation["image"]) + sample['question']
        answer["from"] = "gpt"
        answer["value"] = sample['answer']
        conversation["conversations"] = [question, answer]
        conversations.append(conversation)


print(f"total data length: {len(conversations)}")


with open("playground/data/LingoQA/train.json", 'w') as json_file:
    json.dump(conversations, json_file, indent=4)