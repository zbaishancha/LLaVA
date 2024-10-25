import json
import pandas as pd
import jsonlines
import copy
pre_json_file = 'results/llava-v1.5-7b-task-lora-drivelm.jsonl'
test_eval_data_path = 'playground/data/DriveLM/val_llama_q_only.json'
save_file_path = 'llava-v1.5-7b-task-lora-drivelm.json'


"""
[
    {
        "id": "f0f120e4d4b0441da90ec53b16ee169d_4a0798f849ca477ab18009c3a20b7df2_0",
        "question": "<image>\nWhat are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision.",
        "gt_answer": "There is a brown SUV to the back of the ego vehicle, a black sedan to the back of the ego vehicle, and a green light to the front of the ego vehicle. The IDs of these objects are <c1,CAM_BACK,1088.3,497.5>, <c2,CAM_BACK,864.2,468.3>, and <c3,CAM_FRONT,1043.2,82.2>.",
        "answer": "In the current scene, there are several important objects that need to be considered for the future reasoning and driving decision. These objects include a car driving down the street, a traffic light, a stop sign, a street sign, and a building. The car's position and speed, as well as the traffic light's status, are crucial factors in determining the driver's next move. The stop sign and street sign provide information about the street's name and any potential restrictions or directions. The building in the background adds context to the scene, providing a sense of the surroundings and the environment in which the car is driving."
    },
]
"""


data = []
with open(pre_json_file) as file:
    for sample in jsonlines.Reader(file):
        data.append(sample)

with open(test_eval_data_path, 'r') as file:
    test_eval_data_list = json.load(file)      

out_data = []
for item, test_data in zip(data, test_eval_data_list):
    assert item["question_id"] == test_data['id']
    sample = dict()
    sample["id"] = item['question_id']
    sample['question'] = test_data["conversations"][0]["value"]
    sample["gt_answer"] = test_data["conversations"][1]["value"]
    sample['answer'] = item['text']
    out_data.append(sample)


with open(save_file_path, 'w') as  f:
    json.dump(out_data, f)
        

print('JSON has been saved successfully!')