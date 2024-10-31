import uuid
import json
import os
import random

def ensure_period(text):
    return text if text.endswith('.') else text + '.'

def construct_individual_dialogues(data_list):
    dialogues = []
    for item in data_list:
        if item.get('Risk', 'No') == 'No':
            continue

        image_path = item.get('s3_fileUrl', None)
        image_path = image_path.replace("https://s3-us-west-2.amazonaws.com/hrius.scaleapi/data/drama","playground/data/DRAMA/drama_data")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image does not exist at {image_path}.")
        
        suggestion = item.get('Suggestions (to the ego car)', '')
        suggestion_text = suggestion.split('(')[0].strip() if '(' in suggestion else suggestion
        
        
        conversation_pairs = [
            ("<image>\nIs there any risk in the scene?", ensure_period(item.get('Risk', 'No'))),
            ("<image>\nWhat is the intention of the ego-car?", ensure_period(item.get('Intention of Ego-vehicle', 'Unknown'))),
            ("<image>\nWhat is this scene?", ensure_period(item.get('Caption', 'Unknown'))),
            ("<image>\nWhat suggestions do you give for the driver to avoid risk?", ensure_period(suggestion_text))
        ]
        for question, answer in conversation_pairs:
            if answer == "N/A.":
                continue
            dialogue = {
                "id": str(uuid.uuid4()),
                "image": [image_path],
                "conversations": [
                    {
                        "from": "human",
                        "value": question
                    },
                    {
                        "from": "gpt",
                        "value": answer
                    }
                ]
            }
            dialogues.append(dialogue)
    
    return dialogues

file_path = 'playground/data/DRAMA/drama_data/integrated_output_v2.json'
with open(file_path, 'r') as f:
    data_list = json.load(f)
    
dialogues_output = construct_individual_dialogues(data_list)

train_size = int(len(dialogues_output) * 11871 / 14415)
random.shuffle(dialogues_output)
train_data = dialogues_output[:train_size]
val_data = dialogues_output[train_size:]

with open('playground/data/DRAMA/train.json', 'w') as train_file:
    json.dump(train_data, train_file, indent=4)

with open('playground/data/DRAMA/val.json', 'w') as val_file:
    json.dump(val_data, val_file, indent=4)

print("Train and validation files have been created.")