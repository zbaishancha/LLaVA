import argparse
import torch
import os
import json
import time
from tqdm import tqdm
import shortuuid
import transformers
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.train.train import CLIP_PATH
from PIL import Image
import math

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len, prompt_image_processor, object_processor = load_pretrained_model(model_path, args.model_base, model_name)
    clip_tokenizer = transformers.AutoTokenizer.from_pretrained(CLIP_PATH)
    
    # Load questions
    if "lingoqa" in args.question_file.lower() or "bdd" in args.question_file.lower() or "deeproute" in args.question_file.lower():
        with open(args.question_file, 'r', encoding='utf-8') as file:  
            data = json.load(file)
        questions = [
            {
                "question_id": example["id"],
                "image_path_list": example["image"],
                "text": example["conversations"][0]["value"].replace("<image>\n", ""),
                **({"source_id": example["source_id"], "time_length": example["time_length"]} if "bdd" in args.question_file.lower() else {})
            }
            for example in data
        ]
    else:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if "lingoqa" in args.question_file.lower():
        questions = questions[::2]

    # Performance metrics initialization
    total_frames = 0
    torch.cuda.reset_peak_memory_stats(device="cuda:0")
    start_time = time.time()

    for line in tqdm(questions):
        idx = line["question_id"]
        image_path_list = line["image_path_list"]
        qs = line["text"]
        if "bdd" in args.question_file.lower():
            source_id = line["source_id"]
            time_length = line["time_length"]
        cur_prompt = qs

        question_ids = tokenizer(qs, return_tensors="pt", padding='max_length', max_length=20, truncation=True).input_ids
        qs = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs) if model.config.mm_use_im_start_end else (DEFAULT_IMAGE_TOKEN + '\n' + qs)

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        
        images = [Image.open(os.path.join(args.image_folder, img_path)).convert("RGB") for img_path in image_path_list]
        image_tensor = process_images(images, image_processor, model.config)
        prompt_images = process_images(images, prompt_image_processor, model.config)
        object_images = process_images(images, object_processor, model.config)

        frame_start_time = time.time()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[images[0].size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                question_ids=question_ids.cuda(),
                prompt_images=prompt_images.unsqueeze(0).half().cuda(),
                object_images=object_images.unsqueeze(0).half().cuda()
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        frame_end_time = time.time()
        
        total_frames += 1

        ans_id = shortuuid.uuid()
        answer_entry = {
            "source_id": source_id,
            "time_length": time_length,
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        } if "bdd" in args.question_file.lower() else {
            "question_id": idx,
            "prompt": cur_prompt,
            "text": outputs,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {},
            "image_path_list": image_path_list
        }
        
        ans_file.write(json.dumps(answer_entry) + "\n")
        ans_file.flush()

    end_time = time.time()
    total_duration = end_time - start_time
    fps = total_frames / total_duration
    peak_memory_used = torch.cuda.max_memory_allocated(device="cuda:0") / (1024 ** 3)  # GB

    ans_file.close()
    print(f"Total frames processed: {total_frames}")
    print(f"Total time: {total_duration:.2f} seconds")
    print(f"FPS (Frames Per Second): {fps:.2f}")
    print(f"Peak GPU memory used: {peak_memory_used:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)