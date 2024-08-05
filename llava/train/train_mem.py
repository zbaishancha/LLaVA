import os
from llava.train.train import train

conda_path = "/opt/conda/bin/"
os.environ['PATH'] = f'{conda_path}:{os.environ["PATH"]}'

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
