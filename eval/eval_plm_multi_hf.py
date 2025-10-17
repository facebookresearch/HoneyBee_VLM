# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the HoneyBee License found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Union

import torch
from datasets import load_dataset
import json
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
import argparse
from pathlib import Path
from enum import Enum

# Import custom modules
from data import (
    DatasetType,
    DatasetConfig,
    get_dataset_config,
    get_formatted_instruction,
    process_response,
    save_descriptions,
    load_image_dataset,
    get_processed_response
)
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from transformers import AutoProcessor, AutoModelForImageTextToText

import io
import base64
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INSTRUCTION =  "\n\nYour final answer MUST BE put in \\boxed{}."

def pil_to_base64(image_pil, format="PNG"):
    buffered = io.BytesIO()
    image_pil.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def base64_to_pil(base64_string):
    img_data = base64.b64decode(base64_string)
    image_pil = Image.open(io.BytesIO(img_data))
    return image_pil

class InstanceDataset(Dataset):

    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        for k in item:
            if k == 'options' or k == 'choices':
                if item[k] == None:
                    item[k] = ""
                else:
                    item[k] = str(item[k])
        if 'image_url' in item:
            image_url = item['image_url']
            image_str = pil_to_base64(image_url)
            item['image_url'] = image_str
        instance = {'index': index, 'item': item}
        return instance

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on various math datasets')
    parser.add_argument('--dataset', type=str, choices=['mathvista', 'mathverse', 'mathvision', 'mathvision-mini', 'hallusionbench', 'mmmu-pro-vision', 'we-math', 'math500', 'gpqa'],
                      default='mathvista', help='Dataset to evaluate on')
    parser.add_argument('--model_path', type=str, help='Path to the model', default="facebook/Perception-LM-1B")
    parser.add_argument('--name', type=str, help='model save name', default="plm")

    args = parser.parse_args()
    
    device = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(f'cuda:{device}')
    
    # Configuration
    dataset_type = DatasetType(args.dataset)
    dataset_config = get_dataset_config(dataset_type)
    
    output_folder = f"./outputs/{dataset_type.value}_{args.name}"
    os.makedirs(output_folder, exist_ok=True)

    MODEL_PATH = args.model_path
    processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to(device)

    # Load dataset
    logger.info(f"Loading dataset {dataset_config.name}")
    data = load_image_dataset(dataset_config)
    
    dist.init_process_group()
    dataset = InstanceDataset(data)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)

    # Load model
    local_rank = int(os.environ['LOCAL_RANK'])
    logger.info(f"Loaded model {args.model_path} | local rank: {local_rank}")

    for batch in tqdm(dataloader):
        
        index = batch['index'][0].item()
        output_file = os.path.join(output_folder, f'{index}.json')

        if not os.path.exists(output_file):

            item = batch['item']
            for k in item:
                if len(item[k]) > 0:
                    if k == 'choices' or k == 'options':
                        try:
                            item[k] = eval(item[k][0])
                        except:
                            item[k] = item[k][0]
                    else:
                        item[k] = item[k][0]
                if k == 'image_url':
                    item['image_url'] = base64_to_pil(item['image_url'])

            correct_flag = 0

            formatted_instruction = get_formatted_instruction(dataset_type, item)
            formatted_instruction = formatted_instruction + INSTRUCTION

            if 'image_url' in item:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": item['image_url']},
                            {"type": "text", "text": formatted_instruction},
                        ],
                    }
                ]
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": formatted_instruction},
                        ],
                    }
                ]

            inputs = processor.apply_chat_template(
                [conversation],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            generate_ids = model.generate(**inputs, max_new_tokens=2048)
            input_length = inputs["input_ids"].shape[1]
            generate_ids_without_inputs = generate_ids[:, input_length:]
            answer = processor.batch_decode(generate_ids_without_inputs, skip_special_tokens=True)[0]

            processed_response = get_processed_response(dataset_type, item)
            print(f'processed_response: {processed_response}')            
            
            if 'image_url' in item:
                del item['image_url']

            description = {
                'index': index,
                'item': json.dumps(item),
                'formatted_instruction': formatted_instruction,
                'processed_response': processed_response,
                'answer': answer
            }

            with open(output_file, 'w') as f:
                json.dump(description, f, indent = 4)

if __name__ == "__main__":
    main()