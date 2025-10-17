# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the HoneyBee License found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from PIL import Image
from pathlib import Path
from loguru import logger
from datasets import load_dataset

class DatasetType(Enum):
    MATHVISTA = "mathvista"
    MATHVERSE = "mathverse"
    MATHVISION = "mathvision"
    MATHVISION_MINI = "mathvision-mini"
    HALLUSIONBENCH = "hallusionbench"
    MMMU_PRO_VISION = "mmmu-pro-vision"
    DYNAMATH = "dynamath"
    LOGICVISTA = "logicvista"
    WE_MATH = "we-math"
    MATH500 = "math500"
    GPQA = "gpqa"

@dataclass
class DatasetConfig:
    name: str
    split: str
    response_field: str
    pid: Optional[str] = None
    image_field: Optional[str] = None
    instruction_field: Optional[str] = None
    subset: Optional[str] = None
    choices_field: Optional[str] = None
    options_field: Optional[str] = None
    problem_version: Optional[str] = None
    answer_type: Optional[str] = None

def get_dataset_config(dataset_type: DatasetType) -> DatasetConfig:
    configs = {
        DatasetType.MATHVISTA: DatasetConfig(
            name="AI4Math/MathVista",
            split="testmini",
            pid="pid",
            image_field="decoded_image",
            instruction_field="query",
            response_field="answer",
            choices_field="choices"
        ),
        DatasetType.MATHVERSE: DatasetConfig(
            name="hbXNov/MATHVERSE_NUMERIC_OR_CHOICE_ANSWER",
            split="testmini",
            pid="problem_index",
            image_field="image",
            instruction_field="query_cot",
            response_field="numeric_or_choice_answer",
            problem_version="Vision Only"
        ),
        DatasetType.DYNAMATH: DatasetConfig(
            name="hbXNov/dynamath_mini",
            split="train",
            image_field="decoded_image",
            instruction_field="question",
            response_field="ground_truth",
            answer_type="answer_type"
        ),
        DatasetType.LOGICVISTA: DatasetConfig(
            name="lscpku/LogicVista",
            split="test",
            image_field="image",
            instruction_field="question",
            response_field="answer",
        ),
        DatasetType.MATHVISION: DatasetConfig(
            name="MathLLMs/MathVision",
            split="test",
            pid="id",
            image_field="decoded_image",
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.MATHVISION_MINI: DatasetConfig(
            name="MathLLMs/MathVision",
            split="testmini",
            pid="id",
            image_field="decoded_image",
            instruction_field="question",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.HALLUSIONBENCH: DatasetConfig(
            name="lmms-lab/HallusionBench",
            split="image",
            image_field="image",
            instruction_field="question",
            response_field="gt_answer"
        ),
        DatasetType.MMMU_PRO_VISION: DatasetConfig(
            name="MMMU/MMMU_Pro",
            subset="vision",
            split="test",
            image_field="image",
            response_field="answer",
            options_field="options"
        ),
        DatasetType.WE_MATH: DatasetConfig(
            name="We-Math/We-Math",
            split="testmini",
            image_field="image_path",
            response_field="answer",
            instruction_field="question",
            options_field="option"
        ),
        DatasetType.MATH500: DatasetConfig(
            name="HuggingFaceH4/MATH-500",
            split="test",
            instruction_field="problem",
            response_field="answer",
        ),
        DatasetType.GPQA: DatasetConfig(
            name="Idavidrein/gpqa",
            subset="gpqa_diamond",
            split="train",
            instruction_field="Question",
            options_field=["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
            response_field="Correct Answer"
        )
    }
    return configs[dataset_type]


def save_descriptions(descriptions: List[Dict], output_file: str) -> None:
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
        logger.info(f"Saved {len(descriptions)} descriptions to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save descriptions: {str(e)}")
        raise

def process_response(response: str, choices: Optional[List[str]], options: Optional[List[str]] = None) -> str:
    if choices is not None and len(choices) > 0:
        try:
            response_index = choices.index(response)
            return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][response_index]
        except ValueError:
            pass
    if options is not None and len(options) > 0:
        try:
            response_index = options.index(response)
            return ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'][response_index]
        except ValueError:
            pass
    return response

def format_instruction(instruction: str, options: Optional[List[str]] = None, yes_no: bool = False, vision: bool = False, is_we_math: bool = False) -> str:
    
    if is_we_math:
        options = options.split('; ')
        options = [opt[len('A. '):] for opt in options]
    else:
        if isinstance(options, str):
            if len(options) > 0:
                options = eval(options) ## convert string to list
    
    if vision:
        prompt_hint = "Hint: Please answer the question shown in the image."
        if options and len(options) > 0:
            prompt_hint += " Provide the correct option letter, e.g., A, B, C, D, E, at the end."
            choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
            return f"{prompt_hint}\nChoices:\n{choice_list}"
        return prompt_hint
    elif yes_no:
        prompt_hint = "Hint: Please answer the question requiring an answer of Yes or No."
        return f"{prompt_hint}\nQuestion: {instruction}"
    elif options and len(options) > 0:
        prompt_hint = "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end."
        choice_list = "\n".join(f"({chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"{prompt_hint}\nQuestion: {instruction}\nChoices:\n{choice_list}"
    else:
        prompt_hint = "Hint: Please answer the question requiring an answer."
        return f"{prompt_hint}\nQuestion: {instruction}"

def generate_multiple_choice_answers(data):
    """Generate multiple choice string and correct answer letter."""
    answers = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    random.shuffle(answers)

    options = ["A", "B", "C", "D"]
    options_to_answers = {letter: answer for letter, answer in zip(options, answers)}

    multiple_choice_string = ", ".join(f"{letter}: {options_to_answers[letter]}" for letter in options)
    correct_answer_letter = next(
        letter for letter, answer in options_to_answers.items() if answer == data["Correct Answer"]
    )

    return multiple_choice_string, correct_answer_letter
    
def load_image_dataset(dataset_config: DatasetConfig) -> List[Dict]:
    try:
        if dataset_config.subset:
            data = load_dataset(dataset_config.name, dataset_config.subset, split=dataset_config.split)
        else:
            data = load_dataset(dataset_config.name, split=dataset_config.split)
        
        if dataset_config.problem_version:
            data = data.filter(lambda x: x['problem_version'] == dataset_config.problem_version)
        
        items = []
        for item in data:
            if isinstance(dataset_config.image_field, list):
                dataset_item = {
                    'image_url': [item.get(x) for x in dataset_config.image_field if item.get(x) is not None],
                    'instruction': item.get(dataset_config.instruction_field, ''),
                    'response': item.get(dataset_config.response_field, ''),
                }
            elif dataset_config.image_field == None:
                dataset_item = {
                    'instruction': item.get(dataset_config.instruction_field, ''),
                    'response': item.get(dataset_config.response_field, ''),
                }
            else:
                dataset_item = {
                    'image_url': item[dataset_config.image_field],
                    'instruction': item.get(dataset_config.instruction_field, ''),
                    'response': item.get(dataset_config.response_field, ''),
                }
            if dataset_config.choices_field:
                dataset_item['choices'] = item.get(dataset_config.choices_field)
            if dataset_config.options_field:
                ## will be used for GPQA dataset
                if isinstance(dataset_config.options_field, list):
                    instance = {x: item.get(x) for x in dataset_config.options_field}
                    mcq_string, gt_answer = generate_multiple_choice_answers(instance)
                    dataset_item['options'] = mcq_string
                    dataset_item['response'] = gt_answer
                else:
                    dataset_item['options'] = item.get(dataset_config.options_field, [])
            
            if dataset_config.answer_type:
                dataset_item['answer_type'] = item.get(dataset_config.answer_type)
            
            if dataset_config.pid:
                dataset_item['pid'] = item.get(dataset_config.pid)

            items.append(dataset_item)
        return items
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

def get_formatted_instruction(dataset_type, item):
    if dataset_type == DatasetType.GPQA:
        PROMPT = """Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end.\nProblem: {problem}\nOptions: {options}"""
        formatted_instruction = PROMPT.format(problem=item['instruction'], options=item['options'])
    elif dataset_type == DatasetType.DYNAMATH:
        answer_type_hints = {
            'multiple choice': "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, E, at the end.",
            'float': "Hint: Please answer the question and provide the numerical or float answer, e.g., 9, 2.5, 1.24 at the end."
        }
        prompt_hint = answer_type_hints.get(
            item['answer_type'],
            "Hint: Please answer the question requiring an answer."
        )   
        formatted_instruction = f"{prompt_hint}\nQuestion: {item['instruction']}"
    elif dataset_type == DatasetType.MATHVISION or dataset_type == DatasetType.MATHVISION_MINI:
        formatted_instruction = format_instruction(item['instruction'], item.get('options'))
    elif dataset_type == DatasetType.HALLUSIONBENCH:
        formatted_instruction = format_instruction(item['instruction'], yes_no=True)
    elif dataset_type == DatasetType.MMMU_PRO_VISION:
        formatted_instruction = format_instruction(item['instruction'], item.get('options'), vision=True)
    elif dataset_type == DatasetType.WE_MATH:
        formatted_instruction = format_instruction(item['instruction'], item.get('options'), is_we_math=True)
    else:
        formatted_instruction = item['instruction']
    return formatted_instruction

def get_processed_response(dataset_type, item):
    
    if dataset_type == DatasetType.MMMU_PRO_VISION or dataset_type == DatasetType.WE_MATH or dataset_type == DatasetType.MATH500 or dataset_type == DatasetType.GPQA or dataset_type == DatasetType.DYNAMATH:
            processed_response = item['response']
    else:
        processed_response = process_response(
            item['response'],
            item.get('choices'),
            item.get('options')
        )
    if dataset_type == DatasetType.HALLUSIONBENCH:
        processed_response = "Yes" if processed_response == "1" else "No"
    
    return processed_response

def load_image(media_path):
    if isinstance(media_path, Image.Image):
        image = media_path
        image = image.convert("RGB")
    else:
        image = Image.open(media_path).convert("RGB")
    return image