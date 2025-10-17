# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the HoneyBee License found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import argparse
from tqdm import tqdm 

from mathruler.grader import grade_answer, extract_boxed_content


def is_mcq(item):
    """Check if the item is a multiple choice question."""
    if 'choices' in item:
        if item['choices'] is not None:
            return True 
    if 'options' in item:
        if item['options'] is None:
            return False
        if len(item['options']) > 0:
            return True
    if 'answer_type' in item:
        if item['answer_type'] == 'multiple choice':
            return True
    return False


def score_file(filename):
    """Score the predictions in a file against ground truth using MathRuler."""
    with open(filename, "r") as f:
        data = json.load(f)
    
    correct = 0
    total = 0
    
    for example in tqdm(data, desc="Scoring predictions"):
        correct_flag = 0
        total += 1
        item = json.loads(example['item'])
        answer = example['answer']
        processed_response = example['processed_response']
        
        # Extract answer from model output
        answer_keys = [
            "Answer: ", "The answer is: ", "The answer is ", "the answer is: ",
            "--------Summary--------\n", "**Answer** ", "the correct answer is ",
            "**Answer:** ", "answer: ", "**Answer**: "
        ]
        
        if "\\boxed{" in answer:
            answer = extract_boxed_content(answer)     
        elif "</answer>" in answer:
            answer = answer.split("<answer>")[-1].split("</answer>")[0].strip()
        else:
            for key in answer_keys:
                if key in answer:
                    answer = answer.split(key)[-1].strip()
                    if "\n" in answer:
                        answer = answer.split("\n")[0]
                    break
            answer = answer.strip()
        
        if answer.endswith('.'):
            answer = answer[:-1]            

        mcq = is_mcq(item)
        
        # Check for MCQ format matches
        if mcq and (f'({processed_response})' in answer or f'{processed_response}. ' in answer):
            correct += 1
            correct_flag = 1
        
        # Grade using MathRuler
        if correct_flag == 0:
            score = grade_answer(processed_response, answer)
            if processed_response.lower() == answer.lower() or score:
                correct += 1

    accuracy = 100 * correct / total
    
    results = {
        'total': total,
        'correct': correct,
        'accuracy': accuracy
    }
    
    return results


def collect_data(output_dir):
    """Collect all JSON files from a directory and merge them."""
    if not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    if not files:
        raise ValueError(f"No JSON files found in directory: {output_dir}")
    
    files.sort()
    results = []
    
    print(f"Found {len(files)} JSON files in {output_dir}")
    for file in tqdm(files, desc="Collecting files"):
        filepath = os.path.join(output_dir, file)
        with open(filepath, 'r') as f:
            results.append(json.load(f))
    
    # Save merged file
    output_dir_clean = output_dir.rstrip('/')
    merged_file = output_dir_clean + ".json"
    with open(merged_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Merged data saved to: {merged_file}")
    return merged_file


def main(args):
    output_dir = args.output_dir
    
    print(f"\n{'='*60}")
    print(f"Evaluating outputs from: {output_dir}")
    print(f"{'='*60}\n")
    
    # Collect and merge individual JSON files
    merged_file = collect_data(output_dir)
    
    # Score the merged file
    print("\nScoring predictions...")
    results = score_file(merged_file)
    
    # Display results
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Total examples:        {results['total']}")
    print(f"Correct answers:       {results['correct']}")
    print(f"Accuracy:              {results['accuracy']:.2f}%")
    print(f"{'='*60}\n")
    
    # Save detailed results
    output_name = os.path.basename(output_dir.rstrip('/'))
    results_file = f"results_{output_name}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {results_file}")
    
    return results

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate model outputs on math datasets using MathRuler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate outputs from a specific directory
  python eval.py --output_dir ./outputs/mathverse_plm1b/
        """
    )
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to directory containing instance-level JSON files')
    
    args = parser.parse_args()
    main(args)