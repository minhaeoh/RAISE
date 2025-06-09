import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from solver import MultiHopSolver
import torch
import gc
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Run MultiHopSolver on MMLU-STEM dataset')
    parser.add_argument('--mode', type=str, default="Step_RAG", help='Mode to use')
    parser.add_argument('--ex_num', type=int, default=5, help='Number of examples to generate')
    parser.add_argument('--trigger', action='store_true', help='Whether to use trigger')
    parser.add_argument('--trigger_value', type=float, default=0.8, help='Trigger value')
    parser.add_argument('--doc_num', type=int, default=10, help='Number of documents to retrieve')
    parser.add_argument('--query_mode', type=str, default="subq", help='Query mode')
    parser.add_argument('--api_key', type=int, default=0, help='API key')
    parser.add_argument('--subject', type=str, default="science", help='Subject')
    parser.add_argument('--diff', type=str, default="hard", help='Difficulty level')
    parser.add_argument('--prompt', action='store_true', help='Whether to change prompt')
    args = parser.parse_args()

    # Load dataset once
    ds = load_dataset("m-a-p/SuperGPQA")
    print("Dataset loaded successfully")
    data = ds['train']

    np.random.seed(42)
    subjects = ['philosophy', 'military science', 'history', 'management', 'economics', 'science', 'engineering', 'education', 'law', 'agronomy', 'sociology', 'literature and arts', 'medicine']
    subject = args.subject.lower()
    diff = args.diff
    if subject=='medicine' and diff=='hard':
        subject_data = data.filter(lambda x: x['discipline'].lower() == subject and x['difficulty'] == diff)
        idx = list(range(1, len(subject_data)+1))
    else:
        subject_data = data.filter(lambda x: x['discipline'].lower() == subject and x['difficulty'] == diff)
        print(f"Found {len(subject_data)} problems for {subject}")
        idx = np.random.randint(1, len(subject_data), size=199)
     
    # Create a single MultiHopSolver instance
    solver = MultiHopSolver(
        mode = args.mode,
        ex_num = args.ex_num,
        trigger = args.trigger,
        trigger_value = args.trigger_value,
        doc_num = args.doc_num,
        query_mode = args.query_mode,
        api_key = args.api_key
    )

    
    print(f"Found {len(idx)} problems for {subject}")
    
    start = True
    # Determine problem range
    for prob_num in tqdm(idx):
        print(f"\nProcessing problem {prob_num}...")
        # Get problem directly from filtered dataset
        problem = subject_data[int(prob_num) - 1]
        solver.solve_problem(f"{subject}_{diff}", prob_num, problem=problem, start=start)
        start = False
        
        # Clear GPU memory after each problem
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()