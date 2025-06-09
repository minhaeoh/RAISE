import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from solver_deep import MultiHopSolver
import torch
import gc
import json

def main():
    parser = argparse.ArgumentParser(description='Run MultiHopSolver on MMLU-STEM dataset')
    parser.add_argument('--mode', type=str, default="Step_RAG", help='Mode to use')
    parser.add_argument('--ex_num', type=int, default=5, help='Number of examples to generate')
    parser.add_argument('--trigger', action='store_true', help='Whether to use trigger')
    parser.add_argument('--trigger_value', type=float, default=0.8, help='Trigger value')
    parser.add_argument('--doc_num', type=int, default=10, help='Number of documents to retrieve')
    parser.add_argument('--query_mode', type=str, default="subq", help='Query mode')
    args = parser.parse_args()

    # Load dataset once
    subject = "gpqa_diamond"
    with open('/home/minhae/multihop-RAG2/codes/gpqa/gpqa_diamond_problems.json', 'r', encoding='utf-8') as f:
        ds = json.load(f)
    print("Dataset loaded successfully")

    # Create a single MultiHopSolver instance
    solver = MultiHopSolver(
        mode = args.mode,
        ex_num = args.ex_num,
        trigger = args.trigger,
        trigger_value = args.trigger_value,
        doc_num = args.doc_num,
        query_mode = args.query_mode
    )

    subject_data = ds
    print(f"Found {len(subject_data)} problems for {subject}")
    
    # Determine problem range
    start_prob = 1
    end_prob = len(subject_data)
    for prob_num in tqdm(range(start_prob, end_prob + 1)):
        print(f"\nProcessing problem {prob_num}...")
        # Get problem directly from filtered dataset
        problem = subject_data[prob_num - 1]
        solver.solve_problem(subject, prob_num, problem)
        
        # Clear GPU memory after each problem
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()