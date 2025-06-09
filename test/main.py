import argparse
from tqdm import tqdm
from solver import MultiHopSolver
import torch
import gc
import dataset
from model import *


def main():
    parser = argparse.ArgumentParser(description='Run MultiHopSolver on MMLU-STEM dataset')
    parser.add_argument('--dataset', type=str, default="gpqa", help='Dataset to use: [gpqa, supergpqa, mmlu_stem, mmlu_pro]')
    parser.add_argument('--subject', type=str, default=None, help='Subject to process for mmlu_stem, mmlu_pro, supergpqa')
    parser.add_argument('--difficulty', type=str, default=None, help='Difficulty level for supergpqa')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo", help='Model to use')
    parser.add_argument('--mode', type=str, default="Step_RAG", help='Mode to use')
    parser.add_argument('--trigger', action='store_true', help='Whether to use trigger')
    parser.add_argument('--trigger_value', type=float, default=0.8, help='Trigger value')
    parser.add_argument('--doc_num', type=int, default=10, help='Number of documents to retrieve')
    parser.add_argument('--query_mode', type=str, default="subq", help='Query mode')
    args = parser.parse_args()

    # Load dataset 
    config = dataset.DatasetConfig(
        dataset_name=args.dataset,
        subject=args.subject,
        difficulty=args.difficulty
    )
    data = dataset.load_and_preprocess_dataset(config)

    # Create a single MultiHopSolver instance
    solver = MultiHopSolver(
        model = args.model,
        mode = args.mode,
        trigger = args.trigger,
        trigger_value = args.trigger_value,
        doc_num = args.doc_num,
        query_mode = args.query_mode,
    )

    
    print(f"Found {len(data)} problems for {args.dataset}")
    
    # Determine problem range
    start_prob = 1
    end_prob = len(data)
    for prob_num in tqdm(range(start_prob, end_prob + 1)):
        print(f"\nProcessing problem {prob_num}...")
        # Get problem directly from filtered dataset
        problem = data[prob_num - 1]
        solver.solve_problem(prob_num, problem)
        
        # Clear GPU memory after each problem
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()