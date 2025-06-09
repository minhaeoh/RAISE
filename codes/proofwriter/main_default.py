import os
import argparse
from datasets import load_dataset
from tqdm import tqdm
from solver_default import MultiHopSolver
import torch
import gc
import random
import wandb


def main():
    parser = argparse.ArgumentParser(description='Run MultiHopSolver on MMLU-STEM dataset')
    parser.add_argument('--model_id', type=str, default="meta-llama/Llama-3.1-8B-Instruct", help='Model ID to use')
    parser.add_argument('--ex_num', type=int, default=4, help='Number of examples to generate')
    parser.add_argument('--subject', type=str, default="high_school_chemistry", help='Subject to evaluate on')
    parser.add_argument('--generate_ex', action='store_true', help='Whether to generate examples')
    parser.add_argument('--review_ex', action='store_true', help='Whether to review examples')
    parser.add_argument('--zeroshot', action='store_true', help='Whether to use zero-shot approach')
    parser.add_argument('--PD_baseline', action='store_true', help='Whether to use PD baseline')
    parser.add_argument('--zeroshot_RAG', action='store_true', help='Whether to use zero-shot RAG')
    parser.add_argument('--trigger', action='store_true', help='Whether to use trigger')
    parser.add_argument('--trigger_value', type=float, default=0.8, help='Trigger value')
    parser.add_argument('--filter', action='store_true', help='Whether to filter')
    parser.add_argument('--doc_num', type=int, default=10, help='Number of documents to retrieve')
    parser.add_argument('--PD_trigger', action='store_true', help='Whether to use PD trigger')
    #parser.add_argument('--subjects', nargs='+', default=["high_school_chemistry"], help='List of subjects to evaluate on')
    args = parser.parse_args()

    # Load dataset once
    print("Loading MMLU-stem dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-STEM")
    print("Dataset loaded successfully")

    # Create a single MultiHopSolver instance
    solver = MultiHopSolver(
        model_id=args.model_id,
        ex_num=args.ex_num,
        generate_ex=args.generate_ex,
        review_ex=args.review_ex,
        zeroshot=args.zeroshot,
        PD_baseline=args.PD_baseline,
        zeroshot_RAG=args.zeroshot_RAG,
        trigger=args.trigger,
        trigger_value=args.trigger_value,
        filter=args.filter,
        doc_num=args.doc_num,
        PD_trigger=args.PD_trigger
    )


    subjects = [args.subject]

    # Iterate over subjects
    for subject in subjects:
        print(f"\nProcessing subject: {subject}")
        # Filter dataset for current subject once
        subject_data = ds['test'].filter(lambda x: x['subject'] == subject)
        print(f"Found {len(subject_data)} problems for {subject}")
        
        # Determine problem range
        start_prob = 1
        end_prob = len(subject_data)
        for prob_num in range(start_prob, end_prob + 1):
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