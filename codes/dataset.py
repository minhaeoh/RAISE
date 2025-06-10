from datasets import load_dataset
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import random


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing"""
    dataset_name: str
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    cache_dir: Optional[str] = None
    # Add other configuration parameters as needed


def load_mmlu_stem(config: DatasetConfig) -> Dict[str, Any]:
    print("Loading MMLU-STEM dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-STEM", cache_dir=config.cache_dir)
    print("Dataset loaded successfully")
    if config.subject:
        subject = config.subject
        data = ds['test'].filter(lambda x: x['subject'] == subject)
        print(f"Found {len(data)} problems for {subject}")
    else:
        data = ds['test']
        print(f"Found {len(data)} problems for all subjects")
    
    print("Formatting data...")
    problems = []
    for i in range(len(data)):
        problem = {}
        problem['question'] = data[i]['question']
        problem['answerKey'] = data[i]['answer']
        answer_choices = {}
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(data[i]['choices'])]
        formatted_choices = [f"({i}) {choice}" for i, choice in zip(options, data[i]['choices'])]
        answer_choices['text'] = "\n".join(formatted_choices)
        answer_choices['label'] = options
        problem['choices'] = answer_choices
        problems.append(problem)

    print("Data formatted successfully")
    return problems


def load_mmlu_pro(config: DatasetConfig) -> Dict[str, Any]:
    print("Loading MMLU-Pro dataset...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=config.cache_dir)
    print("Dataset loaded successfully")
    if config.subject:
        subject = config.subject
        data = ds['test'].filter(lambda x: x['category'] == subject)
        print(f"Found {len(data)} problems for {subject}")
    else:
        data = ds['test']
        print(f"Found {len(data)} problems for all subjects")
    
    print("Formatting data...")
    problems = []
    for i in range(len(data)):
        problem = {}
        problem['question'] = data[i]['question']
        problem['answerKey'] = data[i]['answer']
        answer_choices = {}
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(data[i]['options'])]
        formatted_choices = [f"({i}) {choice}" for i, choice in zip(options, data[i]['options'])]
        answer_choices['text'] = "\n".join(formatted_choices)
        answer_choices['label'] = options
        problem['choices'] = answer_choices
        problems.append(problem)
        
    print("Data formatted successfully")
    return problems

def load_gpqa(config: DatasetConfig) -> Dict[str, Any]:
    print("Loading GPQA dataset...")
    data_path = "path/to/your/gpqa/file.json"  
    if not os.path.exists(data_path):
        print(f"GPQA dataset file not found at: {data_path}")
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", cache_dir=config.cache_dir)
        dataset = dataset['train']
        problems = []
        for i in range(len(dataset)):
            problem = dataset[i]
            gpqa_problem = {}
            gpqa_problem['question'] = problem['Question']
            gpqa_problem['choices'] = {}
            gpqa_problem['choices']['label'] = ['A', 'B', 'C', 'D']
            ANSWER_LABELS = ['A', 'B', 'C', 'D']
            choice_indices = [1,2,3,4]
            choice_order = random.sample(choice_indices, len(choice_indices))
            ans_idx = choice_order.index(4)
            ordered_choices = [
                    problem[f"Incorrect Answer {i}"] if i != 4 else problem["Correct Answer"]
                    for i in choice_order
                ]

            ordered_choices = [
                f"({ANSWER_LABELS[i]}) {choice}" for i, choice in enumerate(ordered_choices)
            ]
            gpqa_problem['choices']['text'] = ordered_choices
            gpqa_problem['answerKey'] = ANSWER_LABELS[ans_idx]
                        
            problems.append(gpqa_problem)

        # Save the problems list to a JSON file
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
    else:
        with open(data_path, 'r', encoding='utf-8') as f:
            problems = json.load(f)

    print(f"Found {len(problems)} problems")
    print("Data formatted successfully")

    return problems


def load_supergpqa(config: DatasetConfig) -> Dict[str, Any]:
    print("Loading SuperGPQA dataset...")
    ds = load_dataset("m-a-p/SuperGPQA", cache_dir=config.cache_dir)
    print("Dataset loaded successfully")
    if config.subject:
        subject = config.subject
        data = ds['train'].filter(lambda x: x['discipline'].lower() == subject and x['difficulty'] == config.difficulty)
    else: 
        raise ValueError(f"Enter subject and difficulty. Available subjects: {list(ds['train'].unique('discipline'))}")
    
    print("Formatting data...")
    problems = []
    for i in range(len(data)):
        problem = {}
        problem['question'] = data[i]['question']
        problem['answerKey'] = data[i]['answer_letter']
        answer_choices = {}
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(data[i]['options'])]
        formatted_choices = [f"({i}) {choice}" for i, choice in zip(options, data[i]['options'])]
        answer_choices['text'] = "\n".join(formatted_choices)
        answer_choices['label'] = options
        problem['choices'] = answer_choices
        problems.append(problem)
    print(f"Found {len(problems)} problems")    
    print("Data formatted successfully")
    return problems

def get_dataset_loader(dataset_name: str):
    """
    Get the appropriate dataset loader function based on dataset name
    Args:
        dataset_name: Name of the dataset to load
    Returns:
        Dataset loader function
    """
    dataset_loaders = {
        "mmlu_stem": load_mmlu_stem,
        "mmlu_pro": load_mmlu_pro,
        "gpqa": load_gpqa,
        "supergpqa": load_supergpqa,
    }
    
    if dataset_name not in dataset_loaders:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Available datasets: {list(dataset_loaders.keys())}")
    
    return dataset_loaders[dataset_name]


def load_and_preprocess_dataset(config: DatasetConfig) -> Dict[str, Any]:
    """
    Main function to load and preprocess dataset based on configuration
    Args:
        config: Dataset configuration including dataset name and other parameters
    Returns:
        Processed dataset
    """
    loader = get_dataset_loader(config.dataset_name)
    return loader(config)


if __name__ == "__main__":
    # Example usage
    config = DatasetConfig(
        dataset_name="gpqa"
    )
    dataset = load_and_preprocess_dataset(config)

