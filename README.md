 # RAISE Project

This repository contains the RAISE (step-by-step Retrieval-Augmented Inference for Scientific rEasoning) project.

## Configuration Notes
Before running the experiments, make sure to update the following file paths in the code:
```python
# dataset.py
data_path = "path/to/your/gpqa/file.json"

# model.py
env_path = "/path/to/.env"

# solver.py
base_dir = "/path/to/your/results/directory"
```

## Getting Started
To run an experiment, use the following command:

```bash
python main.py --dataset gpqa --model_name llama --mode Step_RAG --query_mode RAISE
```

| Argument       | Description                                                                                                                                                                       |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--dataset`    | Dataset to use. Options:<br>• `gpqa`<br>• `supergpqa`<br>• `mmlu_pro`                                                                                                             |
| `--subject`    | **Required only for** `supergpqa` and `mmlu_pro`. <br>• For `supergpqa`: `science`, `engineering`<br>• For `mmlu_pro`: `chemistry`                                                |
| `--difficulty` | **Required only for** `supergpqa`. Options:<br>• `hard`<br>• `middle`<br>• `easy`                                                                                                 |
| `--model_name` | Name of the base model to use. Options: <br>• `llama` (for LLama-3.1-8B-instruct)<br>•  `mistral` (for Mistral-Small-3.1-24B-Instruct-2503)<br>•  `gpt` (for gpt-4o-mini)                                                                                                                      |
| `--mode`       | Reasoning strategy to apply. Options:<br>• `Direct_COT`<br>• `Direct_PS`<br>• `Direct_RAG`<br>• `step_back`<br>• `PD` (Problem Decomposition)<br>• `PD_step_back`<br>• `Step_RAG` |
| `--query_mode` | **Required only for** `Direct_RAG` and `Step_RAG`. Options:<br>• `question`<br>• `RAISE`<br>• `step-back`<br>• `hyde`                                                                 |
