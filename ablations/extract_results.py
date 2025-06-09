import json
import re
import os

def extract_information(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Creating an empty file.")
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump([], file)  # Create an empty JSON array
        return []

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expressions to match the required patterns
    problem_pattern = re.compile(r'Problem (\d+):')
    subquestion_pattern = re.compile(r'Subquestion (\d+):')
    query_pattern = re.compile(r'Query: (.*?)\n', re.DOTALL)
    distances_pattern = re.compile(r'Distances: (\[.*?\]|.*?)\n', re.DOTALL)
    document_pattern = re.compile(r'Documents:\s*(.*?)\s*Solution:', re.DOTALL)

    # Find all problem matches
    problems = list(problem_pattern.finditer(content))
    data = []
    print(f"Number of problems found: {len(problems)}")
    
    # Iterate over each problem section
    for problem_match in problems:
        problem_number = problem_match.group(1)
        start = problem_match.end()
        end = problems[problems.index(problem_match) + 1].start() if problems.index(problem_match) + 1 < len(problems) else len(content)
        problem_content = content[start:end]

        # Extract the problem content up to the "Correct Answer:"
        correct_answer_index = problem_content.find("Correct Answer:")
        problem_text = problem_content[:correct_answer_index].strip() if correct_answer_index != -1 else problem_content.strip()

        # Split problem content by subquestions
        subquestions = subquestion_pattern.split(problem_content)
        print(f"len(subquestions): {len(subquestions)}")
        for j in range(1, len(subquestions), 2):
            subquestion_number = subquestions[j]
            subquestion_content = subquestions[j + 1]

            # Extract the subquestion content up to the first newline
            subquestion_text = subquestion_content.split('\n', 1)[0].strip()

            # Extract query, distances, and documents for each subquestion
            query_match = query_pattern.search(subquestion_content)
            distances_match = distances_pattern.search(subquestion_content)
            document_match = document_pattern.search(subquestion_content)

            entry = {
                "problem number": problem_number,
                "problem": problem_text,  # Add problem content
                "subquestion number": subquestion_number,
                "subquestion": subquestion_text,
                "query": query_match.group(1).strip() if query_match else None,
                "document distances": distances_match.group(1).strip() if distances_match else None,
                "document": document_match.group(1).strip() if document_match else None
            }
            data.append(entry)

    return data

# Specify the path to your detailed.txt file
base_path = '/home/minhae/multihop-RAG2/ablation/gpqa/llama'
json_file_name = "ours_4.json"
json_path = os.path.join(base_path, json_file_name)
os.makedirs(base_path, exist_ok=True)

file_path  = '/home/minhae/multihop-RAG2/final-results/gpqa_diamond-0512/llama/Step_RAG_q2_trigger0.84_doc10/20250518_110755/detailed.txt'
# Extract information and convert to JSON
extracted_data = extract_information(file_path)

# Ensure json_path is a file, not a directory
if os.path.isdir(json_path):
    raise IsADirectoryError(f"The path {json_path} is a directory, not a file.")

# Save to a JSON file
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)

print("Data extracted and saved to extracted_results.json")

