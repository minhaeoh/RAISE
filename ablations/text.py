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

    # Find all problem matches
    problems = list(problem_pattern.finditer(content))
    print(f"Number of problems found: {len(problems)}")
    
    data = []
    # Iterate over each problem section
    for problem_match in problems:
        problem_number = problem_match.group(1)
        start = problem_match.end()
        end = problems[problems.index(problem_match) + 1].start() if problems.index(problem_match) + 1 < len(problems) else len(content)
        problem_content = content[start:end]

        # Extract the problem content up to the "Correct Answer:"
        correct_answer_index = problem_content.find("Correct Answer:")
        problem_text = problem_content[:correct_answer_index].strip() if correct_answer_index != -1 else problem_content.strip()
        data.append(problem_content)

    return data





file_path  = '/home/minhae/multihop-RAG2/ablations/gpqa/mistral-evaluations/1.txt'
# Extract information and convert to JSON
extracted_data = extract_information(file_path)
print(extracted_data[0])
