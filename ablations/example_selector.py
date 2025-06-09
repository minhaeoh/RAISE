import os
from openai import OpenAI
import dotenv
import json
from tqdm import tqdm
import argparse
import re 

env_path = "/home/minhae/multihop-RAG2/.env"
dotenv.load_dotenv(dotenv_path=env_path)

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

        data.append(problem_content)

    return data

def generate_text(client, prompt):
    # Call GPT-4 API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an evaluator for model-generated answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

def evaluate_with_gpt4(data):
    # Initialize OpenAI client with API key
    api_key = os.getenv("OPENAI_API_KEY")  # Ensure you have set this environment variable
    print("Got api key")
    client = OpenAI(api_key=api_key)
    print("Got client")
    prompt = f"""You are given a complex scientific problem, which has been decomposed into several subquestions.  
For each subquestion, the following information is provided:

- Subquestion
- RAISE Query
- Subquestion Documents: Document retrieved by baseline RAG, which is Document 1
- RAISE Query Documents: Document retrieved using RAISE query, which is Document 2
- Evaluation from GPT-4o-mini indicating which document is more useful and why

Your task is to identify subquestions where the RAISE-retrieved document (Document 2) is not just superficially or semantically similar, but instead provides **substantive knowledge that meaningfully helps with reasoning through the subquestion**.

Please pay attention to cases where the RAISE document:
- Contains important **scientific equations**, definitions, or conceptual explanations
- Helps clarify key **reasoning steps or logical relationships**
- Provides information that is **more directly relevant to solving the subquestion** than the RAG document

Be thoughtful but not overly strict. If the RAISE document adds genuine value to the reasoning process—even if not absolutely required—include it.

For each such subquestion, list the **subquestion number** and briefly explain why the RAISE document is more logically helpful.

If there are no strong examples, but some are worth noting, say so.

If you truly find **no examples at all**, say:
> "No subquestions show clear logical advantages from the RAISE document, though all were reasonable attempts."
"""
    # Use the generate_text function
    evaluation = generate_text(client, prompt)
    
    return evaluation

if __name__ == "__main__":
    file_path  = '/home/minhae/multihop-RAG2/ablations/gpqa/mistral-evaluations/151.txt'
    # Extract information and convert to JSON
    extracted_data = extract_information(file_path)
    results = []
    for i, data in tqdm(enumerate(extracted_data)):
        result = evaluate_with_gpt4(data)
        results.append({
            "problem number": i + 1,
            "evaluation": result
        })

    with open(f"/home/minhae/multihop-RAG2/ablations/gpqa/mistral-example/6.json", "w") as file:
        json.dump(results, file)
