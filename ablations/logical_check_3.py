import os
from openai import OpenAI
import dotenv
import json
from tqdm import tqdm
import argparse

env_path = "/home/minhae/multihop-RAG2/.env"
dotenv.load_dotenv(dotenv_path=env_path)

def generate_text(client, prompt):
    # Call GPT-4 API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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
    prompt = prompt = f"""You are given the following three items:

- Original Problem: {data['problem']}  
- Subquestion: {data['subquestion']}  
- Retrieved Document: {data['document']}

Your task is to evaluate how helpful the retrieved document is for answering the subquestion.

Please follow these instructions:
- Do not just check if the topic is related.
- Instead, check if the document includes information that helps someone reason through and solve the subquestion.
- Focus on whether the document supports actual thinking or steps needed to get the answer.

Give your final judgment using **only one** of the following ratings:
- "No relevance at all" - does not have any domain similarity
- "Partially relevant" - has domain similarity but does not contain crucial information for solving the subquestion
- "Mostly relevant" - has domain similarity and contains crucial information for solving the subquestion
- "Fully relevant" - has domain similarity and contains all crucial information for solving the subquestion

Then explain your reasoning briefly.

### Output Format:
Helpfulness Rating: <one of the 4 options above>  
Explanation: <your short explanation>"""
    
    # Use the generate_text function
    evaluation = generate_text(client, prompt)
    
    return evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLM evaluation')
    parser.add_argument('--mode', type=str, default="subq", help='Mode to use')
    args = parser.parse_args() 
    json_path = f"/home/minhae/multihop-RAG2/ablation/gpqa/{args.mode}.json"  # Replace with your actual JSON file path
    with open(json_path, 'r') as file:
        full_data = json.load(file)
    results = []
    for data in tqdm(full_data):
        # Evaluate with GPT-4
        evaluation = evaluate_with_gpt4(data)
        if "No relevance at all" in evaluation:
            eval = 0
        elif "Partially relevant" in evaluation:
            eval = 1
        elif "Mostly relevant" in evaluation:
            eval = 2
        elif "Fully relevant" in evaluation:
            eval = 3
        else:
            eval = -1
        
        result = {"problem number": data['problem number'], "subquestion number": data['subquestion number'], "evaluation": eval}
        results.append(result)

    with open(f"/home/minhae/multihop-RAG2/ablation/gpqa/{args.mode}_results_3.json", "w") as file:
        json.dump(results, file)
