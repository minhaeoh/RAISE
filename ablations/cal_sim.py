import torch
from transformers import (
    DPRQuestionEncoderTokenizer, 
    DPRQuestionEncoder,
    DPRContextEncoderTokenizer,
    DPRContextEncoder
)
import json
import torch.nn.functional as F  # Add this import for cosine similarity
import re  # Add this import for regular expressions
device = "cuda" if torch.cuda.is_available() else "cpu"

q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
q_encoder = q_encoder.to(device)
q_encoder.eval()

p_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
p_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
p_encoder = p_encoder.to(device)
p_encoder.eval()

input_path = '/home/minhae/multihop-RAG2/ablation/gpqa/llama/cosine/subq.json'

with open(input_path, 'r') as f:
    data = json.load(f)

sim_total = 0

# def extract_numbers(text):
#     # Find all numbers (including decimals) in the text
#     numbers = re.findall(r'\d+\.\d+', text)
#     # Convert strings to floats
#     return [float(num) for num in numbers]

# total_scores = 0
# total_count = 0
# # Example usage:
# for item in data:
#     text = item.get('document distances', '')
#     if not text:
#         continue
#     numbers = extract_numbers(text)
#     for number in numbers:
#         total_scores += number
#     total_count += len(numbers)

# print(f"Average DPR score (dot product): {total_scores/total_count:.4f}")

for item in data:
    subquestion = item['subquestion']
    document = item['document']

    inputs = q_tokenizer(subquestion, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = p_tokenizer(document, return_tensors="pt", padding=True, truncation=True)
    outputs = {k: v.to(device) for k, v in outputs.items()}

    with torch.no_grad():
        # Debug: Print actual model output
        q_output = q_encoder(**inputs)
        p_output = p_encoder(**outputs)
        print("Query encoder output type:", type(q_output))
        print("Query encoder output:", q_output)
        print("Query encoder output dict:", q_output.__dict__)
        
        # Use pooler_output for now
        query_embedding = q_output.pooler_output.cpu()
        document_embedding = p_output.pooler_output.cpu()

        # Convert embeddings to tensors if they aren't already
        query_embedding = torch.tensor(query_embedding)
        document_embedding = torch.tensor(document_embedding)
        
        # Calculate dot product similarity (DPR's original scoring method)
        # Reshape tensors to 2D if needed (batch_size x embedding_dim)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if len(document_embedding.shape) == 1:
            document_embedding = document_embedding.unsqueeze(0)
            
        # Normalize the embeddings (L2 normalization)
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        document_embedding = F.normalize(document_embedding, p=2, dim=1)
        
        # Calculate dot product
        similarity = torch.sum(query_embedding * document_embedding, dim=1)
        
        score = similarity.item()
        sim_total += score

# Print individual scores
print("Individual DPR scores:")
print("\n")

print(f"SB Average DPR score (dot product): {sim_total/len(data):.4f}")