import os
import re
import time
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DPRQuestionEncoderTokenizer, 
    DPRQuestionEncoder,
    DPRContextEncoderTokenizer,
    DPRContextEncoder
)
import faiss
import gzip
import csv
import wandb
import json
import numpy as np
import logging
import random
from mistralai import Mistral
from vllm import LLM, SamplingParams
from openai import OpenAI
import argparse
import gc


class MultiHopSolver:
    def __init__(self):
        # Initialize environment
        env_path = "/home/minhae/multihop-RAG2/.env"
        load_dotenv(dotenv_path=env_path)
        
        # Get API keys from environment variables
        self.hf_token = os.getenv('HF_TOKEN')

        
        self.model_id =  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        # Set device for CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Get number of available GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Number of available GPUs: {self.num_gpus}")

        # Initialize Mistral model
        try:
            print("Initializing Mistral model...")
            self.llm = LLM(
                    model="/data/minhae/models/mistral",
                    dtype="float16",  # 또는 "bfloat16"
                    trust_remote_code=True,
                    tensor_parallel_size=self.num_gpus  # Use available GPU count
                )

        except Exception as e:
            print(f"Error initializing Mistral model: {e}")
            raise

        # self.setup_wiki_retriever()


    def setup_wiki_retriever(self):
        """Setup FAISS retriever with DPR encoders"""
        print("Setting up DPR components...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            print("Loading DPR dataset with embeddings...")
            self.ds = load_dataset("facebook/wiki_dpr", 'psgs_w100.nq.exact',split='train')
            print("Finished loading DPR dataset")
            
            # 2. Get the FAISS index from the dataset
            self.index = self.ds.get_index("embeddings").faiss_index
            print("Finished loading FAISS index")
            

            
            # 3. Get the passages from the dataset
            self.passages = self.ds
            print(f"Loaded {len(self.passages)} passages")
            
            # 4. Initialize query encoder & tokenizer
            self.q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.q_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
            self.q_encoder = self.q_encoder.to(self.device)
            self.q_encoder.eval()

            self.p_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.p_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
            self.p_encoder = self.p_encoder.to(self.device)
            self.p_encoder.eval()
            
        except Exception as e:
            print(f"Error in setup_wiki_retriever: {e}")
            raise

    def generate_text(self, prompt, max_length=384, temperature=0.15):
        try:
            sampling_params = SamplingParams(
                max_tokens=max_length,
                temperature=temperature
            )
            response = self.llm.generate([prompt], sampling_params=sampling_params)
            response = response[0].outputs[0].text
            return response
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""


    
    def get_wiki_search_results(self, query, num_results=10):
        """Get search results using DPR and FAISS"""
        try:    
            # Convert query to tensor
            inputs = self.q_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                query_embedding = self.q_encoder(**inputs).pooler_output.cpu().numpy()
            
            search_num = num_results
            distances, indices = self.index.search(query_embedding, search_num)

            # Convert numpy.int64 indices to int
            documents = '\n'.join([self.passages[int(i)]['text'] for i in indices[0]])
                
            if not documents:
                return "No relevant documents found."
            return documents
            
        except Exception as e:
            print(f"Error in get_wiki_search_results: {e}")
            return "No relevant documents found."

    
    def generate_query(self, subquestion, query_mode, query):
        if query_mode == "subq":
            return subquestion
        elif query_mode == "q1" :
            prompt = f"""
            You are given a search query.

            Briefly explain the core concept or relationship described by the query in 2–3 sentences. Focus only on the essential scientific or mathematical idea.

            Search Query: {query}
            Explanation: 
            """
            response = self.generate_text(prompt, max_length=100)
            if response == None:
                return ""
            else:
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Search Query:"):
                        return "\n".join(response.split("\n")[:i])
                return response
        elif query_mode == "q2":
            prompt = f"""
            You are given a subquestion and a search query.
            The search query is a realistic phrase that someone might use to find knowledge or reasoning support to answer the subquestion.

            Your task is to anticipate what essential scientific or mathematical explanation the search result would contain, and write it concisely (2–3 sentences).
            Focus only on the core concept or principle that would help answer the subquestion.
            Avoid restating the subquestion, and do not include unrelated or overly general information.
            
            Subquestion: {subquestion}
            Search Query: {query}

            Explanation:
            """
            response = self.generate_text(prompt, max_length=100)
            if response == None:
                return ""
            else:
                query = response
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Subquestion:"):
                        return "\n".join(response.split("\n")[:i])
            return response
                        

        elif query_mode == "step-back":
            prompt = f"""
            You are an expert at Science. You are given a Science problem. Your task is to extract the Science concepts and principles involved in solving the problem.

            Question: {subquestion}
            Principles Involved:
            """
            response = self.generate_text(prompt, max_length=100)
            if response == None:
                return ""
            else:
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Question:"):
                        return "\n".join(response.split("\n")[:i])
                return response
        
        elif query_mode == "hyde":
            prompt = f"""
            Generate a paragraph that answers the question.

            Question: {subquestion}"""
            response = self.generate_text(prompt, max_length=100)
            if response == None:
                return ""
            else:
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Question:"):
                        return "\n".join(response.split("\n")[:i])
                return response
        else:
            raise ValueError(f"Invalid query mode: {query_mode}. Query mode should be one of subq, q0, q1, q2, q3 hyde, step-back")


    def generate_subquestions(self, problem):
        
        prompt = f"""
        You are given a multiple-choice question.
        Break this problem into essential subquestions that directly help solve the original problem.
        Each subquestion MUST also include its search query.
        Each search query should reflect scientific or mathematical knowledge needed to answer the subquestion. 

        STRICT FORMAT REQUIREMENTS:
        1. For each subquestion, you MUST provide exactly two parts in this order:
        - The subquestion
        - A search query for that subquestion

        2. Use EXACTLY this format for each subquestion:
        Subquestion 1: [your specific subquestion]
        Search Query for Subquestion 1: [Write a search query someone might realistically use to learn how to answer this subquestion]

        End your response with "End of generation" after you answer the instructions.

        Question: {problem['question']}
        Answer Choices: {problem['choices']['text']}
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            subquestions = []
            queries = []
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("Subquestion"):
                    subquestion = line.split(":")[1].strip()
                    subquestions.append(subquestion)
                elif line.startswith("Search Query for Subquestion"):
                    query = line.split(":")[1].strip()
                    queries.append(query)
            return subquestions, queries
                    
        except Exception as e:
            print(f"Error in generate_subquestions: {e}")
            return [], []
        
        return subquestions, queries

    def generate_search_query(self, question):
        prompt = f"""
        You are an expert in Science. You are given a science problem. 
        Your task is to write a realistic search query that would help someone find the scientific concepts, principles, or methods needed to solve the problem. 
        Each search query should reflect general scientific or mathematical knowledge needed to answer the subquestion. Do not include numeric values or overly specific phrasing from the original question in the search query.

        Question: {question}
        Search Query:
        """
        response = self.generate_text(prompt, max_length=100)
        if response == None:
            return ""
        else:
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question:"):
                    return "\n".join(response.split("\n")[:i])
            return response

    
    
    def evaluate_gpt(self,original_question,subquestion,subq_query,sb_query,hyde_query,raise_query):
        prompt = f"""You are given an original problem and a subquestion derived from it.
Along with these, you are provided with four different search queries generated by different methods:
(1) Subquestion only, (2) Step-back prompting, (3) HyDE prompting, and (4) RAISE.

Your task is to identify which query is most likely to retrieve logically relevant knowledge needed to answer the subquestion.
Specifically, select the query that best reflects the reasoning intent behind the subquestion and is most likely to lead to useful background information.

Inputs:

- Original Question: 
{original_question}  

- Subquestion: 
{subquestion}  

- Subq Query: 
{subq_query}

- Step-back Query: 
{sb_query}

- HyDE Query: 
{hyde_query}

- RAISE Query: 
{raise_query}


Please answer in the following format:

Best Query: [Subquestion / Step-back / HyDE / RAISE]  
Justification: [brief explanation why this query is the most suitable.]
Additional Explanation: [Explain why others are not suitable.]
"""


        response = self.openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an evaluator for model-generated answers."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
        return response.choices[0].message.content
    
    def evaluate_step_by_step(self,start_num,prob_num,problem):
        base_dir = "/home/minhae/multihop-RAG2/ablations/gpqa/query"
        os.makedirs(base_dir, exist_ok=True)

        evalaution_file = os.path.join(base_dir, f"{start_num}.txt")
        def file_print(*args, **kwargs):
            with open(evalaution_file, "a") as f:
                print(*args, **kwargs, file=f)

        file_print("="*50)
        file_print("\nProblem {}:".format(prob_num))
        file_print("Question: {}".format(problem['question']))
        file_print("Choices: {}".format(problem['choices']['text']))
        options = problem['choices']['label']
        correct_answer = problem['answerKey']
        file_print("Correct Answer: {}".format(correct_answer))
        file_print("--------------------------------")

        subquestions,queries = self.generate_subquestions(problem)
        for i,subquestion,query in zip(range(len(subquestions)),subquestions,queries):
            file_print(f"Subquestion {i+1}: \n{subquestion}")
            raise_query = self.generate_query(subquestion,"q2",query)
            subq_query = self.generate_query(subquestion,"subq",query)
            sb_query = self.generate_query(subquestion,"step-back",query)
            hyde_query = self.generate_query(subquestion,"hyde",query)
            file_print(f"Subq Query {i+1}: \n{subq_query}")
            file_print(f"Raise Query {i+1}: \n{raise_query}")
            file_print(f"Step-back Query {i+1}: \n{sb_query}")
            file_print(f"Hyde Query {i+1}: \n{hyde_query}")
            file_print("--------------------------------")
            subq_eval = self.evaluate_gpt(problem['question'],subquestion,subq_query,sb_query,hyde_query,raise_query)
            file_print(f"Query Evaluation: \n{subq_eval}")
            file_print("="*50)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--start_num", type=int, default=181, help="start number")
    args = args.parse_args()

    with open('/home/minhae/multihop-RAG2/codes/gpqa/gpqa_diamond_problems.json', 'r', encoding='utf-8') as f:
        ds = json.load(f)

    solver = MultiHopSolver()

    start_prob = args.start_num
    end_prob = start_prob + 29
    for prob_num in tqdm(range(start_prob, min(end_prob + 1, len(ds)))):
        problem = ds[prob_num-1]
        solver.evaluate_step_by_step(start_prob,prob_num,problem)
            # Clear GPU memory after each problem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()