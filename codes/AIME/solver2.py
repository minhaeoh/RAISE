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

class MultiHopSolver:
    def __init__(self, mode="Step_RAG",ex_num=2, trigger=False, trigger_value=0.8, doc_num=10, query_mode="q1", wandb_run=None
    ):
        # Initialize environment
        env_path = "/home/minhae/multihop-RAG2/.env"
        load_dotenv(dotenv_path=env_path)
        
        # Get API keys from environment variables
        self.hf_token = os.getenv('HF_TOKEN')
        self.wandb_api_key = os.getenv('WANDB_API_KEY')
        self.wandb_entity = os.getenv('WANDB_ENTITY', 'minhae')  # Default to 'minhae' if not set

        # Initialize counters for tracking overall performance
        self._total_problems = 0
        self._correct_problems = 0

        self.mode = mode
        self.ex_num = ex_num
        self.trigger = trigger
        self.trigger_value = float(trigger_value)
        self.doc_num = doc_num
        self.query_mode = query_mode
        self.wandb_run = wandb_run
        self.model_id =  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

        # Initialize timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        # configuration details to directory name
        if self.mode == "zeroshot_COT":
            config_str = "zeroshot_COT"
        elif self.mode == "zeroshot_PS":
            config_str = "zeroshot_PS"
        elif self.mode == "zeroshot_RAG":
            config_str = "zeroshot_RAG"
            if self.trigger:
                config_str += f"_trigger_{self.trigger_value}"
            config_str += f"_doc{self.doc_num}"
        elif self.mode == "PD":
            config_str = "PD"
        else:
            config_str = self.query_mode
            config_str += f"_{self.mode}"
            if self.trigger:
                config_str += f"_trigger_{self.trigger_value}"
            config_str += f"_doc{self.doc_num}"
        self.config_str = config_str

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
        
        # Initialize DPR components
        if self.mode == "zeroshot_RAG" or self.mode == "Step_RAG" or self.mode == "Step_RAG_EX":
            print("Setting up Retrieval System...")
            self.setup_wiki_retriever()
        
        # Store wandb run if provided
        self.run = wandb_run

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
        """Generate text using the Mistral pipeline with output cleaning"""
        try:
            sampling_params = SamplingParams(
                max_tokens=max_length,
                temperature=temperature
            )
            response = self.llm.generate([prompt], sampling_params=sampling_params)
            response = response[0].outputs[0].text
            print("="*50)
            print(f"Prompt: \n{prompt}")
            print(f"Response: \n{response}")
            return response
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""

    def solve_with_zeroshot_COT(self, problem):
        prompt = f"""You are solving a math problem. Think step by step and show your reasoning clearly.  
        At the end, state your answer in the format: "The final answer is X."  
        Here, X must be the correct integer answer.

        Question: {problem['Problem']} 
        """
        try:
            response = self.generate_text(prompt, max_length=1500)
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""
        
    def solve_with_zeroshot_PS(self, problem):
        prompt = f"""
        Let's first understand the problem and devise a plan to solve the problem. 
        Then, let's carry out the plan and solve the problem step by step. 

        Use the following format exactly:
        <Plan>
        Step 1: [state what you will do]
        Step 2: [state next reasoning step]
        ... (as needed)

        <Solution>
        Step 1: [carry out Step 1]
        Step 2: [carry out Step 2]
        ... (as needed)

        "The final answer is X."  Here, X must be the correct integer answer.

        
        Question: {problem['Problem']}  
        """
        try:
            response = self.generate_text(prompt, max_length=1500)
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""

    def solve_with_zeroshot_RAG(self, problem, documents):
        prompt = f"""
        You are given a problem and a set of related documents.

        Use the information from the documents to help solve the problem step by step. If the documents contain useful facts, definitions, or formulas, apply them in your reasoning.

        Clearly explain your reasoning, and then finish your answer with "The final answer is X."  Here, X must be the correct integer answer.

        Documents: 
        {documents} 


        Question: {problem['Problem']}       
        """
        try:
            response = self.generate_text(prompt, max_length=1500)
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
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

            documents = []
            select_num = 0
            for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
                idx = int(idx)
                score = float(score)

                if idx >= len(self.passages):
                    continue
                
                passage = self.passages[idx]
                content = passage['text']
                doc = f"[Title: {passage['title']}]:\n{content}"
                if self.trigger and score<(self.trigger_value*100):
                    continue
                else:
                    documents.append(doc)
                    select_num += 1
                    if len(documents) >= num_results:
                        break
                
            if not documents:
                return "No relevant documents found.", 0, distances[0]
            else:
                return "\n".join(documents), select_num, distances[0]
            
        except Exception as e:
            print(f"Error in get_wiki_search_results: {e}")
            return "No relevant documents found.", 0, distances[0]

    def extract_answer(self, solution):
        for line in solution.split("\n"):
            if "the final answer is" in line.lower():
                line = line.replace("The final answer is", "the final answer is").strip()
                answer = line.split("the final answer is")[1].strip()
                # Try to find integer values in the answer
                numbers = re.findall(r'\b\d+\b', answer)
                if numbers:
                    return numbers[0]
        return "None"

    def generate_query(self, subquestion, query_mode, query):
        if query_mode == "subq":
            return subquestion
        elif query_mode == "q0" :
            return query
        elif query_mode == "q4" :
            prompt = f"""
            You are given a search query.

            Briefly explain the core concept or relationship described by the query in 2–3 sentences. Focus only on the essential scientific or mathematical idea.

            Search Query: {query}
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
            You are an expert at Mathematics. You are given a math problem. 
            Your task is to generate a realistic search query that would help someone find the mathematical principles, theorems, or techniques involved in solving the problem. 
            The search query should reflect how a student might search online to get help solving the problem—not just name a concept, but frame it in a useful and problem-relevant way.

            Here are a few examples:

            Question: A triangle has angles in the ratio 2:3:4. What is the measure of the largest angle?
            Search Query: how to find triangle angles given ratio of angles

            Question: Find the number of integers between 1 and 100 that are divisible by 2 or 5.
            Search Query: how to count numbers divisible by multiple numbers using inclusion exclusion

            Question: Solve for x: 2x^2 - 5x + 2 = 0
            Search Query: how to solve quadratic equation by factoring or using formula

            Question: A fair coin is flipped 5 times. What is the probability of getting exactly 3 heads?
            Search Query: binomial probability formula for getting k heads in n coin tosses

            Question: A function f(x) satisfies f(x+y) = f(x)f(y) for all real numbers x, y, and f(0) = 1. What is f(x)?
            Search Query: how to solve functional equations of the form f(x+y)=f(x)f(y)

            Question: {subquestion}
            Search Query:  
            """
            response = self.generate_text(prompt, max_length=100)
            if response == None:
                return ""
            else:
                query = response
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Question:"):
                        return "\n".join(response.split("\n")[:i])
            return response
                        

        elif query_mode == "q3":
            prompt = f"""
            You are an expert at Mathematics. You are given a math problem. 
            Your task is to generate a realistic search query that would help someone find the mathematical principles, theorems, or techniques involved in solving the problem. 
            The search query should reflect how a student might search online to get help solving the problem—not just name a concept, but frame it in a useful and problem-relevant way.

            Here are a few examples:

            Question: A triangle has angles in the ratio 2:3:4. What is the measure of the largest angle?
            Search Query: how to find triangle angles given ratio of angles

            Question: Find the number of integers between 1 and 100 that are divisible by 2 or 5.
            Search Query: how to count numbers divisible by multiple numbers using inclusion exclusion

            Question: Solve for x: 2x^2 - 5x + 2 = 0
            Search Query: how to solve quadratic equation by factoring or using formula

            Question: A fair coin is flipped 5 times. What is the probability of getting exactly 3 heads?
            Search Query: binomial probability formula for getting k heads in n coin tosses

            Question: A function f(x) satisfies f(x+y) = f(x)f(y) for all real numbers x, y, and f(0) = 1. What is f(x)?
            Search Query: how to solve functional equations of the form f(x+y)=f(x)f(y)

            Question: {subquestion}
            Search Query:  
            """
            response = self.generate_text(prompt, max_length=100)
            if response == None:
                return ""
            else:
                query = response
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Question:"):
                        query =  "\n".join(response.split("\n")[:i])
                        break
                
                prompt = f"""
                You are given a search query.

                Briefly explain the core concept or relationship described by the query in 2–3 sentences. Focus only on the essential scientific or mathematical idea.

                Search Query: {query}
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
            
        elif query_mode == "step-back":
            prompt = f"""
            You are an expert at Mathematics. You are given a math problem. 
            Your task is to extract the mathematical concepts, theorems, or problem-solving principles involved in solving the problem. 
            Here are a few examples:

            Question: A triangle has angles in the ratio 2:3:4. What is the measure of the largest angle?
            Principles Involved: Sum of angles in a triangle, Ratio reasoning, Basic algebra

            Question: Find the number of integers between 1 and 100 that are divisible by 2 or 5.
            Principles Involved: Inclusion-exclusion principle, Divisibility rules

            Question: Solve for x: 2x^2 - 5x + 2 = 0
            Principles Involved: Quadratic equation, Factoring method or quadratic formula

            Question: A fair coin is flipped 5 times. What is the probability of getting exactly 3 heads?
            Principles Involved: Binomial probability, Combinatorics

            Question: A function f(x) satisfies f(x+y) = f(x)f(y) for all real numbers x, y, and f(0) = 1. What is f(x)?
            Principles Involved: Functional equations, Exponential functions

            Now extract the principles involved in this question.

            Question: {subquestion}
            Principles Involved:
            """
            response = self.generate_text(prompt, max_length=30)
            if response == None:
                return ""
            else:
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Question:"):
                        return "\n".join(response.split("\n")[:i])
                return response
        elif query_mode == "step-hyde":
            prompt = f"""
            You are an expert at Mathematics. You are given a math problem. 
            Your task is to extract the mathematical concepts, theorems, or problem-solving principles involved in solving the problem. 
            Here are a few examples:

            Question: A triangle has angles in the ratio 2:3:4. What is the measure of the largest angle?
            Principles Involved: Sum of angles in a triangle, Ratio reasoning, Basic algebra

            Question: Find the number of integers between 1 and 100 that are divisible by 2 or 5.
            Principles Involved: Inclusion-exclusion principle, Divisibility rules

            Question: Solve for x: 2x^2 - 5x + 2 = 0
            Principles Involved: Quadratic equation, Factoring method or quadratic formula

            Question: A fair coin is flipped 5 times. What is the probability of getting exactly 3 heads?
            Principles Involved: Binomial probability, Combinatorics

            Question: A function f(x) satisfies f(x+y) = f(x)f(y) for all real numbers x, y, and f(0) = 1. What is f(x)?
            Principles Involved: Functional equations, Exponential functions

            Now extract the principles involved in this question.

            Question: {subquestion}
            Principles Involved:
            """
            response = self.generate_text(prompt, max_length=30)
            prompt = f"""
            Generate a paragraph that explains the principle.

            Principles: {response}"""
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
            raise ValueError(f"Invalid query mode: {query_mode}. Query mode should be one of subq, q0, q1, q2, q3, q4, hyde, step-back")


    def generate_subquestions(self, problem):
        prompt = f"""
        You are given a math problem. Your task is to break it down into a series of essential subquestions that reflect the full reasoning process needed to solve the problem.

        Guidelines:
        Each subquestion should correspond to a clear reasoning step, concept, or calculation.
        Subquestions may include specific numbers or variables from the original question to make the reasoning concrete.
        Each search query should reflect general scientific or mathematical knowledge needed to answer the subquestion. Do not include numeric values or overly specific phrasing from the original question in the search query.
        The goal is to simulate how someone would think step-by-step through the problem and search for relevant background knowledge at each stage.

        STRICT FORMAT REQUIREMENTS:
        1. For each subquestion, you MUST provide exactly two parts in this order:
        - The subquestion
        - Search query for that subquestion

        2. Use EXACTLY this format for each subquestion:
        Subquestion 1: [Write a concrete, reasoning-based subquestion—may include values or variables from the problem]
        Search Query for Subquestion 1: [Write a general search query someone might realistically use to learn how to answer this subquestion]

        Question: {problem['Problem']}
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


    def solve_subquestion_base(self, problem, step_num, subquestions, subsolutions):
        prompt = f"""
        You are solving a math problem. Problem is decomposed into several subquestions. You will be given:
            1. The original math problem
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve

        Your task is to solve the current subquestion.

        Question: {problem['Problem']}
        """
        if step_num > 1:
            prompt += "Previous subquestions and their solutions:"
            for i in range(step_num - 1):
                prompt += f"""
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        Current subquestion to solve:
        
        Subquestion {step_num}: {subquestions[step_num-1]}
        """

        try:
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion"):
                    return "\n".join(response.split("\n")[:i])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""


    def solve_subquestion_with_docs(self, problem, step_num, subquestions, subsolutions, documents):
        prompt = f"""
        You are solving a math problem. Problem is decomposed into several subquestions. You will be given:
            1. The original math problem
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Documents that are relevant to the current subquestion

        Your task is to solve the current subquestion.

        Documents:
        {documents}


        Question: {problem['Problem']}
        """
        if step_num > 1:
            prompt += "Previous subquestions and their solutions:"
            for i in range(step_num - 1):
                prompt += f"""
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        Current subquestion to solve:
        
        Subquestion {step_num}: {subquestions[step_num-1]}
        """
        

        try:
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion"):
                    return "\n".join(response.split("\n")[:i])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""

    def solve_subquestion_with_ex(self, problem, step_num, subquestions, subsolutions, examples):
        prompt = f"""
        You are solving a math problem. Problem is decomposed into several subquestions. You will be given:
            1. The original math problem
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Examples that are relevant to the current subquestion

        Your task is to solve the current subquestion.


        Relevant Examples:
        {examples}


        Question: {problem['Problem']}
        """
        if step_num > 1:
            prompt += "Previous subquestions and their solutions:"
            for i in range(step_num - 1):
                prompt += f"""
                Previous subquestion {i+1}: {subquestions[i]}
                Previous subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        Current subquestion to solve:
        
        Subquestion {step_num}: {subquestions[step_num-1]}
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion"):
                    return "\n".join(response.split("\n")[:i])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""
        return solution

    def generate_examples(self, step_num, problem, documents, subquestions):
        prompt = f"""
        You are a example generator. You will be given the original problem, a subquestion, and a knowledge document.

        Your task is:
        - Generate {self.ex_num} example questions with full solutions based on the knowledge document.
        - The questions should be similar to the subquestion.
        - The questions should help understand the concept mentioned in the subquestion.
        - Do not solve the original question.
        - Do not mention or refer to the document in the questions or solutions. The outputs should be self-contained.
        - Each solution should be accurate, clear, and informative.

        Use EXACTLY this format for each example question and solution:
        Question 1: [Write the question here]
        Solution 1: [Write the full solution here]

        Knowledge Documents: 
        {documents}

        Original Question:
        Question: {problem['Problem']} 

        Subquestion {step_num}: {subquestions[step_num-1]}
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            
            # Initialize variables to track the start and end indices
            start_idx = None
            end_idx = None

            # Iterate over the response to find the first and last question-solution pairs
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question") and start_idx is None:
                    start_idx = i
                if line.startswith("Solution"):
                    end_idx = i

            # Slice the response to include only the relevant part
            if start_idx is not None and end_idx is not None:
                sliced_response = "\n".join(response.split("\n")[start_idx:end_idx+1])
            else:
                sliced_response = ""

            return sliced_response

        except Exception as e:
            print(f"Error in generate_examples: {e}")
            return ""
    

    def generate_final_answer(self, problem, subquestions, subsolutions):
        prompt = f"""
        You are solving a math problem. Problem is decomposed into several subquestions. Each subquestion has already been solved. Your task is to carefully read the original problem and the several subquestion solutions, then use them to determine the final answer. Think step by step and then finish your answer with "The final answer is X."  Here, X must be the correct integer answer.

        Original Question:
        Question: {problem['Problem']}

        Subquestions and Solutions:
        """
        for i, (subquestion, subsolution) in enumerate(zip(subquestions, subsolutions)):
            prompt += f"""
            Subquestion {i+1}: {subquestion}
            Subquestion {i+1} Solution: {subsolution}
            """

        try:
            response = self.generate_text(prompt, max_length=1500)
            return response
        except Exception as e:
            print(f"Error in generate_final_answer: {e}")
            return ""

    def solve_problem(self, subject, prob_num, problem=None):
        base_dir = f"/home/minhae/multihop-RAG2/results/{subject}-0510/notrig_doc5/mistral-small-latest"
        
        self.output_dir = os.path.join(base_dir, f"{self.config_str}",self.timestamp)
        if prob_num == 1:
            os.makedirs(self.output_dir, exist_ok=True)
            # Initialize wandb if not already initialized
            if self.run is None and self.wandb_api_key:
                os.environ['WANDB_API_KEY'] = self.wandb_api_key
                self.run = wandb.init(
                    project=f"{subject}-notrig_doc5",
                    entity=self.wandb_entity,
                    name=f"{self.config_str}_{self.timestamp}",
                    config = {
                        "mode" : self.mode,
                        "ex_num" : self.ex_num,
                        "trigger" : self.trigger,
                        "trigger_value" : self.trigger_value,
                        "doc_num" : self.doc_num,
                        "subject" : subject
                    }
                )
                # Create output files
        detailed_file = os.path.join(self.output_dir, "detailed.txt")
        summary_file = os.path.join(self.output_dir, "summary.txt")

        def file_print(*args, **kwargs):
            with open(detailed_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)

        def summary_print(*args, **kwargs):
            with open(summary_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)

        if prob_num == 1:
            summary_print("Configuration Information:")
            summary_print("=" * 50)
            summary_print(f"Model ID: {self.model_id}")
            summary_print(f"Mode: {self.mode}")
            summary_print(f"Query Mode: {self.query_mode}")
            summary_print(f"Trigger: {self.trigger}")
            summary_print(f"Trigger Value: {self.trigger_value}")
            summary_print(f"Doc Num: {self.doc_num}")
            summary_print("=" * 50)
            summary_print("\nProblem-by-Problem Results:")
            summary_print(f"{'Problem':^10} | {'Result':^10} | {'Predicted':^10} | {'Actual':^10} | {'Running Acc':^12}")
            summary_print("-" * 60)


        for print_func in [file_print, summary_print]:
            print_func("=" * 50)
            print_func("\nProblem {}:".format(prob_num))
            print_func("Question: {}".format(problem['Problem']))
            correct_answer = problem['Answer']
            print_func("Correct Answer: {}".format(correct_answer))
            print_func("--------------------------------")

        
                
        # Add timing dictionary
        start_total = time.time()

        if self.mode == "zeroshot_COT":
            final_solution = self.solve_with_zeroshot_COT(problem)
            file_print(f"Solution: {final_solution}")

        elif self.mode == "zeroshot_PS":
            final_solution = self.solve_with_zeroshot_PS(problem)
            file_print(f"Solution: {final_solution}")

        elif self.mode == "zeroshot_RAG":
            query = problem['Problem']
            documents, select_num, distances = self.get_wiki_search_results(query, self.doc_num)
            if select_num == 0:
                final_solution = self.solve_with_zeroshot_COT(problem)
            else:
                final_solution = self.solve_with_zeroshot_RAG(problem, documents)
            file_print(f"Query: {query}")
            file_print(f"There are {select_num} documents")
            file_print(f"Distances: {distances}")
            if select_num>0:
                file_print(f"Documents: {documents}")
            file_print(f"Solution: {final_solution}")            
        
        else:
            file_print(f"Stage 1: Generate subquestions and principles")
            subquestions, queries = self.generate_subquestions(problem)
            file_print(f"There are {len(subquestions)} subquestions and principles")
            file_print(f"Subquestions: {subquestions}")
            file_print(f"Principles: {queries}")
            file_print("--------------------------------")
            
            solutions = []
            file_print(f"\n\nStage 2: Solve each subquestions")
            for i, (subquestion, query) in enumerate(zip(subquestions, queries)):
                step_num = i + 1
                file_print(f"Subquestion {step_num}: {subquestion}")
                file_print(f"Query {step_num}: {query}")
                if self.mode == "PD":
                    solution = self.solve_subquestion_base(problem, step_num, subquestions, solutions)
                    solutions.append(solution)
                    file_print(f"Solution: {solution}")
                    file_print("--------------------------------")

                elif self.mode == "Step_RAG":
                    query = self.generate_query(subquestion, self.query_mode, query)
                    if query == "":
                        solution = self.solve_subquestion_base(problem, step_num, subquestions, solutions)
                    else:
                        documents, select_num, distances = self.get_wiki_search_results(query, self.doc_num)
                        if select_num == 0 :
                            solution = self.solve_subquestion_base(problem, step_num, subquestions, solutions)
                        else:
                            solution = self.solve_subquestion_with_docs(problem, step_num, subquestions, solutions, documents)
                    solutions.append(solution)
                    if query=="":
                        file_print("Error: Query is empty")
                    else:
                        file_print(f"Query: {query}")
                        file_print(f"There are {select_num} documents")
                        file_print(f"Distances: {distances}")
                        if select_num>0:
                            file_print(f"Documents: {documents}")
                    file_print(f"Solution: {solution}")
                    file_print("--------------------------------")

                elif self.mode == "Step_RAG_EX":
                    query = self.generate_query(subquestion, self.query_mode, query)
                    if query == "":
                        solution = self.solve_subquestion_base(problem, step_num, subquestions, solutions)
                    else:
                        documents, select_num, distances = self.get_wiki_search_results(query, self.doc_num)
                        if select_num == 0:
                            solution = self.solve_subquestion_base(problem, step_num, subquestions, solutions)
                        else:
                            examples = self.generate_examples(step_num, problem, documents, subquestions)
                            solution = self.solve_subquestion_with_ex(problem, step_num, subquestions, solutions, examples)
                    solutions.append(solution)
                    
                    if query=="":
                        file_print("Error: Query is empty")
                    else:
                        file_print(f"Query: {query}")
                        file_print(f"There are {select_num} documents")
                        file_print(f"Distances: {distances}")
                        if select_num>0:
                            file_print(f"Documents: {documents}")
                            file_print(f"Examples: {examples}")
                    file_print(f"Solution: {solution}")
                    file_print("--------------------------------")

                else:
                    raise ValueError(f"Invalid mode: {self.mode}. Mode should be one of zeroshot_COT, zeroshot_PS, zeroshot_RAG, PD, Step_RAG, Step_RAG_EX")
            file_print("--------------------------------")
            file_print(f"\n\nStage 3: Generate final answer")
            final_solution = self.generate_final_answer(problem, subquestions, solutions)
            file_print(f"Final Solution: {final_solution}")
        
        predicted_answer = self.extract_answer(final_solution)
        if predicted_answer != "None":
            if int(predicted_answer) == int(correct_answer):
                if hasattr(self, '_correct_problems'):
                    self._correct_problems += 1
                    result = "Correct"
                else:
                    self._correct_problems = 1
                    result = "Correct"
            else:
                result = "Wrong"
        else: 
            result = "Wrong"
        
        if hasattr(self, '_total_problems'):
            self._total_problems += 1
        else:
            self._total_problems = 1
        
        self._current_accuracy = self._correct_problems / self._total_problems

        self.run.log({
            "total_problems": self._total_problems,
            "correct_problems": self._correct_problems,
            "current_accuracy": self._current_accuracy
        })
        if final_solution is not None:
            # Save detailed results to the detailed file
            summary_print(f"Final Solution: {final_solution}")
            summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {self._current_accuracy:^11.2f}%")
            file_print("\nFinal Results:")
            file_print(f"Final Solution: {final_solution}")
            file_print(f"Predicted Answer: {predicted_answer}")
            file_print(f"Actual Answer: {correct_answer}")
            file_print(f"Running Accuracy: {self._current_accuracy:.2%}")
        # Print timing statistics
        total_time = time.time() - start_total
        file_print(f"Total Time: {total_time:.2f} seconds")


                