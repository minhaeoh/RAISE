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

class MultiHopSolver:
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct", ex_num=2, generate_ex=True,  review_ex=True, zeroshot=False,  PD_baseline=False, zeroshot_RAG=False, trigger=False,trigger_value=0.8,filter=False,doc_num=10, wandb_run=None, PD_trigger=False
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
        self._subject_stats = {}
        
        self.model_id = model_id
        self.ex_num = ex_num
        #self.subject = subject
        self.generate_ex = generate_ex
        self.review_ex = review_ex
        self.zeroshot = zeroshot
        self.PD_baseline = PD_baseline
        self.zeroshot_RAG = zeroshot_RAG
        self.trigger = trigger
        self.trigger_value = float(trigger_value)
        self.filter = filter
        self.doc_num = doc_num
        self.PD_trigger = PD_trigger

        # Initialize timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Create output directory with all argument information

        # Add configuration details to directory name

        if self.zeroshot:
            config_str = "zeroshot"
        elif self.zeroshot_RAG:
            config_str = "zeroshot_RAG"
        elif self.PD_baseline:
            config_str = "PD_baseline"
        else:
            config_str = "q3"
            if not self.generate_ex:
                config_str += "_only_Doc"
            else:
                if self.review_ex:
                    config_str += "_R_Ex"
            if self.PD_trigger:
                config_str += "_PD_trigger"
            if self.trigger:
                config_str += f"_trigger_{self.trigger_value}"
            if self.filter:
                config_str += "_filter"
            if self.doc_num != 10:
                config_str += f"_{self.doc_num}doc"
            if self.ex_num != 4:
                config_str += f"_{self.ex_num}ex"
            
        self.config_str = config_str
        
        # Set device for CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize the Llama model 
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=self.hf_token
            )
            print("EOS token:", self.tokenizer.eos_token)
            print("EOS token id:", self.tokenizer.eos_token_id)
            print("EOS token ID type:", type(self.tokenizer.eos_token_id))

            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=self.hf_token
            )
            self.model.eval()
            print("Successfully initialized Llama model")
        except Exception as e:
            print(f"Error initializing Llama model: {e}")
            raise
        
        # Initialize DPR components
        if not self.PD_baseline:
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

    def generate_text(self, prompt, strip_prompt, max_length=384, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
        """Generate text using the Llama pipeline with output cleaning"""
        try:
            stop_phrases = [
                "Let me",
                "I hope",
                "Please",
                "Here's",
                "I'll",
                "I can",
                "Feel free",
                "Do you",
                "Note:",
                "Remember:",
                "Best regards",
                "Best,",
                "(Note:",
                "Thank you",
                "I am happy",
                "def ",
                "import ",
            ]
            bad_words_ids = [self.tokenizer.encode(phrase, add_special_tokens=False) for phrase in stop_phrases]
            
            # Encode the input with proper attention mask and truncation
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                bad_words_ids=bad_words_ids
            )
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = output_text[len(strip_prompt):].strip()
            response = re.sub(r"(\|\s*){3,}", "", response).strip()
            
            return response
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""

    def subquestion_trigger(self,problem):
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        "Check if problem decomposition is needed"
        base_prompt = f"""Question: {problem['question']}
        Options: {choices_text}

        You are given a question that may or may not require multiple reasoning steps to solve.
        Determine if the problem requires multi-step reasoning or can be solved in a single step.
        
        If it requires decomposition, output: "Decompose"
        If it can be solved directly, output: "Direct"
        Output only one word: either "Decompose" or "Direct". Do not explain your reasoning.
        """
        prompt = base_prompt+"\noutput:"
        response = self.generate_text(prompt,base_prompt, max_length=3)
        return response

    def generate_subquestions(self, problem):
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

#         base_prompt = f"""Question: {problem['question']}
# Options: {choices_text}

# Break this problem into essential subquestions that directly help solve the original problem.
# Each subquestion MUST include its solution and a search query.

# STRICT FORMAT REQUIREMENTS:
# 1. For each subquestion, you MUST provide exactly three parts in this order:
#    - The subquestion
#    - The solution to that subquestion
#    - A search query for that subquestion

# 2. Use EXACTLY this format for each subquestion:
# Subquestion 1: [your specific subquestion]
# Solution for Subquestion 1: [your specific solution]
# Search Query for Subquestion 1: [your specific search query]
# """
        base_prompt = f"""Question: {problem['question']}
        Options: {choices_text}

        Break this problem into essential subquestions that directly help solve the original problem.  
        For each subquestion, also identify the relevant **search category** (e.g., the academic subject or reasoning type needed to answer the subquestion, such as physics, algebra, biology, or logical deduction). Then, write a search query that would retrieve relevant knowledge or reasoning support. The query should reflect what someone would realistically search for to find useful information—not just a rewording of the subquestion.

        STRICT FORMAT REQUIREMENTS:  
        1. For each subquestion, provide **exactly three parts** in this order:  
        - The subquestion  
        - The search category (e.g., chemistry, math, logical reasoning — choose based on the knowledge area or reasoning method involved)  
        - The search query

        2. Use **EXACTLY** this format for each subquestion:
        Subquestion 1: [your specific subquestion]  
        Search Category for Subquestion 1: [your specific search category]  
        Search Query for Subquestion 1: [your specific search query]
        """
        prompt = base_prompt + "\n\nSubquestion 1: "

        # Get raw response and clean it
        raw_response = self.generate_text(prompt,base_prompt)
        
        # Split into lines and find where "Final Answer" appears
        lines = raw_response.split('\n')
        final_lines = []
        current_index = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("Subquestion") or line.strip().startswith("Search Category for Subquestion") or line.strip().startswith("Search Query for Subquestion"):
                current_index = i
            elif "final answer" in line.lower() or "Answer:" in line or "Answer :" in line:
                current_index = i+1
                break
            elif line.strip() == "":
                current_index = i
            else: break

        final_lines = lines[:current_index]  # Keep only lines before "Final Answer"
                
        # Join the lines back together
        cleaned_response = '\n'.join(final_lines)
        
        return cleaned_response


    def get_wiki_search_results(self,problem, subquestion, query, category, num_results=3, subject=None):
        """Get search results using DPR and FAISS"""
        prompt = f"""
        You are given a search query and its associated search category.  
        Your task is to provide a clear and informative explanation that directly answers or elaborates on the search query.  
        Focus only on the content directly relevant to the search query.

        End your response with "End of explanation."

        Search Category: {category}  
        Search Query: {query} 
        Explanation:
        """
        
        response = self.generate_text(prompt,prompt, max_length=100)
        lines = []
        for line in response.split("\n"):
            if "End of explanation".lower() in line.lower():
                line = line.lower().split("end of explanation")[0]
                lines.append(line)
                break
            else:
                lines.append(line)
        response = "\n".join(lines)

        print(f"Explanation: {response}")
        try:
            print(f"\nDebug: Starting search for query: '{query}'")
            
            # Encode query
            print("Debug: Encoding query...")
            inputs = self.q_tokenizer(response, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            # inputs = self.p_tokenizer(response, return_tensors="pt", padding=True, truncation=True)
            # inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                query_embedding = self.q_encoder(**inputs).pooler_output.cpu().numpy()
            print(f"Debug: Query embedding shape: {query_embedding.shape}")
            
            # Search in FAISS index
            search_num = num_results 
            print(f"Debug: Searching FAISS index for {search_num} results...")
            distances, indices = self.index.search(query_embedding, search_num)
            print(f"Debug: Found {len(indices[0])} results")
            print(f"Debug: Distances: {distances[0]}")
            print(f"Debug: Indices: {indices[0]}")
            
            documents = []
            select_num = 0
            for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
                print(f"\nDebug: Processing result {i+1}")
                print(f"Debug: Score: {score}, Index: {idx}")
                
                # Convert numpy types to Python native types
                idx = int(idx)
                score = float(score)
                #score = score/160*100
                
                if idx >= len(self.passages):
                    print(f"Debug: Index {idx} out of range (max: {len(self.passages)})")
                    continue
                    
                passage = self.passages[idx]
                content = passage['text']
                title = passage['title']
                print(f"Debug: Retrieved passage title: {title}")
                
                doc = f"Document {i+1} (score: {score:.2f}) [Title: {title}]:\n{content}"

                if self.trigger and score<(self.trigger_value*100):
                    print(f"original score: {score}")
                    continue
                else:
                    print(f"passed score: {score}")
                    documents.append(content)
                    select_num += 1
                    if len(documents) >= num_results:
                        break
            if not documents:
                print("Debug: No relevant documents found after processing all results")
                return "No relevant documents found.", 0, distances[0], response

            return "\n\n".join(documents), select_num, distances[0], response
            
        except Exception as e:
            print(f"Error in wiki search: {e}")
            import traceback
            traceback.print_exc()
            return "No relevant documents found.", 0, distances[0], response

    def get_wiki_search_zeroshot(self, problem, num_results=3):
        """Get search results using DPR and FAISS"""
        query = problem['question']
        try:
            print(f"\nDebug: Starting search for query: '{query}'")
            
            # Encode query
            print("Debug: Encoding query...")
            inputs = self.q_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                query_embedding = self.q_encoder(**inputs).pooler_output.cpu().numpy()
            print(f"Debug: Query embedding shape: {query_embedding.shape}")
            
            # Search in FAISS index
            search_num = num_results 
            print(f"Debug: Searching FAISS index for {search_num} results...")
            distances, indices = self.index.search(query_embedding, search_num)
            print(f"Debug: Found {len(indices[0])} results")
            print(f"Debug: Distances: {distances[0]}")
            print(f"Debug: Indices: {indices[0]}")
            
            documents = []
            select_num = 0
            for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
                print(f"\nDebug: Processing result {i+1}")
                print(f"Debug: Score: {score}, Index: {idx}")
                
                # Convert numpy types to Python native types
                idx = int(idx)
                score = float(score)
                
                if idx >= len(self.passages):
                    print(f"Debug: Index {idx} out of range (max: {len(self.passages)})")
                    continue
                
                if self.trigger and score<(self.trigger_value*100):
                    print(f"original score: {score}")
                    continue
                else:
                    print(f"passed score: {score}")
                    passage = self.passages[idx]
                    content = passage['text']
                    title = passage['title']
                    documents.append(content)
                    select_num += 1
                    if len(documents) >= num_results:
                        break
                       
            if not documents:
                print("Debug: No relevant documents found after processing all results")
                return "No relevant documents found.",select_num
            
            return "\n\n".join(documents), select_num
            
        except Exception as e:
            print(f"Error in wiki search: {e}")
            import traceback
            traceback.print_exc()
            return "No relevant documents found."

    def review_example(self, subquestion, example):
        """Review if a generated example is relevant and helpful for understanding the subquestion"""
        base_prompt = f"""You are an example reviewer. Your task is to determine if the given example is relevant and helpful for understanding the subquestion.

Subquestion: {subquestion}

Example:
{example}

Please analyze if this example is relevant and helpful. Consider:
1. Does the example directly relate to the concept in the subquestion?
2. Is the example clear and well-structured?
3. Does the example help in understanding the reasoning needed for the subquestion?
4. Is the solution detailed and instructive?

Respond with only "RELEVANT" or "NOT_RELEVANT"."""
        prompt = base_prompt + "\nThis example is"


        try:
            response = self.generate_text(prompt, base_prompt, max_length=10)
            result = response.strip().upper()
            return result == "RELEVANT"
        except Exception as e:
            print(f"Error in example review: {e}")
            return False


    def generate_examples(self, problem, documents, subquestions, step_num):
        """Step 2: Generate example questions for each subquestion"""
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        
        num_to_generate = self.ex_num * 2 if self.review_ex else self.ex_num

        base_prompt = f"""You are a example generator. You will be given the original question, a subquestion, and a knowledge document.

        Your task is:
        - Generate {num_to_generate} example questions with full solutions based on the knowledge document.
        - The questions should be similar to the subquestion.
        - The questions should help understand the concept mentioned in the subquestion.
        - Do not solve the original question.
        - Do not use multiple choice or ask the user to select an option. Make each question open-ended or short-answer style.
        - Do not mention or refer to the document in the questions or solutions. The outputs should be self-contained.
        - Each solution should be accurate, clear, and informative.

        Format:
        Question 1: [Write the question here]
        Solution 1: [Write the full solution here]

        ...

        Here is the input:
        Question: {problem['question']} | Options: {choices_text}
        Stepquestion {step_num}: {subquestions[step_num-1]}
        Doc: {documents}"""
        prompt = base_prompt + "\n\nQuestion 1:"

        response = self.generate_text(prompt,base_prompt)
        
        # Parse the generated examples
        raw_examples = re.split(r"```+", response)[0]
        examples_list = []
        current_example = ""
        
        # Split the response into individual examples
        for line in raw_examples.split('\n'):
            if line.strip().startswith("Solution 4:"):
                current_example += '\n' + line
                examples_list.append(current_example.strip())
                break  # 이후 내용 무시
            elif line.strip().startswith("Question"):
                if current_example:
                    examples_list.append(current_example.strip())
                current_example = line
            elif line.strip():
                current_example += '\n' + line

        if current_example and not any("Solution 4:" in ex for ex in examples_list):
            examples_list.append(current_example.strip())
        
        if self.review_ex:
            print(f"\nReviewing {len(examples_list)} examples...")
            # Review and filter examples
            relevant_examples = []
            for i, example in enumerate(examples_list, 1):
                print(f"Reviewing example {i}/{len(examples_list)}...")
                if self.review_example(subquestions[step_num-1], example):
                    relevant_examples.append(example)
                    if len(relevant_examples) >= self.ex_num:
                        break
            if len(relevant_examples)>0:
                print(f"Found {len(relevant_examples)} relevant examples")
                return "\n\n".join(relevant_examples)
            else:
                print(f"Found {len(relevant_examples)} relevant examples")
                return "None"
        else:
            #print("\nReview example error")
            # Return all examples without review
            return "\n\n".join(examples_list)

    def solve_subquestion(self, problem, subquestions, examples, subsolutions, step_num, subject):
        """Steps 3-5: Solve each subquestion"""
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        # base_prompt = f"""You are solving a complex problem step by step. You will be given:

        #     1. The original question
        #     2. Previous subquestions and their solutions (if any)
        #     3. The current subquestion to solve
        #     4. Example problems with solutions that demonstrate how to solve the current subquestion

        #     Your task:
        #     - Carefully read the original question, any previous solutions, and the current subquestion
        #     - Use the reasoning and methods shown in the example problems to solve the current subquestion
        #     - Use your existing knowledge to solve the subquestion
        #     - Your solution should be detailed and logically structured

        #     Question: {problem['question']} | Options: {choices_text}"""
        base_prompt = f"""You are solving a multiple-choice question about {subject}. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choicequestion
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Example problems with solutions that demonstrate how to solve the current subquestion

            Your task:
            - Carefully read the original question, any previous subquestion and their solutions, and the current subquestion
            - Use the reasoning and methods shown in the example problems to solve the current subquestion
            - Use your existing knowledge to solve the subquestion
            - Your solution should be detailed and logically structured

            Question: {problem['question']} | Options: {choices_text}"""
        
        # Add previous subquestions and their solutions if they exist
        if step_num > 1:
            for i in range(step_num - 1):
                base_prompt += f"""
                
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}"""
        
        # Add current subquestion and examples
        base_prompt += f"""
                Now solve the current subquestion. Use the examples and your knowledge to solve the subquestion.
                
                Example Problems:
                {examples}
                
                Subquestion {step_num}: {subquestions[step_num-1]}
                Now write the Step {step_num} Solution.
                """
        prompt = base_prompt + f"\n\nStep {step_num} Solution:"

        response = self.generate_text(prompt,base_prompt)
        solutions = []

        for line in response.split("\n"):
            if 'the final answer is' in line.lower():
                solutions.append(line)
                break
            else: 
                solutions.append(line)

        return "\n".join(solutions)

    def solve_subquestion_with_docs(self, problem, subquestions, documents, subsolutions, step_num, subject):
        """Steps 3-5: Solve each subquestion using retrieved documents directly"""
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""You are solving a multiple-choice question about {subject}. Question is decomposed into several subquestions. You will be given:

            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Retrieved documents containing relevant information

            Your task:
            - Carefully read the original question, any previous subquestion and their solutions, and the current subquestion
            - Use the information from the retrieved documents to solve the current subquestion
            - Your solution should be detailed and logically structured

            Question: {problem['question']} | Options: {choices_text}"""
        
        # Add previous subquestions and their solutions if they exist
        if step_num > 1:
            for i in range(step_num - 1):
                base_prompt += f"""
                
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}"""
        
        # Add current subquestion and documents
        base_prompt += f"""                
                Now solve the current subquestion. Use the retrieved documents to solve the subquestion.

                Retrieved Documents:
                {documents}
                
                Subquestion {step_num}: {subquestions[step_num-1]}
                
                Now write the Step {step_num} Solution:
                """
        response = self.generate_text(base_prompt,base_prompt)
        responses = response.split("\n")
        index = 0
        for i, line in enumerate(responses):
            if 'final answer' in line.lower():
                index = min(i+3, len(responses))
                break
            else: index = i

        return "\n".join(responses[:index])

    def solve_subquestion_base(self, problem, subquestions, subsolutions, step_num, subject):
        """Steps 3-5: Solve each subquestion step by step with zeroshot COT"""
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""You are solving a multiple-choice question about {subject}. Question is decomposed into several subquestions. You will be given:

            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve

            Your task:
            - Carefully read the original question, any previous subquestion and their solutions, and the current subquestion
            - Your solution should be detailed and logically structured

            Question: {problem['question']} | Options: {choices_text}"""
        
        # Add previous subquestions and their solutions if they exist
        if step_num > 1:
            for i in range(step_num - 1):
                base_prompt += f"""
                
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}"""
        
        # Add current subquestion and documents
        base_prompt += f"""
                
                Subquestion {step_num}: {subquestions[step_num-1]}
                
                Now write the Step {step_num} Solution:
                """
        response = self.generate_text(base_prompt,base_prompt)
        responses = response.split("\n")
        index = 0
        for i, line in enumerate(responses):
            if 'final answer' in line.lower():
                index = min(i+3, len(responses))
                break
            else: index = i

        return "\n".join(responses[:index])


    def generate_final_answer(self, problem, subquestions, subsolutions, subject):
        """Step 6: Generate final answer using all subquestion solutions"""
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)
        # Instructions:
        #     - You will be given the original question and the several subquestion solutions.
        #     - Read the original question carefully.
        #     - Review each subquestion and its solution.
        #     - Use the information and reasoning from all four subsolutions to logically determine the final answer.
        #     - Make sure your answer is consistent with the subsolutions.
        #     - Your output should include a brief justification followed by the final answer.

        base_prompt = f"""
            You are solving a multiple-choice question about {subject}. Question is decomposed into several subquestions. Each subquestion has already been solved. Your task is to carefully read the original question and the several subquestion solutions, then use them to determine the final answer. Think step by step and then finish your answer with "The answer is (X)" where X is the correct letter choice.

            Output format:

            Final Reasoning:
            [Explain how the subsolutions lead to the final answer.]

            Final Answer:
            [The final answer is (X), where X is the correct letter choice.]

            Here is the input:

            Question: {problem['question']}
            Options: {choices_text}

            """
        
        # Add all subquestions and their solutions
        for i, (subq, sol) in enumerate(zip(subquestions, subsolutions), 1):
            base_prompt += f"\nSubquestion {i}: {subq}\nSubquestion {i} Solution: {sol}\n"
        prompt = base_prompt + "\n\nFinal Reasoning:"

        response = self.generate_text(prompt,base_prompt, max_length=500)
        print(response)
        responses = response.split("\n")
        index = 0
        for i, line in enumerate(responses):
            if 'the final answer is' in line.lower() or 'the answer is' in line.lower():
                index = min(i+3, len(responses))
                break
            else: index = i+1
        
        return "\n".join(responses[:index])

    def parse_subquestions_and_queries(self, result_text):
        """Parse subquestions, search queries, and search categories from the generated text"""
        subquestions = []
        search_queries = []
        search_categories = []
        
        # Split the text into lines and process each line
        lines = result_text.strip().split('\n')
        
        current_subq = None
        current_category = None
        current_query = None
        current_subq_num = None
        
        #print("Processing lines:")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            #print(f"Processing line: {line}")
            
            # Extract subquestion number if present
            subq_match = re.match(r"Subquestion (\d+):", line)
            if subq_match:
                # If we have a previous incomplete set with subquestion and query
                if current_subq and current_query and current_category and current_subq_num:
                    print(f"Adding set with missing category or query: {current_subq} | {current_category} | {current_query}")
                    subquestions.append(current_subq)
                    search_categories.append(current_category)
                    search_queries.append(current_query)
                
                current_subq_num = int(subq_match.group(1))
                current_subq = line.split(":", 1)[1].strip()
                current_query = None
                current_category = None
                
            # Process search category if it matches current subquestion number
            elif line.startswith("Search Category for Subquestion"):
                category_match = re.match(r"Search Category for Subquestion (\d+):", line)
                if category_match and int(category_match.group(1)) == current_subq_num:
                    current_category = line.split(":", 1)[1].strip()
                
            # Process search query if it matches current subquestion number
            elif line.startswith("Search Query for Subquestion"):
                query_match = re.match(r"Search Query for Subquestion (\d+):", line)
                if query_match and int(query_match.group(1)) == current_subq_num:
                    current_query = line.split(":", 1)[1].strip()
                
                    # If we have both subquestion, category and query
                    if current_subq and current_query and current_category:
                        print(f"Adding complete set: {current_subq} | {current_category} | {current_query}")
                        subquestions.append(current_subq)
                        search_categories.append(current_category)
                        search_queries.append(current_query)
                    
                        # Reset for next set
                        current_subq = None
                        current_query = None
                        current_category = None
                        current_subq_num = None
        
        # Handle the last set if incomplete
        if current_subq and current_query and current_category:
            print(f"Adding final set: {current_subq} | {current_category} | {current_query}")
            subquestions.append(current_subq)
            search_categories.append(current_category)
            search_queries.append(current_query)

        print(f"\nFinal counts - Subquestions: {len(subquestions)}, Categories: {len(search_categories)}, Queries: {len(search_queries)}")
        
        # Verify that we have equal numbers of subquestions, categories and queries
        if not (len(subquestions) == len(search_queries) == len(search_categories)):
            raise ValueError("Parsing error: Mismatch in number of subquestions, categories and queries")
        
        return subquestions, search_queries, search_categories

    def solve_with_zeroshot(self, problem, subject):
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""You are solving a multiple choice question about {subject}.
Question: {problem['question']} | Options: {choices_text}

Output format:

Final Reasoning:
[Explain how the subsolutions lead to the final answer.]

Final Answer:
[The final answer is (X), where X is the correct letter choice.]

"""
        prompt = base_prompt + "\n\nFinal Reasoning:"

        try:
            response = self.generate_text(prompt,base_prompt,max_length=500)
            print(response)
            responses = response.split("\n")
            index = 0
            for i, line in enumerate(responses):
                if 'the final answer is' in line.lower() or 'the answer is' in line.lower():
                    index = min(i+3, len(responses))
                    break
                else: index = i

            return "\n".join(responses[:index])
        except Exception as e:
            print(f"Error in zeroshot solution: {e}")
            return None

    def solve_with_zeroshot_RAG(self, problem, documents):
        """Solve the problem directly without generating subquestions but using RAG"""
        options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
        options = options[:len(problem['choices'])]
        formatted_choices = [f"{i}: {choice}" for i, choice in zip(options, problem['choices'])]
        choices_text = "\n".join(formatted_choices)

        base_prompt = f"""You are solving a complex problem step by step. You will be given:

            1. Question to solve
            2. Relevant documents

            Your task:
            - Carefully read the question and the documents
            - Solve the question using the documents

            Solve this multiple choice question step by step.
Question: {problem['question']} | Options: {choices_text}

Relevant Documents:
{documents}

Output format:

Final Reasoning:
[Explain how the subsolutions lead to the final answer.]

Final Answer:
[The final answer is (X), where X is the correct letter choice.]

"""
        prompt = base_prompt + "\n\nFinal Reasoning:"

        try:
            response = self.generate_text(prompt,base_prompt)
            print(response)
            responses = response.split("\n")
            index = 0
            for i, line in enumerate(responses):
                if 'the final answer is' in line.lower() or 'the answer is' in line.lower():
                    index = min(i+3, len(responses))
                    break
                else: index = i

            return "\n".join(responses[:min(index+1, len(responses))])
        except Exception as e:
            print(f"Error in zeroshot solution: {e}")
            return None

    def solve_problem(self, subject, prob_num, problem=None):
        base_dir = f"/home/minhae/multihop-RAG2/results/mmlu-new/{subject}"
        self.output_dir = os.path.join(base_dir, f"{self.config_str}",self.timestamp)
        if prob_num == 1:
            os.makedirs(self.output_dir, exist_ok=True)
            # Initialize wandb if not already initialized
            if self.run is None and self.wandb_api_key:
                os.environ['WANDB_API_KEY'] = self.wandb_api_key
                self.run = wandb.init(
                    project=f"final-mmlu-stem",
                    entity=self.wandb_entity,
                    name=f"{self.config_str}_{self.timestamp}",
                    config={
                        "model_id": self.model_id,
                        "ex_num": self.ex_num,
                        "generate_ex": self.generate_ex,
                        "review_ex": self.review_ex,
                        "zeroshot": self.zeroshot,
                        "PD_baseline": self.PD_baseline,
                        "zeroshot_RAG": self.zeroshot_RAG,
                        "trigger": self.trigger,
                        "trigger_value": self.trigger_value,
                        "filter": self.filter,
                        "doc_num": self.doc_num,
                        "PD_trigger": self.PD_trigger,
                        "subject": subject
                    }
                )
        """Main method to solve the problem end-to-end"""
        # Add timing dictionary
        timing_stats = {}
        start_total = time.time()

        # Initialize subject-specific counters if they don't exist
        if subject not in self._subject_stats:
            self._subject_stats[subject] = {
                'total_problems': 0,
                'correct_problems': 0
            }

        # Create output files
        detailed_file = os.path.join(self.output_dir, "detailed.txt")
        summary_file = os.path.join(self.output_dir, "summary.txt")

        def file_print(*args, **kwargs):
            with open(detailed_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)

        def summary_print(*args, **kwargs):
            with open(summary_file, 'a', encoding='utf-8') as f:
                print(*args, **kwargs, file=f)
                
        try:
            # Print configuration information only for the first problem
            if prob_num == 1:
                summary_print("Configuration Information:")
                summary_print("=" * 50)
                summary_print(f"Model ID: {self.model_id}")
                summary_print(f"Subject: {subject}")
                summary_print(f"Number of Examples: {self.ex_num}")
                summary_print(f"Number of Documents: {self.doc_num}")
                summary_print(f"Generate Examples: {self.generate_ex}")
                summary_print(f"Review Examples: {self.review_ex}")
                summary_print(f"PD Baseline: {self.PD_baseline}")
                summary_print(f"Zero-shot: {self.zeroshot}")
                summary_print(f"Zero-shot RAG: {self.zeroshot_RAG}")
                summary_print(f"Trigger: {self.trigger}")
                summary_print(f"Trigger Value: {self.trigger_value}")
                summary_print(f"Filter: {self.filter}")
                summary_print(f"Device: {self.device}")
                summary_print(f"Timestamp: {self.timestamp}")
                summary_print("=" * 50)
                summary_print("\nProblem-by-Problem Results:")
                summary_print(f"{'Problem':^10} | {'Result':^10} | {'Predicted':^10} | {'Actual':^10} | {'Running Acc':^12}")
                summary_print("-" * 60)

            # Print problem and options to both files
            for print_func in [file_print, summary_print]:
                print_func("\nProblem {}:".format(prob_num))
                print_func("Question: {}".format(problem['question']))
                print_func("Choices: {}".format(problem['choices']))
                options = ['A', 'B', 'C', 'D','E','F','G','H','I','J']
                correct_answer = options[int(problem['answer'])]
                print_func("Correct Answer: {}".format(correct_answer))
                print_func("--------------------------------")

            if self.zeroshot:
                start = time.time()
                file_print("Solving with zero-shot approach...")
                solution = self.solve_with_zeroshot(problem, subject)
                timing_stats['zeroshot_solution'] = time.time() - start
                
                file_print("\nZero-shot Solution:")
                file_print(solution)
                
                # Extract predicted answer - look for "final answer" in any case
                predicted_answer = None
                solution_lines = solution.split('\n')
                for i, line in enumerate(solution_lines):
                    if 'the final answer is' in line.lower():
                        try:
                            line = line.replace("the final answer is", "The final answer is")
                            # Get the first letter after "The final answer is"
                            answer_part = line.split('The final answer is')[1].strip()
                            letters = re.findall(r'[A-D]', answer_part)
                            if letters:
                                predicted_answer = letters[0]
                                break
                        except:
                            continue
                    elif 'the answer is' in line.lower():
                        try:
                            line = line.replace("the answer is", "The answer is")
                            # Get the first letter after "The answer is"
                            answer_part = line.split('The answer is')[1].strip()
                            letters = re.findall(r'[A-D]', answer_part)
                            if letters:
                                predicted_answer = letters[0]
                                break
                        except:
                            continue
                if predicted_answer is None:
                    predicted_answer = 'None'
                
                # Update running accuracy statistics
                if hasattr(self, '_total_problems'):
                    self._total_problems += 1
                else:
                    self._total_problems = 1
                
                if hasattr(self, '_correct_problems'):
                    if str(predicted_answer).lower() == str(correct_answer).lower():
                        self._correct_problems += 1
                else:
                    self._correct_problems = 1 if str(predicted_answer).lower() == str(correct_answer).lower() else 0
                
                # Update subject-specific statistics
                self._subject_stats[subject]['total_problems'] += 1
                if str(predicted_answer).lower() == str(correct_answer).lower():
                    self._subject_stats[subject]['correct_problems'] += 1
                
                current_accuracy = (self._correct_problems / self._total_problems) * 100
                subject_accuracy = (self._subject_stats[subject]['correct_problems'] / 
                                 self._subject_stats[subject]['total_problems']) * 100

                if str(predicted_answer).lower() == str(correct_answer).lower():
                    correct_check = 1
                else:
                    correct_check = 0

                # Log metrics to wandb
                self.run.log({
                    "accuracy": current_accuracy,
                    "correct_problems": self._correct_problems,
                    "total_problems": self._total_problems,
                    "is_correct": correct_check,
                    f"{subject}_accuracy": subject_accuracy,
                    f"{subject}_correct_problems": self._subject_stats[subject]['correct_problems'],
                    f"{subject}_total_problems": self._subject_stats[subject]['total_problems']
                })
                
                # Save to summary file with running accuracy
                if predicted_answer is not None:
                    result = "Correct" if str(predicted_answer).lower() == str(correct_answer).lower() else "Wrong"
                    summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {subject_accuracy:^11.2f}%")
                    
                    # Save detailed results to the detailed file
                    file_print("\nFinal Results:")
                    file_print(f"Predicted Answer: {predicted_answer}")
                    file_print(f"Actual Answer: {correct_answer}")
                    file_print(f"Result: {result}")
                    file_print(f"Subject Running Accuracy: {self._subject_stats[subject]['correct_problems']}/{self._subject_stats[subject]['total_problems']} = {subject_accuracy:.2f}%")
                    file_print(f"Overall Running Accuracy: {self._correct_problems}/{self._total_problems} = {current_accuracy:.2f}%")
                
                file_print("\nTiming Statistics:")
                file_print("=" * 50)
                file_print(f"Zero-shot Solution: {timing_stats['zeroshot_solution']:.2f} seconds")

            elif self.zeroshot_RAG:
                start = time.time()
                file_print("Solving with zero-shot approach...")
                documents, select_num = self.get_wiki_search_zeroshot(problem, num_results=self.doc_num)
                file_print(f"{select_num} relevant documents found")
                solution = self.solve_with_zeroshot_RAG(problem, documents)
                timing_stats['zeroshot_solution'] = time.time() - start
                
                file_print("\nZero-shot Solution:")
                file_print(solution)
                
                # Extract predicted answer - look for "final answer" in any case
                predicted_answer = None
                solution_lines = solution.split('\n')
                for i, line in enumerate(solution_lines):
                    if 'the final answer is' in line.lower():
                        try:
                            line = line.replace("the final answer is", "The final answer is")
                            # Get the first letter after "The final answer is"
                            answer_part = line.split('The final answer is')[1].strip()
                            letters = re.findall(r'[A-D]', answer_part)
                            if letters:
                                predicted_answer = letters[0]
                                break
                        except:
                            continue
                    elif 'the answer is' in line.lower():
                        try:
                            line = line.replace("the answer is", "The answer is")
                            # Get the first letter after "The answer is"
                            answer_part = line.split('The answer is')[1].strip()
                            letters = re.findall(r'[A-D]', answer_part)
                            if letters:
                                predicted_answer = letters[0]
                                break
                        except:
                            continue    
                if predicted_answer is None:
                    predicted_answer = 'None'
                
                # Update running accuracy statistics
                if hasattr(self, '_total_problems'):
                    self._total_problems += 1
                else:
                    self._total_problems = 1
                
                if hasattr(self, '_correct_problems'):
                    if str(predicted_answer).lower() == str(correct_answer).lower():
                        self._correct_problems += 1
                else:
                    self._correct_problems = 1 if str(predicted_answer).lower() == str(correct_answer).lower() else 0
                
                # Update subject-specific statistics
                self._subject_stats[subject]['total_problems'] += 1
                if str(predicted_answer).lower() == str(correct_answer).lower():
                    self._subject_stats[subject]['correct_problems'] += 1
                
                current_accuracy = (self._correct_problems / self._total_problems) * 100
                subject_accuracy = (self._subject_stats[subject]['correct_problems'] / 
                                 self._subject_stats[subject]['total_problems']) * 100

                if str(predicted_answer).lower() == str(correct_answer).lower():
                    correct_check = 1
                else:
                    correct_check = 0

                # Log metrics to wandb
                self.run.log({
                    "accuracy": current_accuracy,
                    "correct_problems": self._correct_problems,
                    "total_problems": self._total_problems,
                    "is_correct": correct_check,
                    f"{subject}_accuracy": subject_accuracy,
                    f"{subject}_correct_problems": self._subject_stats[subject]['correct_problems'],
                    f"{subject}_total_problems": self._subject_stats[subject]['total_problems']
                })
                
                # Save to summary file with running accuracy
                if predicted_answer is not None:
                    result = "Correct" if str(predicted_answer).lower() == str(correct_answer).lower() else "Wrong"
                    summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {subject_accuracy:^11.2f}%")
                    
                    # Save detailed results to the detailed file
                    file_print("\nFinal Results:")
                    file_print(f"Predicted Answer: {predicted_answer}")
                    file_print(f"Actual Answer: {correct_answer}")
                    file_print(f"Result: {result}")
                    file_print(f"Subject Running Accuracy: {self._subject_stats[subject]['correct_problems']}/{self._subject_stats[subject]['total_problems']} = {subject_accuracy:.2f}%")
                    file_print(f"Overall Running Accuracy: {self._correct_problems}/{self._total_problems} = {current_accuracy:.2f}%")
                
                file_print("\nTiming Statistics:")
                file_print("=" * 50)
                file_print(f"Zero-shot Solution: {timing_stats['zeroshot_solution']:.2f} seconds")

            elif self.PD_baseline:
                # Step 1: Generate subquestions
                start = time.time()
                file_print("Step 1: Generating subquestions and search queries...")
                subq_result = self.generate_subquestions(problem)
                file_print(subq_result)
                timing_stats['generate_subquestions'] = time.time() - start
                
                # Parse the results
                start = time.time()
                subquestions, search_queries, search_categories = self.parse_subquestions_and_queries(subq_result)
                timing_stats['parse_subquestions'] = time.time() - start
                
                file_print("There are ", len(subquestions), " subquestions")
                if len(subquestions) == 0:
                    start = time.time()
                    final_answer = self.solve_with_zeroshot(problem, subject)
                    timing_stats['final_answer'] = time.time() - start
                    file_print("No subquestions generated. Solving with zeroshot...")
                    file_print(final_answer)
                else:
                    # Initialize storage for final solutions
                    subsolutions = []
                    
                    # Steps 2-5: Generate examples and solve each subquestion
                    timing_stats['subquestions'] = {}
                    for step_num in range(1, len(subquestions) + 1):
                        step_timing = {}
                        file_print(f"\nStep {step_num + 1}: Processing subquestion {step_num}...")
                        
                        start = time.time()
                        solution = self.solve_subquestion_base(problem, subquestions, subsolutions, step_num, subject)
                        step_timing['solution_generation'] = time.time() - start
                        
                        
                        timing_stats['subquestions'][f'step_{step_num}'] = step_timing
                        subsolutions.append(solution)
                        
                        file_print(f"\nSolution for subquestion {step_num}:")
                        file_print(solution)
                        
                        file_print("\n" + "="*50 + "\n")  # 구분선 추가
                    
                    # Step 6: Generate final answer
                    start = time.time()
                    file_print("\nStep 6: Generating final answer...")
                    final_answer = self.generate_final_answer(problem, subquestions, subsolutions, subject)
                    file_print(final_answer)
                    timing_stats['final_answer'] = time.time() - start

                # Extract predicted answer and update accuracy
                predicted_answer = None
                final_answer_lines = final_answer.split('\n')
                for i, line in enumerate(final_answer_lines):
                    if 'the final answer is' in line.lower():
                        try:    
                            line = line.replace("the final answer is", "The final answer is")
                            answer_part = line.split('The final answer is')[1].strip()
                            # Get the first letter after "The final answer is"
                            letters = re.findall(r'[A-D]', answer_part)
                            if letters:
                                predicted_answer = letters[0]
                                break
                        except:
                            continue
                    elif 'the answer is' in line.lower():
                        try:
                            line = line.replace("the answer is", "The answer is")
                            answer_part = line.split('The answer is')[1].strip()
                            letters = re.findall(r'[A-D]', answer_part)
                            if letters:
                                    predicted_answer = letters[0]
                                    break
                        except:
                            continue
                if predicted_answer is None:
                    predicted_answer = 'None'
                
                # Update running accuracy statistics
                if hasattr(self, '_total_problems'):
                    self._total_problems += 1
                else:
                    self._total_problems = 1
                
                if hasattr(self, '_correct_problems'):
                    if str(predicted_answer).lower() == str(correct_answer).lower():
                        self._correct_problems += 1
                else:
                    self._correct_problems = 1 if str(predicted_answer).lower() == str(correct_answer).lower() else 0
                
                # Update subject-specific statistics
                self._subject_stats[subject]['total_problems'] += 1
                if str(predicted_answer).lower() == str(correct_answer).lower():
                    self._subject_stats[subject]['correct_problems'] += 1
                
                current_accuracy = (self._correct_problems / self._total_problems) * 100
                subject_accuracy = (self._subject_stats[subject]['correct_problems'] / 
                                 self._subject_stats[subject]['total_problems']) * 100
                
                if str(predicted_answer).lower() == str(correct_answer).lower():
                    correct_check = 1
                else:
                    correct_check = 0
                
                # Log metrics to wandb
                self.run.log({
                    "accuracy": current_accuracy,
                    "correct_problems": self._correct_problems,
                    "total_problems": self._total_problems,
                    "is_correct": correct_check,
                    f"{subject}_accuracy": subject_accuracy,
                    f"{subject}_correct_problems": self._subject_stats[subject]['correct_problems'],
                    f"{subject}_total_problems": self._subject_stats[subject]['total_problems']
                })

                # Save to summary file with running accuracy
                if predicted_answer is not None:
                    result = "Correct" if str(predicted_answer).lower() == str(correct_answer).lower() else "Wrong"
                    summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {subject_accuracy:^11.2f}%")
                    
                    # Save detailed results to the detailed file
                    file_print("\nFinal Results:")
                    file_print(f"Predicted Answer: {predicted_answer}")
                    file_print(f"Actual Answer: {correct_answer}")
                    file_print(f"Result: {result}")
                    file_print(f"Subject Running Accuracy: {self._subject_stats[subject]['correct_problems']}/{self._subject_stats[subject]['total_problems']} = {subject_accuracy:.2f}%")
                    file_print(f"Overall Running Accuracy: {self._correct_problems}/{self._total_problems} = {current_accuracy:.2f}%")
                
                # Print timing statistics
                timing_stats['total_time'] = time.time() - start_total
                file_print("\nTiming Statistics:")
                file_print("=" * 50)
                file_print(f"Total Time: {timing_stats['total_time']:.2f} seconds")
                #file_print(f"Data Loading: {timing_stats['data_loading']:.2f} seconds")
                file_print(f"Generate Subquestions: {timing_stats['generate_subquestions']:.2f} seconds")
                file_print(f"Parse Subquestions: {timing_stats['parse_subquestions']:.2f} seconds")
                
                if 'subquestions' in timing_stats:
                    for step_num, step_timing in timing_stats['subquestions'].items():
                        file_print(f"\n{step_num.upper()}:")
                        for operation, duration in step_timing.items():
                            file_print(f"  {operation}: {duration:.2f} seconds")
                    
                file_print(f"\nFinal Answer Generation: {timing_stats['final_answer']:.2f} seconds")

           

            else:
                # Step 1: Generate subquestions
                start = time.time()
                if self.PD_trigger:
                    file_print("Determine if problem decomposition is needed...")
                    PD_trigger = self.subquestion_trigger(problem)
                    timing_stats['PD_trigger'] = time.time() - start
                    file_print(PD_trigger)
                if self.PD_trigger and 'direct' in PD_trigger.lower():
                    file_print("Problem decomposition is not needed.")
                    start = time.time()
                    file_print("Solving with zero-shot approach...")
                    solution = self.solve_with_zeroshot(problem,subject)
                    timing_stats['final_answer'] = time.time() - start
                    
                    file_print("\nZero-shot Solution:")
                    file_print(solution)
                    
                    # Extract predicted answer - look for "final answer" in any case
                    predicted_answer = None
                    solution_lines = solution.split('\n')
                    for i, line in enumerate(solution_lines):
                        if 'the final answer is' in line.lower():
                            try:
                                line = line.replace("the final answer is", "The final answer is")
                                # Get the first letter after "The final answer is"
                                answer_part = line.split('The final answer is')[1].strip()
                                letters = re.findall(r'[A-D]', answer_part)
                                if letters:
                                    predicted_answer = letters[0]
                                    break
                            except:
                                continue
                        elif 'the answer is' in line.lower():
                            try:
                                line = line.replace("the answer is", "The answer is")
                                # Get the first letter after "The answer is"
                                answer_part = line.split('The answer is')[1].strip()
                                letters = re.findall(r'[A-D]', answer_part)
                                if letters:
                                    predicted_answer = letters[0]
                                    break
                            except:
                                continue
                    if predicted_answer is None:
                        predicted_answer = 'None'

                else:
                    if self.PD_trigger:
                        file_print("Problem decomposition is needed.")
                    file_print("Step 1: Generating subquestions and search queries...")
                    subq_result = self.generate_subquestions(problem)
                    file_print(subq_result)
                    timing_stats['generate_subquestions'] = time.time() - start
                    
                    # Parse the results
                    start = time.time()
                    subquestions, search_queries, search_categories = self.parse_subquestions_and_queries(subq_result)
                    timing_stats['parse_subquestions'] = time.time() - start
                    
                    file_print("There are ", len(subquestions), " subquestions")
                    
                    if len(subquestions) == 0:
                        start = time.time()
                        final_answer = self.solve_with_zeroshot(problem, subject)
                        timing_stats['final_answer'] = time.time() - start
                        file_print("No subquestions generated. Solving with zeroshot...")
                        file_print(final_answer)
                    else:
                        # Initialize storage for final solutions
                        subsolutions = []
                        
                        # Steps 2-5: Generate examples and solve each subquestion
                        timing_stats['subquestions'] = {}
                        for step_num in range(1, len(subquestions) + 1):
                            step_timing = {}
                            file_print(f"\nStep {step_num + 1}: Processing subquestion {step_num}...")
                            
                            # Time document retrieval
                            start = time.time()
                            documents, select_num, distances, response = self.get_wiki_search_results(problem, subquestions[step_num - 1], search_queries[step_num - 1], search_categories[step_num - 1], num_results=self.doc_num, subject=subject)
                            step_timing['document_retrieval'] = time.time() - start
                            file_print(f"Document Distances: {distances}")
                            file_print(f"Explanation generated: {response}")
                            if self.filter:
                                if select_num == 0:
                                    file_print(f"\nNo relevant documents found for Subquestion {step_num}.")
                                    start=time.time()
                                    solution = self.solve_subquestion_base(problem, subquestions, subsolutions, step_num, subject)
                                    step_timing['solution_generation'] = time.time() - start
                                    file_print(f"\nSolve subquestion {step_num} with zeroshot:")
                                    file_print(solution)
                                else:
                                    file_print(f"{select_num} relevant documents found for Subquestion {step_num}.")
                                    file_print(f"\nSearch Results for Subquestion {step_num}:")
                                    file_print(documents)
                                    start=time.time()
                                    if self.generate_ex:
                                        examples = self.generate_examples(problem, documents, subquestions, step_num)
                                        if examples != "None":
                                            step_timing['example_generation'] = time.time() - start
                                            file_print(f"\nGenerated Examples for Subquestion {step_num}:")
                                            file_print(examples)
                                            start=time.time()
                                            solution = self.solve_subquestion(problem, subquestions, examples, 
                                                                        subsolutions, step_num, subject)
                                            step_timing['solution_generation'] = time.time() - start
                                            file_print(f"\nSolve subquestion {step_num} with examples:")
                                            file_print(solution)
                                        else:
                                            file_print(f"\nNo relevant examples found for Subquestion {step_num}.")
                                            start=time.time()
                                            solution = self.solve_subquestion_with_docs(problem, subquestions, 
                                                                            documents, subsolutions, step_num, subject)
                                            step_timing['solution_generation'] = time.time() - start
                                            file_print(f"\nSolve subquestion {step_num} with documents:")
                                            file_print(solution)
                                    else:
                                        file_print(f"\nNo example generation for Subquestion {step_num}.")
                                        start=time.time()
                                        solution = self.solve_subquestion_with_docs(problem, subquestions, 
                                                                            documents, subsolutions, step_num, subject)
                                        step_timing['solution_generation'] = time.time() - start
                                        file_print(f"\nSolve subquestion {step_num} with documents:")
                                        file_print(solution)
                            else:      
                                if documents == "No relevant documents found.":
                                    documents = ""
                                    file_print(f"\nNo relevant documents found for Subquestion {step_num}.")
                                else:
                                    file_print(f"{select_num} relevant documents found for Subquestion {step_num}.")
                                    # Save search results
                                    file_print(f"\nSearch Results for Subquestion {step_num}:")
                                    file_print(documents) 
                                
                                if self.generate_ex:
                                    # Time example generation
                                    start = time.time()
                                    examples = self.generate_examples(problem, documents, subquestions, step_num)
                                    if examples != "None":
                                        step_timing['example_generation'] = time.time() - start
                                        
                                        file_print(f"\nGenerated Examples for Subquestion {step_num}:")
                                        file_print(examples)
                                        
                                        # Time solution generation with examples
                                        start = time.time()
                                        file_print("Solving subquestion with examples...")
                                        solution = self.solve_subquestion(problem, subquestions, examples, 
                                                                subsolutions, step_num, subject)
                                        step_timing['solution_generation'] = time.time() - start
                                        file_print(f"\nSolve subquestion {step_num} with examples:")
                                        file_print(solution)
                                    else:
                                        file_print(f"\nNo relevant examples found for Subquestion {step_num}.")
                                        start=time.time()
                                        solution = self.solve_subquestion_with_docs(problem, subquestions, 
                                                                            documents, subsolutions, step_num, subject)
                                        step_timing['solution_generation'] = time.time() - start
                                        file_print(f"\nSolve subquestion {step_num} with documents:")
                                        file_print(solution)
                                else:
                                    # Time direct solution generation
                                    start = time.time()
                                    solution = self.solve_subquestion_with_docs(problem, subquestions, 
                                                                            documents, subsolutions, step_num, subject)
                                    step_timing['solution_generation'] = time.time() - start
                                    file_print(f"\nSolve subquestion {step_num} with documents:")
                                    file_print(solution)
                                
                                

                            timing_stats['subquestions'][f'step_{step_num}'] = step_timing
                            subsolutions.append(solution)
                            
                            
                            
                            file_print("\n" + "="*50 + "\n")  # 구분선 추가
                        
                        # Step 6: Generate final answer
                        start = time.time()
                        file_print("\nStep 6: Generating final answer...")
                        final_answer = self.generate_final_answer(problem, subquestions, subsolutions, subject)
                        file_print(final_answer)
                        timing_stats['final_answer'] = time.time() - start

                    # Extract predicted answer and update accuracy
                    predicted_answer = None
                    final_answer_lines = final_answer.split('\n')
                    for i, line in enumerate(final_answer_lines):
                        if 'the final answer is' in line.lower():
                            try:   
                                line = line.replace("the final answer is", "The final answer is")
                                answer_part = line.split('The final answer is')[1].strip()
                                # Get the first letter after "The final answer is"
                                letters = re.findall(r'[A-D]', answer_part)
                                if letters:
                                    predicted_answer = letters[0]
                                    break
                            except:
                                continue
                        elif 'the answer is' in line.lower():
                            try:
                                line = line.replace("the answer is", "The answer is")
                                answer_part = line.split('The answer is')[1].strip()
                                letters = re.findall(r'[A-D]', answer_part)
                                if letters:
                                    predicted_answer = letters[0]
                                    break
                            except:
                                continue
                    if predicted_answer is None:
                        predicted_answer = 'None'
                
                # Update running accuracy statistics
                if hasattr(self, '_total_problems'):
                    self._total_problems += 1
                else:
                    self._total_problems = 1
                
                if hasattr(self, '_correct_problems'):
                    if str(predicted_answer).lower() == str(correct_answer).lower():
                        self._correct_problems += 1
                else:
                    self._correct_problems = 1 if str(predicted_answer).lower() == str(correct_answer).lower() else 0
                
                # Update subject-specific statistics
                self._subject_stats[subject]['total_problems'] += 1
                if str(predicted_answer).lower() == str(correct_answer).lower():
                    self._subject_stats[subject]['correct_problems'] += 1
                
                current_accuracy = (self._correct_problems / self._total_problems) * 100
                subject_accuracy = (self._subject_stats[subject]['correct_problems'] / 
                                 self._subject_stats[subject]['total_problems']) * 100
                
                if str(predicted_answer).lower() == str(correct_answer).lower():
                    correct_check = 1
                else:
                    correct_check = 0
                
                # Log metrics to wandb
                self.run.log({
                    "accuracy": current_accuracy,
                    "correct_problems": self._correct_problems,
                    "total_problems": self._total_problems,
                    "is_correct": correct_check,
                    f"{subject}_accuracy": subject_accuracy,
                    f"{subject}_correct_problems": self._subject_stats[subject]['correct_problems'],
                    f"{subject}_total_problems": self._subject_stats[subject]['total_problems']
                })

                # Save to summary file with running accuracy
                if predicted_answer is not None:
                    result = "Correct" if str(predicted_answer).lower() == str(correct_answer).lower() else "Wrong"
                    summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {subject_accuracy:^11.2f}%")
                    
                    # Save detailed results to the detailed file
                    file_print("\nFinal Results:")
                    file_print(f"Predicted Answer: {predicted_answer}")
                    file_print(f"Actual Answer: {correct_answer}")
                    file_print(f"Result: {result}")
                    file_print(f"Subject Running Accuracy: {self._subject_stats[subject]['correct_problems']}/{self._subject_stats[subject]['total_problems']} = {subject_accuracy:.2f}%")
                    file_print(f"Overall Running Accuracy: {self._correct_problems}/{self._total_problems} = {current_accuracy:.2f}%")
                
                # Print timing statistics
                timing_stats['total_time'] = time.time() - start_total
                file_print("\nTiming Statistics:")
                file_print("=" * 50)
                file_print(f"Total Time: {timing_stats['total_time']:.2f} seconds")
                #file_print(f"Data Loading: {timing_stats['data_loading']:.2f} seconds")
                if 'PD_trigger' in timing_stats:
                    file_print(f"PD Trigger: {timing_stats['PD_trigger']:.2f} seconds")
                if 'generate_subquestions' in timing_stats:
                    file_print(f"Generate Subquestions: {timing_stats['generate_subquestions']:.2f} seconds")
                if 'parse_subquestions' in timing_stats:
                    file_print(f"Parse Subquestions: {timing_stats['parse_subquestions']:.2f} seconds")
                if 'subquestions' in timing_stats:
                    for step_num, step_timing in timing_stats['subquestions'].items():
                        file_print(f"\n{step_num.upper()}:")
                        for operation, duration in step_timing.items():
                            file_print(f"  {operation}: {duration:.2f} seconds")
                    
                file_print(f"\nFinal Answer Generation: {timing_stats['final_answer']:.2f} seconds")
                
            # Print timing statistics
   
        except Exception as e:
            file_print(f"Error occurred: {e}")
            summary_print(f"{prob_num:^10} | {'Error':^10} | {'-':^10} | {problem['answer']:^10} | {'-':^11}")
            raise

        print(f"\nProcessing complete! Results saved to:")
        print(f"Detailed results: {detailed_file}")
        print(f"Summary results: {summary_file}")

    
    