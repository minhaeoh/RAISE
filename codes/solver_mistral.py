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

class MultiHopSolver:
    def __init__(self, mode="Step_RAG",ex_num=2, trigger=False, trigger_value=0.8, doc_num=10, query_mode="q1", wandb_run=None, api_key=0
    ):
        # Initialize environment
        env_path = "/home/minhae/multihop-RAG2/.env"
        load_dotenv(dotenv_path=env_path)
        
        # Get API keys from environment variables
        self.hf_token = os.getenv('HF_TOKEN')
        self.wandb_api_key = os.getenv('WANDB_API_KEY')
        self.wandb_entity = os.getenv('WANDB_ENTITY', 'minhae')  # Default to 'minhae' if not set
        self.mistral_api_key = os.getenv(f'MISTRAL_API_KEY{api_key}')

        # Initialize counters for tracking overall performance
        self._total_problems = 0
        self._correct_problems = 0
        self._domain_stats = {}
        self._h_domain_stats = {}
        self._difficulty_stats = {'writer':{}, 'ev1':{}, 'ev2':{}}
        self._accuracy_stats = {'domain':{}, 'h_domain':{}, 'difficulty_ev1':{}, 'difficulty_ev2':{}, 'difficulty_writer':{}}

        self.mode = mode
        self.ex_num = ex_num
        self.trigger = trigger
        self.trigger_value = float(trigger_value)
        self.doc_num = doc_num
        self.query_mode = query_mode
        self.wandb_run = wandb_run
        self.model_id = "mistral-small-latest"

        # Initialize timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        # configuration details to directory name
        if self.mode == "zeroshot_COT":
            config_str = "zeroshot_COT"
        elif self.mode == "zeroshot_PS":
            config_str = "zeroshot_PS"
        elif self.mode == "zeroshot_RAG":
            config_str = "zeroshot_RAG"
        elif self.mode == "PD":
            config_str = "PD"
        else:
            config_str = self.query_mode
            config_str += f"_{self.mode}"
        self.config_str = config_str

        # Set device for CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize Mistral model
        try:
            print("Initializing Mistral model...")
            self.model_id = "mistral-small-latest"
            self.client = Mistral(api_key=self.mistral_api_key)
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

    def generate_text(self, prompt, max_length=384, temperature=0.7, top_p=0.9):
        """Generate text using the Mistral pipeline with output cleaning"""
        time.sleep(1)
        try:
            response = self.client.chat.complete(
                model = self.model_id,
                messages = [
                    {
                        "role":"user",
                        "content":prompt,
                    },
                ],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                )
            response = response.choices[0].message.content
            response = re.sub(r"(\|\s*){3,}", "", response).strip()
            
            return response
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""

    def solve_with_zeroshot_COT(self, problem):
        choices_text = problem['choices']['text']

        prompt = f"""You are solving a multiple choice question. Think step by step and then finish your answer with "The final answer is (X)" where X is the correct letter choice.

        Question: {problem['question']} 
        Answer Choices: {choices_text}
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""
        
    def solve_with_zeroshot_PS(self, problem):
        choices_text = problem['choices']['text']

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

        The final answer is (X). Where X is the correct letter choice.

        
        Question: {problem['question']} 
        Answer Choices: {choices_text}
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""

    def solve_with_zeroshot_RAG(self, problem, documents):
        choices_text = problem['choices']['text']

        prompt = f"""
        You are given a problem and a set of related documents.

        Use the information from the documents to help solve the problem step by step. If the documents contain useful facts, definitions, or formulas, apply them in your reasoning.

        Clearly explain your reasoning, and then finish your answer with "The final answer is (X)" where X is the correct letter choice.

        Question: {problem['question']} 
        Answer Choices: {choices_text}

        Documents: 
        {documents}       
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
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
                return "\n\n".join(documents), select_num, distances[0]
            
        except Exception as e:
            print(f"Error in get_wiki_search_results: {e}")
            return "No relevant documents found.", 0, distances[0]

    def extract_answer(self, solution):
        for line in solution.split("\n"):
            if "the final answer is" in line.lower():
                line = line.replace("The final answer is", "the final answer is").strip()
                answer = line.split("the final answer is")[1].strip()
                answer = re.findall(r'[A-D]', answer)
                if answer:
                    return answer[0]
        return "None"

    def generate_query(self, subquestion, query_mode, query):
        if query_mode == "subq":
            return subquestion
        elif query_mode == "q0" :
            return query
        elif query_mode == "q1" :
            prompt = f"""
            You are given a search query.

            Briefly explain the core concept or relationship described by the query in 2–3 sentences. Focus only on the essential scientific or mathematical idea.

            Search Query: {query}
            """
            response = self.generate_text(prompt, max_length=100)
            return response
        else:
            raise ValueError(f"Invalid query mode: {query_mode}. Query mode should be one of subq, q0, q1")

    def generate_subquestions(self, problem):
        choices_text = problem['choices']['text']

        prompt = f"""
        You are given a multiple-choice question. Your task is to break it down into a series of essential subquestions that reflect the full reasoning process needed to solve the problem.

        Guidelines:
        Each subquestion should correspond to a clear reasoning step, concept, or calculation.
        Subquestions may include specific numbers or variables from the original question to make the reasoning concrete.
        Each search query should reflect general scientific or mathematical knowledge needed to answer the subquestion. Do not include numeric values or overly specific phrasing from the original question in the search query.
        The goal is to simulate how someone would think step-by-step through the problem and search for relevant background knowledge at each stage.

        STRICT FORMAT REQUIREMENTS:
        1. For each subquestion, you MUST provide exactly two parts in this order:
        - The subquestion
        - A search query for that subquestion

        2. Use EXACTLY this format for each subquestion:
        Subquestion 1: [Write a concrete, reasoning-based subquestion—may include values or variables from the problem]
        Search Query for Subquestion 1: [Write a general search query someone might realistically use to learn how to answer this subquestion]

        Question: {problem['question']}
        Answer Choices: {choices_text}
        """
        try:
            response = self.generate_text(prompt, max_length=600)
            subquestions = []
            queries = []
            for line in response.split("\n"):
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
        choices_text = problem['choices']['text']

        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve

        Your task is to solve the current subquestion.

        Question: {problem['question']}
        Answer Choices: {choices_text}
        """
        if step_num > 1:
            for i in range(step_num - 1):
                prompt += f"""
                (previous) Subquestion {i+1}: {subquestions[i]}
                (previous) Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        (current) Subquestion {step_num}: {subquestions[step_num-1]}
        """

        try:
            response = self.generate_text(prompt, max_length=1000)
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""


    def solve_subquestion_with_docs(self, problem, step_num, subquestions, subsolutions, documents):
        choices_text = problem['choices']['text']

        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Documents that are relevant to the current subquestion

        Your task is to solve the current subquestion.

        Question: {problem['question']}
        Answer Choices: {choices_text}
        """
        if step_num > 1:
            for i in range(step_num - 1):
                prompt += f"""
                (previous) Subquestion {i+1}: {subquestions[i]}
                (previous) Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        (current) Subquestion {step_num}: {subquestions[step_num-1]}
        """
        prompt += f"""
        Documents:
        {documents}
        """

        try:
            response = self.generate_text(prompt, max_length=1000)
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""

    def solve_subquestion_with_ex(self, problem, step_num, subquestions, subsolutions, examples):
        choices_text = problem['choices']['text']

        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Examples that are relevant to the current subquestion

        Your task is to solve the current subquestion.

        Question: {problem['question']}
        Answer Choices: {choices_text}
        """
        if step_num > 1:
            for i in range(step_num - 1):
                prompt += f"""
                (previous) Subquestion {i+1}: {subquestions[i]}
                (previous) Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        (current) Subquestion {step_num}: {subquestions[step_num-1]}
        """
        prompt += f"""
        Examples:
        {examples}
        """

        try:
            response = self.generate_text(prompt, max_length=1000)
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""
        return solution

    def generate_examples(self,step_num, problem, documents, subquestions):
        choices_text = problem['choices']['text']

        prompt = f"""
        You are a example generator. You will be given the original question, a subquestion, and a knowledge document.

        Your task is:
        - Generate {self.ex_num} example questions with full solutions based on the knowledge document.
        - The questions should be similar to the subquestion.
        - The questions should help understand the concept mentioned in the subquestion.
        - Do not solve the original question.
        - Do not use multiple choice or ask the user to select an option. Make each question open-ended or short-answer style.
        - Do not mention or refer to the document in the questions or solutions. The outputs should be self-contained.
        - Each solution should be accurate, clear, and informative.

        Use EXACTLY this format for each example question and solution:
        Question 1: [Write the question here]
        Solution 1: [Write the full solution here]

        Original Question:
        Question: {problem['question']} 
        Answer Choices: {choices_text}

        Subquestion {step_num}: {subquestions[step_num-1]}

        Knowledge Documents: {documents}"""
        try:
            response = self.generate_text(prompt, max_length=1000)
            return response
        except Exception as e:
            print(f"Error in generate_examples: {e}")
            return ""

    def generate_final_answer(self, problem, subquestions, subsolutions):
        choices_text = problem['choices']['text']

        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. Each subquestion has already been solved. Your task is to carefully read the original question and the several subquestion solutions, then use them to determine the final answer. Think step by step and then finish your answer with "The final answer is (X)" where X is the correct letter choice.

        Original Question:
        Question: {problem['question']}
        Answer Choices: {choices_text}

        Subquestions and Solutions:
        """
        for i, (subquestion, subsolution) in enumerate(zip(subquestions, subsolutions)):
            prompt += f"""
            Subquestion {i+1}: {subquestion}
            Subquestion {i+1} Solution: {subsolution}
            """

        try:
            response = self.generate_text(prompt, max_length=1000)
            return response
        except Exception as e:
            print(f"Error in generate_final_answer: {e}")
            return ""

    def solve_problem(self, subject, prob_num, problem=None):
        base_dir = f"/home/minhae/multihop-RAG2/results/{subject}/mistral-small-latest"
        
        self.output_dir = os.path.join(base_dir, f"{self.config_str}",self.timestamp)
        if prob_num == 1:
            os.makedirs(self.output_dir, exist_ok=True)
            # Initialize wandb if not already initialized
            if self.run is None and self.wandb_api_key:
                os.environ['WANDB_API_KEY'] = self.wandb_api_key
                self.run = wandb.init(
                    project=f"{subject}-mistral-small-3.1",
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
            print_func("Question: {}".format(problem['question']))
            print_func("Choices: {}".format(problem['choices']['text']))
            options = problem['choices']['label']
            correct_answer = problem['answerKey']
            print_func("Correct Answer: {}".format(correct_answer))
            print_func("--------------------------------")

        if problem['domain'] not in self._domain_stats:
            self._domain_stats[problem['domain']] = {
                'total_problems': 0,
                'correct_problems': 0
            }
        
        if problem['h_domain'] not in self._h_domain_stats:
            self._h_domain_stats[problem['h_domain']] = {
                'total_problems': 0,
                'correct_problems': 0
            }

        if problem['difficulty_1'] not in self._difficulty_stats['ev1']:
            self._difficulty_stats['ev1'][problem['difficulty_1']] = {
                'total_problems': 0,
                'correct_problems': 0
            }

        if problem['difficulty_2'] not in self._difficulty_stats['ev2']:
            self._difficulty_stats['ev2'][problem['difficulty_2']] = {
                'total_problems': 0,
                'correct_problems': 0
            }

        if problem['difficulty_w'] not in self._difficulty_stats['writer']:
            self._difficulty_stats['writer'][problem['difficulty_w']] = {
                'total_problems': 0,
                'correct_problems': 0
            }

        
                
        # Add timing dictionary
        start_total = time.time()

        if self.mode == "zeroshot_COT":
            solution = self.solve_with_zeroshot_COT(problem)
            predicted_answer = self.extract_answer(solution)
            file_print(f"Solution: {solution}")

        elif self.mode == "zeroshot_PS":
            solution = self.solve_with_zeroshot_PS(problem)
            predicted_answer = self.extract_answer(solution)
            file_print(f"Solution: {solution}")

        elif self.mode == "zeroshot_RAG":
            query = problem['question']
            documents, select_num, distances = self.get_wiki_search_results(query, self.doc_num)
            if select_num == 0:
                solution = self.solve_with_zeroshot_COT(problem)
            else:
                solution = self.solve_with_zeroshot_RAG(problem, documents)
            predicted_answer = self.extract_answer(solution)
            file_print(f"Query: {query}")
            file_print(f"There are {select_num} documents")
            file_print(f"Distances: {distances}")
            if select_num>0:
                file_print(f"Documents: {documents}")
            file_print(f"Solution: {solution}")            
        
        else:
            file_print(f"Stage 1: Generate subquestions and queries")
            subquestions, queries = self.generate_subquestions(problem)
            file_print(f"There are {len(subquestions)} subquestions and queries")
            file_print(f"Subquestions: {subquestions}")
            file_print(f"Queries: {queries}")
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
                    documents, select_num, distances = self.get_wiki_search_results(query, self.doc_num)
                    if select_num == 0 :
                        solution = self.solve_subquestion_base(problem, step_num, subquestions, solutions)
                    else:
                        solution = self.solve_subquestion_with_docs(problem, step_num, subquestions, solutions, documents)
                    solutions.append(solution)
                    file_print(f"Query: {query}")
                    file_print(f"There are {select_num} documents")
                    file_print(f"Distances: {distances}")
                    if select_num>0:
                        file_print(f"Documents: {documents}")
                    file_print(f"Solution: {solution}")
                    file_print("--------------------------------")

                elif self.mode == "Step_RAG_EX":
                    query = self.generate_query(subquestion, self.query_mode, query)
                    documents, select_num, distances = self.get_wiki_search_results(query, self.doc_num)
                    if select_num == 0:
                        solution = self.solve_subquestion_base(problem, step_num, subquestions, solutions)
                    else:
                        examples = self.generate_examples(step_num, problem, documents, subquestion)
                        solution = self.solve_subquestion_with_ex(problem, step_num, subquestions, solutions, examples)
                    solutions.append(solution)
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
            final_answer = self.generate_final_answer(problem, subquestions, solutions)
            predicted_answer = self.extract_answer(final_answer)
            file_print(f"Final Solution: {final_answer}")
        
        if str(predicted_answer).lower() == str(correct_answer).lower():
            correct = True
        else:
            correct = False

        if hasattr(self, '_total_problems'):
            self._total_problems += 1
        else:
            self._total_problems = 1
        
        if hasattr(self, '_correct_problems'):
            if correct:
                self._correct_problems += 1
        else:
            self._correct_problems = 1 if correct else 0
        
        current_accuracy = (self._correct_problems / self._total_problems) * 100

        if correct:
            correct_check = 1
        else:
            correct_check = 0

        if problem['domain'] in self._domain_stats:
            if correct :
                self._domain_stats[problem['domain']]['correct_problems'] += 1
        else:
            self._domain_stats[problem['domain']]['correct_problems'] = 1 if correct else 0
                    
        if problem['h_domain'] in self._h_domain_stats:    
            if correct :
                self._h_domain_stats[problem['h_domain']]['correct_problems'] += 1
            else:
                self._h_domain_stats[problem['h_domain']]['correct_problems'] = 1 if correct else 0
        if problem['difficulty_1'] in self._difficulty_stats['ev1']:
            if correct :
                self._difficulty_stats['ev1'][problem['difficulty_1']]['correct_problems'] += 1
            else:
                self._difficulty_stats['ev1'][problem['difficulty_1']]['correct_problems'] = 1 if correct else 0
        if problem['difficulty_2'] in self._difficulty_stats['ev2']:
            if correct :
                self._difficulty_stats['ev2'][problem['difficulty_2']]['correct_problems'] += 1
            else:
                self._difficulty_stats['ev2'][problem['difficulty_2']]['correct_problems'] = 1 if correct else 0
        if problem['difficulty_w'] in self._difficulty_stats['writer']:
            if correct :
                self._difficulty_stats['writer'][problem['difficulty_w']]['correct_problems'] += 1
            else:
                self._difficulty_stats['writer'][problem['difficulty_w']]['correct_problems'] = 1 if correct else 0
        if problem['domain'] in self._domain_stats:
            self._domain_stats[problem['domain']]['total_problems'] += 1
        else:
            self._domain_stats[problem['domain']]['total_problems'] = 1
        if problem['h_domain'] in self._h_domain_stats:
            self._h_domain_stats[problem['h_domain']]['total_problems'] += 1
        else:
            self._h_domain_stats[problem['h_domain']]['total_problems'] = 1
        if problem['difficulty_1'] in self._difficulty_stats['ev1']:
            self._difficulty_stats['ev1'][problem['difficulty_1']]['total_problems'] += 1
        else:
            self._difficulty_stats['ev1'][problem['difficulty_1']]['total_problems'] = 1
        if problem['difficulty_2'] in self._difficulty_stats['ev2']:
            self._difficulty_stats['ev2'][problem['difficulty_2']]['total_problems'] += 1
        else:
            self._difficulty_stats['ev2'][problem['difficulty_2']]['total_problems'] = 1
        if problem['difficulty_w'] in self._difficulty_stats['writer']:
            self._difficulty_stats['writer'][problem['difficulty_w']]['total_problems'] += 1
        else:
            self._difficulty_stats['writer'][problem['difficulty_w']]['total_problems'] = 1
        
        self._accuracy_stats['domain'][problem['domain']] = (self._domain_stats[problem['domain']]['correct_problems'] / self._domain_stats[problem['domain']]['total_problems']) * 100
        self._accuracy_stats['h_domain'][problem['h_domain']] = (self._h_domain_stats[problem['h_domain']]['correct_problems'] / self._h_domain_stats[problem['h_domain']]['total_problems']) * 100
        self._accuracy_stats['difficulty_ev1'][problem['difficulty_1']] = (self._difficulty_stats['ev1'][problem['difficulty_1']]['correct_problems'] / self._difficulty_stats['ev1'][problem['difficulty_1']]['total_problems']) * 100
        self._accuracy_stats['difficulty_ev2'][problem['difficulty_2']] = (self._difficulty_stats['ev2'][problem['difficulty_2']]['correct_problems'] / self._difficulty_stats['ev2'][problem['difficulty_2']]['total_problems']) * 100
        self._accuracy_stats['difficulty_writer'][problem['difficulty_w']] = (self._difficulty_stats['writer'][problem['difficulty_w']]['correct_problems'] / self._difficulty_stats['writer'][problem['difficulty_w']]['total_problems']) * 100                    

        self.run.log({
            "accuracy": current_accuracy,
            "correct_problems": self._correct_problems,
            "total_problems": self._total_problems,
            "is_correct": correct_check
        })
        if predicted_answer is not None:
            result = "Correct" if correct else "Wrong"
            summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {current_accuracy:^11.2f}%")
            for key, value in self._accuracy_stats.items():
                summary_print(f"Accuracy of {key}:")
                for subkey, subvalue in value.items():
                    summary_print(f"{subkey}: {subvalue:.2f}%")
                summary_print("-" * 50)
            # Save detailed results to the detailed file
            file_print("\nFinal Results:")
            file_print(f"Predicted Answer: {predicted_answer}")
            file_print(f"Actual Answer: {correct_answer}")
            file_print(f"Result: {result}")
            file_print(f"Overall Running Accuracy: {self._correct_problems}/{self._total_problems} = {current_accuracy:.2f}%")
        
        # Print timing statistics
        total_time = time.time() - start_total
        file_print(f"Total Time: {total_time:.2f} seconds")


                