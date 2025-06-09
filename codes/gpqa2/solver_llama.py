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
    def __init__(self, mode="Step_RAG",ex_num=2, trigger=False, trigger_value=0.8, doc_num=10, query_mode="q1", wandb_run=None, api_key=0, prompt=False
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
        self._domain_stats = {}
        self._h_domain_stats = {}
        self._difficulty_stats = {'writer':{}, 'ev1':{}, 'ev2':{}}
        self._accuracy_stats = {'domain':{}, 'h_domain':{}, 'difficulty_ev1':{}, 'difficulty_ev2':{}, 'difficulty_writer':{}}


        self.mode = mode
        self.prompt = prompt
        
        self.wandb_run = wandb_run
        self.model_id =  "meta-llama/Llama-3.1-8B-Instruct"


        self.ex_num = None
        self.trigger = None
        self.trigger_value = None
        self.doc_num = None
        self.query_mode = None

        # Initialize timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        # configuration details to directory name
        if self.mode == "zeroshot_COT":
            config_str = "zeroshot_COT"
        elif self.mode == "zeroshot_PS":
            config_str = "zeroshot_PS"
        elif self.mode == "step_back":
            config_str = "step_back"
        elif self.mode == "zeroshot_RAG":
            self.query_mode = query_mode
            self.trigger = trigger
            self.doc_num = doc_num
            config_str = "zeroshot_RAG"
            config_str += f"_{self.query_mode}"
            if self.trigger:
                self.trigger_value = float(trigger_value)
                config_str += f"_trigger{self.trigger_value}"
            config_str += f"_doc{self.doc_num}"
        elif self.mode == "PD":
            config_str = "PD"
        elif self.mode == "PD_step_back":
            config_str = "PD_step_back"
        else:
            self.ex_num = ex_num
            self.trigger = trigger
            self.doc_num = doc_num
            self.query_mode = query_mode
            config_str = self.mode
            config_str += f"_{self.query_mode}"
            if self.trigger:
                self.trigger_value = float(trigger_value)
                config_str += f"_trigger{self.trigger_value}"
            config_str += f"_doc{self.doc_num}"
        if self.prompt:
            self.config_str = f"subQ_prompt_{config_str}"
        else:
            self.config_str = config_str
        # Set device for CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Get number of available GPUs
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Number of available GPUs: {self.num_gpus}")

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

    def generate_text(self, prompt, max_length=384, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
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
                "Note",
                "'''",
                "```"
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
            response = output_text[len(prompt):].strip()
            lines = []
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if not line:
                    continue
                if 'the final answer is' in line.lower():
                    lines.append(line)
                    break
                else:
                    lines.append(line)
            response = "\n".join(lines)
            response = re.sub(r"(\|\s*){3,}", "", response).strip()
            print(f"Prompt: \n{prompt}\n")
            print(f"Response: \n{response}\n")
            return response
            
        except Exception as e:
            print(f"Error in text generation: {e}")
            return ""

    def solve_with_zeroshot_COT(self, problem):
        prompt = f"""You are solving a multiple choice question. Think step by step and show your reasoning clearly.  
        At the end, state your answer in the format: "The final answer is (X)".
        Here, X must be the correct letter choice.

        Question: {problem['question']} 
        Answer Choices: {problem['choices']['text']}

        Solution:
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question:"):
                    response = "\n".join(response.split("\n")[:i])
                    break
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""

    def solve_with_step_back(self, problem):
        prompt = f"""
            You are an expert at Science. You are given a Science problem. Your task is to extract the Science concepts and principles involved in solving the problem.
            
            End your response with "End of generation" after you answer the instructions.

            Question: {problem['question']}
            Answer Choices: {problem['choices']['text']}
            Principles Involved:
            """
        try:
            principles = self.generate_text(prompt, max_length=100)
            for i, line in enumerate(principles.split("\n")):
                line = line.strip()
                if line.startswith("End of generation"):
                    principles = "\n".join(principles.split("\n")[:i])
                    break
                elif "End of generation" in line:
                    principles = "\n".join(principles.split("\n")[:i+1])
                    break
            prompt = f"""You are an expert at Science. You are given a Science problem and a set of principles involved in solving the problem. Solve the problem step by step by folowwing the principles.
        At the end, state your answer in the format: "The final answer is (X)".
        Here, X must be the correct letter choice.

        Question: {problem['question']} 
        Principles: {principles}
        Answer Choices: {problem['choices']['text']}

        Solution: 
        """
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question:"):
                    response = "\n".join(response.split("\n")[:i])
                    break
            return response, principles
        except Exception as e:
            print(f"Error in solve_with_step_back: {e}")
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

        "The final answer is (X)."  Here, X must be the correct letter choice.

        Question: {problem['question']} 
        Answer Choices: {problem['choices']['text']}

        Solution:
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question:"):
                    response = "\n".join(response.split("\n")[:i])
                    break
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot_PS: {e}")
            return ""

    def solve_with_zeroshot_RAG(self, problem, documents):
        prompt = f"""
        You are given a multiple choice question and a set of related documents.

        Use the information from the documents to help solve the problem step by step. If the documents contain useful facts, definitions, or formulas, apply them in your reasoning.

        Clearly explain your reasoning, and then finish your answer with "The final answer is (X)."  Here, X must be the correct letter choice.

        Documents: 
        {documents} 
        
        Question: {problem['question']} 
        Answer Choices: {problem['choices']['text']}     

        Solution: 
        """
        try:
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question:") or line.startswith("Documents:"):
                    response = "\n".join(response.split("\n")[:i])
                    break
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""

    def get_wiki_search_results(self, query, num_results=10):
        """Get search results using DPR and FAISS"""
        try:    
            # Convert query to tensor with proper truncation
            inputs = self.q_tokenizer(
                query, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512  # Explicitly set max length to 512
            )
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
                letters = re.findall(r'[A-J]', answer)
                if letters:
                    return letters[0]
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

            End your response with "End of generation" after you answer the instructions.

            Search Query: {query}
            Explanation:
            """
            explanation = self.generate_text(prompt, max_length=100)
            if explanation == None:
                return ""
            else:
                for i, line in enumerate(explanation.split("\n")):
                    line = line.strip()
                    if line.startswith("Search Query:"):
                        return "\n".join(explanation.split("\n")[:i])
                    if line.startswith("End of generation"):
                        return "\n".join(explanation.split("\n")[:i])
                    elif "End of generation" in line:
                        return "\n".join(explanation.split("\n")[:i+1])
                return explanation

        elif query_mode == "q2":
            prompt = f"""
            You are given a subquestion and a search query.
            The search query is a realistic phrase that someone might use to find knowledge or reasoning support to answer the subquestion.

            Your task is to anticipate what essential scientific or mathematical explanation the search result would contain, and write it concisely (2–3 sentences).
            Focus only on the core concept or principle that would help answer the subquestion.
            Avoid restating the subquestion, and do not include unrelated or overly general information.
            
            End your response with "End of generation" after you answer the instructions.
            
            Subquestion: {subquestion}
            Search Query: {query}

            Explanation:
            """
            explanation = self.generate_text(prompt, max_length=100)
            if explanation == None:
                return ""
            else:
                for i, line in enumerate(explanation.split("\n")):
                    line = line.strip()
                    if line.startswith("Search Query:"):
                        return "\n".join(explanation.split("\n")[:i])
                    if line.startswith("End of generation"):
                        return "\n".join(explanation.split("\n")[:i])
                    elif "End of generation" in line:
                        return "\n".join(explanation.split("\n")[:i+1])
                return explanation


        elif query_mode == "q3":
            prompt = f"""
            You are an expert in Science. You are given a science problem. Your task is to write a realistic search query that would help someone find the scientific concepts, principles, or methods needed to solve the problem. The query should reflect how someone might search online to understand and solve the question.

            Here are a few examples:

            Question: A block slides down a frictionless incline. What is its acceleration?
            Search Query: how to calculate acceleration on frictionless inclined plane

            Question: What is the pH of a 0.01 M HCl solution?
            Search Query: how to find pH of strong acid solution using concentration

            Question: During cellular respiration, what is the role of the mitochondria?
            Search Query: mitochondria function in ATP production during aerobic respiration

            Question: Why does Earth experience seasons?
            Search Query: how earth’s axial tilt causes seasons

            Question: A Punnett square is used to cross two heterozygous parents. What is the expected genotype ratio?
            Search Query: punnett square for heterozygous cross genotype probability

            Question: What causes tectonic plates to move?
            Search Query: causes of tectonic plate movement and mantle convection

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
            You are an expert at Science. You are given a Science problem. Your task is to extract the Science concepts and principles involved in solving the problem.
            What are the principles behind this question? 

            End your response with "End of generation" after you answer the instructions.

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
                    if line.startswith("End of generation"):
                        return "\n".join(response.split("\n")[:i])
                    elif "End of generation" in line:
                        return "\n".join(response.split("\n")[:i+1])
                return response

        elif query_mode == "step-hyde":
            prompt = f"""
            You are an expert at Science. You are given a Science problem. Your task is to extract the Science concepts and principles involved in solving the problem.

            Question: {subquestion}
            Principles Involved:
            """
            response = self.generate_text(prompt, max_length=100)
            prompt = f"""
            Generate a paragraph that explains the principle.

            Principles: {response}

            Explanation: 
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

            End your response with "End of generation" after you answer the instructions.

            Question: {subquestion}
            Explanation: 
            """
            response = self.generate_text(prompt, max_length=100)
            if response == None:
                return ""
            else:
                for i, line in enumerate(response.split("\n")):
                    line = line.strip()
                    if line.startswith("Question:"):
                        return "\n".join(response.split("\n")[:i])
                    if line.startswith("End of generation"):
                        return "\n".join(response.split("\n")[:i])
                    elif "End of generation" in line:
                        return "\n".join(response.split("\n")[:i+1])
                return response
        else:
            raise ValueError(f"Invalid query mode: {query_mode}. Query mode should be one of subq, q0, q1, q2, q3 hyde, step-back")


    def generate_subquestions(self, problem):
        if self.prompt:
            prompt = f"""
        You are given a multiple-choice question. Your task is to break it down into a step-by-step reasoning process by generating essential **subquestions** and corresponding **search queries**.

Each **subquestion** should be a concrete reasoning step that brings you closer to solving the original question. Think of it as an intermediate goal that must be answered—something a student would actually write down or solve while working through the problem.

Each **search query** should express what someone would realistically search for to find the general concept, principle, or scientific knowledge needed to solve the corresponding subquestion. Avoid using specific values, phrases, or details from the original question unless absolutely necessary.

The goal is to simulate how a student would think through the problem, identifying what must be solved and what knowledge is needed to solve it.

---

STRICT FORMAT REQUIREMENTS:
1. For each subquestion, provide exactly two parts in this order:
   - The subquestion (a concrete intermediate problem to be solved)
   - The search query (general knowledge needed to solve the subquestion)

2. Use EXACTLY this format for each subquestion:
Subquestion 1: [Concrete reasoning step—may include values or variables from the original question]  
Search Query for Subquestion 1: [General, reusable query that reflects background knowledge for solving this step]

---

EXAMPLE:

Question: An unknown substance is found to have a high melting point. In addition, it is a poor conductor of electricity and does not dissolve in water. The substance most likely contains  
Answer Choices: ['(A) dipole-dipole bonding', '(B) ionic bonding', '(C) covalent network bonding', '(D) nonpolar covalent bonding', '(E) coordinate covalent bonding', '(F) London dispersion bonding', '(G) van der Waals bonding', '(H) metallic bonding', '(I) hydrogen bonding', '(J) polar covalent bonding']

Subquestion 1: Which bonding types are typically associated with high melting points?  
Search Query for Subquestion 1: bonding types with high melting points

Subquestion 2: Of those bonding types, which are poor conductors of electricity in solid and liquid state?  
Search Query for Subquestion 2: electrical conductivity of solids by bonding type

Subquestion 3: Which of the remaining bonding types are typically insoluble in water?  
Search Query for Subquestion 3: solubility of different bonding types in water

Subquestion 4: Which bonding type satisfies all three conditions: high melting point, poor conductivity, and insolubility?  
Search Query for Subquestion 4: properties of covalent network solids

---

Now generate subquestions and search queries for the following question:

Question: {problem['question']}  
Answer Choices: {problem['choices']['text']}

        """
        else:
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
            response = self.generate_text(prompt, max_length=500)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question:"):
                    response = "\n".join(response.split("\n")[:i])
                    break
                if line.startswith("End of generation"):
                    response = "\n".join(response.split("\n")[:i])
                    break
                elif "End of generation" in line:
                    response = "\n".join(response.split("\n")[:i+1])
                    break
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
        Each search query should reflect scientific or mathematical knowledge needed to answer the subquestion.

        End your response with "End of generation" after you answer the instructions.

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
                if line.startswith("End of generation"):
                    return "\n".join(response.split("\n")[:i])
                elif "End of generation" in line:
                    return "\n".join(response.split("\n")[:i+1])
            return response

    def solve_subquestion_base(self, problem, step_num, subquestions, subsolutions):
        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve

        Your task:
        - Carefully read the original question, any previous subquestion and their solutions, and the current subquestion
        - Your solution should be detailed and logically structured
            
        End your response with "End of generation" after you answer the instructions.    

        Question: {problem['question']}
        Answer Choices: {problem['choices']['text']}
        """
        if step_num > 1:
            prompt += "Previous subquestions and their solutions:\n"
            for i in range(step_num - 1):
                prompt += f"""
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        Current subquestion to solve:
        Subquestion {step_num}: {subquestions[step_num-1]}
        Subquestion {step_num} Solution:
        """

        try:
            response = self.generate_text(prompt, max_length=600)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion") and ":" in line:
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("Question:"):
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("End of generation"):
                    return "\n".join(response.split("\n")[:i])
                elif "End of generation" in line:
                    return "\n".join(response.split("\n")[:i+1])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""

    def solve_subquestion_step_back(self, problem, step_num, subquestions, subsolutions):
        prompt = f"""
            You are an expert at Science. You are given a Science problem. Your task is to extract the Science concepts and principles involved in solving the problem.
            What are the principles behind this question? 

            End your response with "End of generation" after you answer the instructions.

            Question: {subquestions[step_num-1]}
            Principles Involved:
            """
        try:
            principles = self.generate_text(prompt, max_length=100)
            for i, line in enumerate(principles.split("\n")):
                line = line.strip()
                if line.startswith("Question:"):
                    principles = "\n".join(principles.split("\n")[:i])
                    break
                if line.startswith("End of generation"):
                    principles = "\n".join(principles.split("\n")[:i])
                    break
                elif "End of generation" in line:
                    principles = "\n".join(principles.split("\n")[:i+1])
                    break
        except Exception as e:
            print(f"Error in solve_subquestion_step_back: {e}")
            principles = ""
            
        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Principles involved in solving the problem

        Your task is to solve the current subquestion.
        End your response with "End of generation" after you answer the instructions.

        Question: {problem['question']}
        Answer Choices: {problem['choices']['text']}
        """
        if step_num > 1:
            prompt += "Previous subquestions and their solutions:\n"
            for i in range(step_num - 1):
                prompt += f"""
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        Current subquestion to solve:
        Subquestion {step_num}: {subquestions[step_num-1]}
        Principles: {principles}
        Subquestion {step_num} Solution:
        """

        try:
            response = self.generate_text(prompt, max_length=600)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion") and ":" in line:
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("Question:"):
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("End of generation"):
                    return "\n".join(response.split("\n")[:i])
                elif "End of generation" in line:
                    return "\n".join(response.split("\n")[:i+1])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_step_back: {e}")
            return ""

    def solve_subquestion_with_docs(self, problem, step_num, subquestions, subsolutions, documents):
        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Documents that are relevant to the current subquestion
        
        Your task:
        - Carefully read the original question, any previous subquestion and their solutions, and the current subquestion
        - Use the information from the retrieved documents to solve the current subquestion
        - Also use your existing knowledge to solve the current subquestion
        - Your solution should be detailed and logically structured

        End your response with "End of generation" after you answer the instructions.

        Documents:
        {documents}


        Question: {problem['question']}
        Answer Choices: {problem['choices']['text']}
        """
        if step_num > 1:
            prompt += "Previous subquestions and their solutions:\n"
            for i in range(step_num - 1):
                prompt += f"""
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}\n
                """

        prompt += f"""
        Current subquestion to solve:
        Subquestion {step_num}: {subquestions[step_num-1]}
        Subquestion {step_num} Solution:
        """
        

        try:
            response = self.generate_text(prompt, max_length=600)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion") and ":" in line:
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("Question:"):
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("End of generation"):
                    return "\n".join(response.split("\n")[:i])
                elif "End of generation" in line:
                    return "\n".join(response.split("\n")[:i+1])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_with_docs: {e}")
            return ""

    def solve_subquestion_with_ex(self, problem, step_num, subquestions, subsolutions, examples):
        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve
            4. Examples that are relevant to the current subquestion

        Your task is to solve the current subquestion.

        End your response with "End of generation" after you answer the instructions.

        Relevant Examples:
        {examples}


        Question: {problem['question']}
        Answer Choices: {problem['choices']['text']}
        """
        if step_num > 1:
            prompt += "Previous subquestions and their solutions:\n"
            for i in range(step_num - 1):
                prompt += f"""
                Subquestion {i+1}: {subquestions[i]}
                Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        Current subquestion to solve:
        Subquestion {step_num}: {subquestions[step_num-1]}
        Subquestion {step_num} Solution:
        """
        try:
            response = self.generate_text(prompt, max_length=600)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()                
                if line.startswith("Subquestion") and ":" in line:
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("Question:"):
                    return "\n".join(response.split("\n")[:i])
                if line.startswith("End of generation"):
                    return "\n".join(response.split("\n")[:i])
                elif "End of generation" in line:
                    return "\n".join(response.split("\n")[:i+1])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_with_ex: {e}")
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
        Question: {problem['question']}
        Answer Choices: {problem['choices']['text']}

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
        You are solving a multiple-choice question. Question is decomposed into several subquestions. Each subquestion has already been solved. Your task is to carefully read the original question and the several subquestion solutions, then use them to determine the final answer. Think step by step and then finish your answer with "The final answer is (X)" where X is the correct letter choice.

        Original Question:
        Question: {problem['question']}
        Answer Choices: {problem['choices']['text']}

        Subquestions and Solutions:
        """
        for i, (subquestion, subsolution) in enumerate(zip(subquestions, subsolutions)):
            prompt += f"""
            Subquestion {i+1}: {subquestion}
            Subquestion {i+1} Solution: {subsolution}
            """

        try:
            prompt += "\n\nFinal Solution: "
            response = self.generate_text(prompt, max_length=700)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Question:"):
                    response = "\n".join(response.split("\n")[:i])
                    break
            return response
        except Exception as e:
            print(f"Error in generate_final_answer: {e}")
            return ""

    def solve_problem(self, subject, prob_num, problem=None):
        base_dir = f"/home/minhae/multihop-RAG2/final-results/{subject}-0512/llama"
        self.output_dir = os.path.join(base_dir, f"{self.config_str}",self.timestamp)
        if prob_num == 1:
            os.makedirs(self.output_dir, exist_ok=True)
            # Initialize wandb if not already initialized
            if self.run is None and self.wandb_api_key:
                os.environ['WANDB_API_KEY'] = self.wandb_api_key
                self.run = wandb.init(
                    project=f"{subject}-llama",
                    entity=self.wandb_entity,
                    name=f"{self.config_str}_{self.timestamp}",
                    config = {
                        "mode" : self.mode,
                        "query_mode" : self.query_mode,
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
                'total_problems': 1,
                'correct_problems': 0
            }
        else:
            self._domain_stats[problem['domain']]['total_problems'] += 1

        if problem['h_domain'] not in self._h_domain_stats:
            self._h_domain_stats[problem['h_domain']] = {
                'total_problems': 1,
                'correct_problems': 0
            }
        else:
            self._h_domain_stats[problem['h_domain']]['total_problems'] += 1

        if problem['difficulty_1'] not in self._difficulty_stats['ev1']:
            self._difficulty_stats['ev1'][problem['difficulty_1']] = {
                'total_problems': 1,
                'correct_problems': 0
            }
        else:
            self._difficulty_stats['ev1'][problem['difficulty_1']]['total_problems'] += 1

        if problem['difficulty_2'] not in self._difficulty_stats['ev2']:
            self._difficulty_stats['ev2'][problem['difficulty_2']] = {
                'total_problems': 1,
                'correct_problems': 0
            }
        else:
            self._difficulty_stats['ev2'][problem['difficulty_2']]['total_problems'] += 1

        if problem['difficulty_w'] not in self._difficulty_stats['writer']:
            self._difficulty_stats['writer'][problem['difficulty_w']] = {
                'total_problems': 1,
                'correct_problems': 0
            }
        else:
            self._difficulty_stats['writer'][problem['difficulty_w']]['total_problems'] += 1
                
        # Add timing dictionary
        start_total = time.time()

        if self.mode == "zeroshot_COT":
            final_solution = self.solve_with_zeroshot_COT(problem)
            file_print(f"Solution: {final_solution}")

        elif self.mode == "zeroshot_PS":
            final_solution = self.solve_with_zeroshot_PS(problem)
            file_print(f"Solution: {final_solution}")

        elif self.mode == "zeroshot_RAG":
            search_query = ""
            if self.query_mode == "q1":
                search_query = self.generate_search_query(problem['question'])
            query = self.generate_query(problem['question'], self.query_mode, search_query)
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
        elif self.mode == "step_back":
            final_solution, principles = self.solve_with_step_back(problem)
            file_print(f"Principles: {principles}")
            file_print(f"Solution: {final_solution}")
            

        else:
            file_print(f"Stage 1: Generate subquestions and search queries")
            subquestions, queries = self.generate_subquestions(problem)
            file_print(f"There are {len(subquestions)} subquestions and search queries")
            file_print(f"Subquestions: {subquestions}")
            file_print(f"Search Queries: {queries}")
            file_print("--------------------------------")
            
            if len(subquestions) == 0:
                final_solution = self.solve_with_zeroshot_COT(problem)
                file_print(f"Solution: {final_solution}")
            else:
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
                    elif self.mode == "PD_step_back":
                        solution = self.solve_subquestion_step_back(problem, step_num, subquestions, solutions)
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
                        raise ValueError(f"Invalid mode: {self.mode}. Mode should be one of zeroshot_COT, zeroshot_PS, zeroshot_RAG,step_back, PD, Step_RAG, Step_RAG_EX")
                file_print("--------------------------------")
                file_print(f"\n\nStage 3: Generate final answer")
                final_solution = self.generate_final_answer(problem, subquestions, solutions)
                file_print(f"Final Solution: {final_solution}")
        
        predicted_answer = self.extract_answer(final_solution)
        if predicted_answer != "None":
            if str(predicted_answer) == str(correct_answer):
                if hasattr(self, '_correct_problems'):
                    self._correct_problems += 1
                    result = "Correct"
                    correct = True
                else:
                    self._correct_problems = 1
                    result = "Correct"
                    correct = True
            else:
                result = "Wrong"
                correct = False
        else: 
            result = "Wrong"
            correct = False
        
        if hasattr(self, '_total_problems'):
            self._total_problems += 1
        else:
            self._total_problems = 1
        
        self._current_accuracy = self._correct_problems / self._total_problems *100

        if correct:
            self._domain_stats[problem['domain']]['correct_problems'] += 1
            self._h_domain_stats[problem['h_domain']]['correct_problems'] += 1
            self._difficulty_stats['ev1'][problem['difficulty_1']]['correct_problems'] += 1
            self._difficulty_stats['ev2'][problem['difficulty_2']]['correct_problems'] += 1
            self._difficulty_stats['writer'][problem['difficulty_w']]['correct_problems'] += 1
            
        self._accuracy_stats['domain'][problem['domain']] = (self._domain_stats[problem['domain']]['correct_problems'] / self._domain_stats[problem['domain']]['total_problems']) * 100
        self._accuracy_stats['h_domain'][problem['h_domain']] = (self._h_domain_stats[problem['h_domain']]['correct_problems'] / self._h_domain_stats[problem['h_domain']]['total_problems']) * 100
        self._accuracy_stats['difficulty_ev1'][problem['difficulty_1']] = (self._difficulty_stats['ev1'][problem['difficulty_1']]['correct_problems'] / self._difficulty_stats['ev1'][problem['difficulty_1']]['total_problems']) * 100
        self._accuracy_stats['difficulty_ev2'][problem['difficulty_2']] = (self._difficulty_stats['ev2'][problem['difficulty_2']]['correct_problems'] / self._difficulty_stats['ev2'][problem['difficulty_2']]['total_problems']) * 100
        self._accuracy_stats['difficulty_writer'][problem['difficulty_w']] = (self._difficulty_stats['writer'][problem['difficulty_w']]['correct_problems'] / self._difficulty_stats['writer'][problem['difficulty_w']]['total_problems']) * 100                    

        self.run.log({
            "total_problems": self._total_problems,
            "correct_problems": self._correct_problems,
            "current_accuracy": self._current_accuracy
        })
        if final_solution is not None:
            # Save detailed results to the detailed file
            #summary_print(f"Final Solution: {final_solution}")
            summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {self._current_accuracy:^11.2f}%")
            for key, value in self._domain_stats.items():
                summary_print(f"Accuracy of {key}:")
                summary_print(f"{value['correct_problems']}/{value['total_problems']} = {value['correct_problems']/value['total_problems']*100:.2f}%")
            summary_print("-" * 50)
            for key, value in self._h_domain_stats.items():
                summary_print(f"Accuracy of {key}:")
                summary_print(f"{value['correct_problems']}/{value['total_problems']} = {value['correct_problems']/value['total_problems']*100:.2f}%")
            summary_print("-" * 50)
            for key, value in self._difficulty_stats['ev1'].items():
                summary_print(f"Accuracy of {key}:")
                summary_print(f"{value['correct_problems']}/{value['total_problems']} = {value['correct_problems']/value['total_problems']*100:.2f}%")
            summary_print("-" * 50)
            for key, value in self._difficulty_stats['ev2'].items():
                summary_print(f"Accuracy of {key}:")
                summary_print(f"{value['correct_problems']}/{value['total_problems']} = {value['correct_problems']/value['total_problems']*100:.2f}%")
            summary_print("-" * 50)
            for key, value in self._difficulty_stats['writer'].items():
                summary_print(f"Accuracy of {key}:")
                summary_print(f"{value['correct_problems']}/{value['total_problems']} = {value['correct_problems']/value['total_problems']*100:.2f}%")
            summary_print("-" * 50)
                
            
            file_print(f"Predicted Answer: {predicted_answer}")
            file_print(f"Actual Answer: {correct_answer}")
            file_print(f"Running Accuracy: {self._current_accuracy:.2%}")
        # Print timing statistics
        total_time = time.time() - start_total
        file_print(f"Total Time: {total_time:.2f} seconds")


                