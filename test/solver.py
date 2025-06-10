import os
import re
import time
from datasets import load_dataset
import torch
from transformers import (
    DPRQuestionEncoderTokenizer, 
    DPRQuestionEncoder,
    DPRContextEncoderTokenizer,
    DPRContextEncoder
)
from model import Model

class MultiHopSolver:
    def __init__(self, mode="Step_RAG",trigger=False, trigger_value=0.8, doc_num=10, query_mode="q1",model_name="mistralai/Mistral-Small-3.1-24B-Instruct-2503"):
        # Initialize counters for tracking overall performance
        self._total_problems = 0
        self._correct_problems = 0

        self.mode = mode
        self.model_name = model_name
        
        self.trigger = None
        self.trigger_value = None
        self.doc_num = None
        self.query_mode = None

        # Initialize timestamp for file naming
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")

        # configuration details to directory name
        if self.mode == "Direct_COT":
            config_str = "Direct_COT"
        elif self.mode == "Direct_PS":
            config_str = "Direct_PS"
        elif self.mode == "step-back":
            config_str = "step-back"
        elif self.mode == "Direct_RAG":
            self.query_mode = query_mode
            self.trigger = trigger
            self.doc_num = doc_num
            config_str = "Direct_RAG"
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
            self.trigger = trigger
            self.doc_num = doc_num
            self.query_mode = query_mode
            config_str = self.mode
            config_str += f"_{self.query_mode}"
            if self.trigger:
                self.trigger_value = float(trigger_value)
                config_str += f"_trigger{self.trigger_value}"
            config_str += f"_doc{self.doc_num}"
        self.config_str = config_str

        # Set device for CUDA if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")


        # Initialize model
        self.model = Model(self.model_name)

        # Initialize DPR components
        if self.mode == "Direct_RAG" or self.mode == "Step_RAG":
            print("Setting up Retrieval System...")
            self.setup_wiki_retriever()
        

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
        return self.model.generate_text(prompt, max_length=max_length, temperature=temperature)


    def solve_with_Direct_COT(self, problem):
        prompt = f"""You are solving a multiple choice question. Think step by step and show your reasoning clearly.  
        At the end, state your answer in the format: "The final answer is (X)".
        Here, X must be the correct letter choice.

        Question: {problem['question']} 
        Answer Choices: {problem['choices']['text']}
        """
        try:
            response = self.generate_text(prompt, max_length=1500)
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""

    def solve_with_step_back(self, problem):
        prompt = f"""
            You are an expert at Science. You are given a Science problem. Your task is to extract the Science concepts and principles involved in solving the problem.

            Question: {problem['question']}
            Answer Choices: {problem['choices']['text']}
            Principles Involved:
            """
        try:
            principles = self.generate_text(prompt, max_length=100)
            prompt = f"""You are an expert at Science. You are given a Science problem and a set of principles involved in solving the problem. Solve the problem step by step by folowwing the principles.
        At the end, state your answer in the format: "The final answer is (X)".
        Here, X must be the correct letter choice.

        Question: {problem['question']} 
        Principles: {principles}
        Answer Choices: {problem['choices']['text']}
        """
            response = self.generate_text(prompt, max_length=1500)
            return response, principles
        except Exception as e:
            print(f"Error in solve_with_step_back: {e}")
            return ""
        
    def solve_with_Direct_PS(self, problem):
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
        """
        try:
            response = self.generate_text(prompt, max_length=1500)
            return response
        except Exception as e:
            print(f"Error in solve_with_zeroshot: {e}")
            return ""

    def solve_with_Direct_RAG(self, problem, documents):
        prompt = f"""
        You are given a multiple choice question and a set of related documents.

        Use the information from the documents to help solve the problem step by step. If the documents contain useful facts, definitions, or formulas, apply them in your reasoning.

        Clearly explain your reasoning, and then finish your answer with "The final answer is (X)."  Here, X must be the correct letter choice.

        Documents: 
        {documents} 
        
        Question: {problem['question']} 
        Answer Choices: {problem['choices']['text']}      
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
                letters = re.findall(r'[A-D]', answer)
                if letters:
                    return letters[0]
        return "None"

    def generate_query(self, subquestion, query_mode, query):
        if query_mode == "subq":
            return subquestion
        elif query_mode == "q0" :
            return query
        elif query_mode == "RAISE":
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
            raise ValueError(f"Invalid query mode: {query_mode}. Query mode should be one of subq, raise, hyde, step-back")


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

    def solve_subquestion_base(self, problem, step_num, subquestions, subsolutions):
        prompt = f"""
        You are solving a multiple-choice question. Question is decomposed into several subquestions. You will be given:
            1. The original multiple-choice question
            2. Previous subquestions and their solutions (if any)
            3. The current subquestion to solve

        Your task is to solve the current subquestion.

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
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion") and i>0:
                    return "\n".join(response.split("\n")[:i])
            return response
        except Exception as e:
            print(f"Error in solve_subquestion_base: {e}")
            return ""

    def solve_subquestion_step_back(self, problem, step_num, subquestions, subsolutions):
        prompt = f"""
            You are an expert at Science. You are given a Science problem. Your task is to extract the Science concepts and principles involved in solving the problem.

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
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion") and i>0:
                    return "\n".join(response.split("\n")[:i])
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
        Your task is to solve the current subquestion.

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
                Subquestion {i+1} Solution: {subsolutions[i]}
                """

        prompt += f"""
        Current subquestion to solve:
        Subquestion {step_num}: {subquestions[step_num-1]}
        Subquestion {step_num} Solution:
        """
        

        try:
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if line.startswith("Subquestion") and i>0:
                    return "\n".join(response.split("\n")[:i])
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
            response = self.generate_text(prompt, max_length=1000)
            for i, line in enumerate(response.split("\n")):
                line = line.strip()                
                if line.startswith("Subquestion") and i>0:
                    return "\n".join(response.split("\n")[:i])
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
            response = self.generate_text(prompt, max_length=1500)
            return response
        except Exception as e:
            print(f"Error in generate_final_answer: {e}")
            return ""

    def solve_problem(self, prob_num, problem=None):
        base_dir = f"/path/to/your/output"
        self.output_dir = os.path.join(base_dir, f"{self.config_str}",self.timestamp)
        if prob_num == 1:
            os.makedirs(self.output_dir, exist_ok=True)

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
            correct_answer = problem['answerKey']
            print_func("Correct Answer: {}".format(correct_answer))
            print_func("--------------------------------")

                
        # Add timing dictionary
        start_total = time.time()

        if self.mode == "Direct_COT":
            final_solution = self.solve_with_Direct_COT(problem)
            file_print(f"Solution: {final_solution}")

        elif self.mode == "Direct_PS":
            final_solution = self.solve_with_Direct_PS(problem)
            file_print(f"Solution: {final_solution}")

        elif self.mode == "Direct_RAG":
            search_query = ""
            if self.query_mode == "q1":
                search_query = self.generate_search_query(problem['question'])
            query = self.generate_query(problem['question'], self.query_mode, search_query)
            documents, select_num, distances = self.get_wiki_search_results(query, self.doc_num)
            if select_num == 0:
                final_solution = self.solve_with_Direct_COT(problem)
            else:
                final_solution = self.solve_with_Direct_RAG(problem, documents)
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
                final_solution = self.solve_with_Direct_COT(problem)
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

                    else:
                        raise ValueError(f"Invalid mode: {self.mode}. Mode should be one of Direct_COT, Direct_PS, Direct_RAG,step_back, PD, PD_step_back, Step_RAG")
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

        
        if final_solution is not None:
            # Save detailed results to the detailed file
            #summary_print(f"Final Solution: {final_solution}")
            summary_print(f"{prob_num:^10} | {result:^10} | {predicted_answer:^10} | {correct_answer:^10} | {self._current_accuracy:^11.2f}%")
            
            file_print(f"Predicted Answer: {predicted_answer}")
            file_print(f"Actual Answer: {correct_answer}")
            file_print(f"Running Accuracy: {self._current_accuracy:.2%}")
        # Print timing statistics
        total_time = time.time() - start_total
        file_print(f"Total Time: {total_time:.2f} seconds")


                