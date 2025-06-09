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
        self._query_stats = {}
         
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
            config_str = "default"
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
            self.ds = load_dataset("facebook/wiki_dpr", 'psgs_w100.nq.exact',split='train',cache_dir="/home/minhae/multihop-RAG2/cache")
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

  


    def get_wiki_search_results(self, query, num_results=3):
        # """Get search results using DPR and FAISS"""
        # prompt = f"""
        # You are given a search query.
        # Your task is to explain everything you know about the topic based on the query.

        # End your response with "End of explanation."

        # Search Query: {query}
        # Explanation:
        # """
        
        # response = self.generate_text(prompt,prompt, max_length=100)
        # lines = []
        # for line in response.split("\n"):
        #     if "End of explanation".lower() in line.lower():
        #         line = line.lower().split("end of explanation")[0]
        #         lines.append(line)
        #         break
        #     else:
        #         lines.append(line)
        # response = "\n".join(lines)

        #print(f"Explanation: {response}")
        #response = "not making explanation"
        #response = f"{subject} explanation: {response}"
        try:
            print(f"\nDebug: Starting search for query: '{query}'")
            
            # Encode query
            print("Debug: Encoding query...")
            inputs = self.q_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
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
                return "No relevant documents found.", 0, distances[0]
            return "\n\n".join(documents), select_num, distances[0]
            
        except Exception as e:
            print(f"Error in wiki search: {e}")
            import traceback
            traceback.print_exc()
            return "No relevant documents found.", 0, distances[0]

    def solve_with_zeroshot(self, problem, subject):
        base_prompt = f"""Based on the context, determine whether the question is True or False.    
        Context: {problem['context']}
        Question: {problem['question']}

        Answer:
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
        self.timing_stats = {}
        start_total = time.time()

        # Initialize subject-specific counters if they don't exist


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
                file_print(f"Zero-shot Solution: {self.timing_stats['zeroshot_solution']:.2f} seconds")

            elif self.zeroshot_RAG:
                if 'retrieval' not in self.timing_stats:
                    self.timing_stats['retrieval'] = {}
                if 'solution' not in self.timing_stats:
                    self.timing_stats['solution'] = {}
                if 'query_generation' not in self.timing_stats:
                    self.timing_stats['query_generation'] = {}
            
                file_print("Solving with zeroshot-RAG approach...")
                queries = []
                documents  = []
                solutions = []
                select_nums = []
                queries.append(problem['question'])
                start = time.time()
                document, select_num = self.get_wiki_search_results(problem['question'], num_results=self.doc_num)
                self.timing_stats['retrieval']['question_retrieval'] = time.time() - start
                documents.append(document)
                select_nums.append(select_num)
                start = time.time()
                query = self.generate_search_query(problem['question'])
                queries.append(query)
                self.timing_stats['query_generation']['search_query_generation'] = time.time() - start
                start = time.time()
                document, select_num = self.get_wiki_search_results(query, num_results=self.doc_num)
                self.timing_stats['retrieval']['query_retrieval'] = time.time() - start
                documents.append(document)
                select_nums.append(select_num)
                start = time.time()
                query = self.generate_explanation_query(query)
                queries.append(query)
                self.timing_stats['query_generation']['explanation_query_generation'] = time.time() - start
                start = time.time()
                document, select_num = self.get_wiki_search_results(query, num_results=self.doc_num)
                self.timing_stats['retrieval']['explanation_retrieval'] = time.time() - start
                documents.append(document)
                select_nums.append(select_num)
                captions = ['question', 'query', 'explanation']
                for i in range(len(captions)):
                    start = time.time()
                    solution = self.solve_with_zeroshot_RAG(problem, documents[i])
                    self.timing_stats['solution'][f'{captions[i]}_solution'] = time.time() - start
                    solutions.append(solution)

               
                # Extract predicted answer - look for "final answer" in any case
                predicted_answers = [None] * len(captions)
                for i in range(len(captions)):
                    solution_lines = solutions[i].split('\n')
                    for line in solution_lines:
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
                    predicted_answers[i] = predicted_answer
                
                 # save method, query, documents, solution, and predicted answer
                for i in range(len(captions)):
                    file_print(f"Method: {captions[i]}")
                    file_print("-" * 10)
                    file_print(f"Query: \n{queries[i]}")
                    file_print("-" * 10)
                    file_print(f"Documents: \n{documents[i]}")
                    file_print("-" * 10)
                    file_print(f"Solution: \n{solutions[i]}")
                    file_print("-" * 10)
                    file_print(f"Predicted Answer: {predicted_answers[i]}")
                    file_print("=" * 50)
                
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

            
                
            # Print timing statistics
   
        except Exception as e:
            file_print(f"Error occurred: {e}")
            summary_print(f"{prob_num:^10} | {'Error':^10} | {'-':^10} | {problem['answer']:^10} | {'-':^11}")
            raise

        print(f"\nProcessing complete! Results saved to:")
        print(f"Detailed results: {detailed_file}")
        print(f"Summary results: {summary_file}")

    
    