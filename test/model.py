import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM)
from mistralai import Mistral
from vllm import LLM, SamplingParams
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

class Model:
    def __init__(self, model_name):
        env_path = "/path/to/.env"
        load_dotenv(dotenv_path=env_path)
        self.hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if 'llama' in model_name:
            self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif 'mistral' in model_name:
            self.model_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
        elif 'gpt' in model_name:
            self.model_name = "gpt-4o-mini"
        else:
            self.model_name = model_name
        self.load_model()
        
    def load_model(self):
        if self.model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
            self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            try:
                self.model = LLM(
                        model=self.model_name,
                        dtype="float16", 
                        trust_remote_code=True,
                        tensor_parallel_size=self.num_gpus  # Use available GPU count
                    )
                print("Successfully initialized Mistral model")
            except Exception as e:
                print(f"Error in loading Mistral model: {e}")
                raise
        elif self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    token=self.hf_token
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "left"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    token=self.hf_token
                )
                self.model.eval()
                print("Successfully initialized Llama model")
            except Exception as e:
                print(f"Error in loading Llama model: {e}")
                raise
        elif self.model_name == "gpt-4o-mini":
            try:
                self.openai_api_key = os.getenv('OPENAI_API_KEY')
                self.model = OpenAI(api_key=self.openai_api_key)
                print("Successfully initialized GPT-4o-mini model")
            except Exception as e:
                print(f"Error in loading GPT-4o-mini model: {e}")
                raise
    
    def generate_text(self, prompt, max_length=384, temperature=0.15):
        if self.model_name == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
            sampling_params = SamplingParams(
                max_tokens=max_length,
                temperature=temperature
            )
            response = self.model.generate([prompt], sampling_params=sampling_params)
            response = response[0].outputs[0].text
            return response
        elif self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
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
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
            )
            output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            response = output_text[len(prompt):].strip()
            lines = []
            for i, line in enumerate(response.split("\n")):
                line = line.strip()
                if not line:
                    continue
                if 'the final answer is' in line.lower():
                    line = line.split(".")[0]
                    lines.append(line)
                    break
                elif 'python' in line.lower():
                    break
                else:
                    lines.append(line)
            response = "\n".join(lines)
            response = re.sub(r"(\|\s*){3,}", "", response).strip()
            return response
            
        elif self.model_name == "gpt-4o-mini":
            response = self.model.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an evaluator for model-generated answers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_length
            )
            return response.choices[0].message.content

def main():
    test_models = [
        "mistral",
        "llama",
        "gpt"
    ]
    
    # 테스트할 프롬프트들
    test_prompts = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms."
    ]
    
    for model_type in test_models:
        print(f"\n--- Testing {model_type.upper()} Model ---")
        try:
            model = Model(model_type)
            print(f"Model initialized: {model.model_name}")
            
            for i, prompt in enumerate(test_prompts, 1):
                print(f"\nTest {i}: {prompt}")
                print("-" * 50)
                
                try:
                    response = model.generate_text(
                        prompt=prompt,
                        max_length=200,  
                        temperature=0.7
                    )
                    
                    print(f"Response: {response}")
                    print(f"Response length: {len(response)} characters")
                    
                except Exception as e:
                    print(f"Error generating text: {e}")
                    
        except Exception as e:
            print(f"Error initializing {model_type} model: {e}")
            continue
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    main()
        
        