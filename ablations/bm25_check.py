import json
import os
from typing import List, Dict
from pathlib import Path

def process_document(doc: str) -> List[str]:
    """Find all phrases that start with [Title: and return them as a list."""
    title_phrases = []
    current_pos = 0
    
    while True:
        # Find the next occurrence of [Title:
        title_idx = doc.find("[Title:", current_pos)
        if title_idx == -1:
            break
            
        # Find the end of this title section (next [Title: or end of string)
        next_title = doc.find("[Title:", title_idx + 1)
        if next_title == -1:
            # If no next title, take until the end of the string
            title_phrases.append(doc[title_idx:])
            break
        else:
            # Take until the next title
            title_phrases.append(doc[title_idx:next_title])
            current_pos = next_title
    
    # If no titles found, return the original document as a single item
    if not title_phrases:
        return [doc]
        
    return title_phrases

def create_subquestion_document_pairs(data: List[Dict]) -> List[Dict]:
    """Create new pairs of (subquestion, document) from the input data."""
    new_pairs = []
    
    for item in data:
        if not isinstance(item, dict):
            continue
            
        subquestion = item.get('subquestion', '')
        document = item.get('document', '')
        
        if not subquestion or not document:
            continue
            
        # Process document to start from 'Title'
        processed_docs = process_document(document)
        for processed_doc in processed_docs:
            # Create new pair
            new_pair = {
                'subquestion': subquestion,
                'document': processed_doc
            }
            new_pairs.append(new_pair)
    
    return new_pairs

def process_json_file(input_path: str, output_path: str):
    """Process a single JSON file and save the results."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create new pairs
        new_pairs = create_subquestion_document_pairs(data)
        
        # Save to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(new_pairs, f, ensure_ascii=False, indent=2)
            
        print(f"Processed {input_path} -> {output_path}")
        print(f"Created {len(new_pairs)} pairs")
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")

def main():
    # Define input and output directories
    input_dir = '/home/minhae/multihop-RAG2/ablation/gpqa/llama/step-back.json'
    
    output_dir = '/home/minhae/multihop-RAG2/ablation/gpqa/llama/cosine'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'step-back.json')
    
    # Process all JSON files in the input directories
    
    if not os.path.exists(input_dir):
        print(f"Directory not found: {input_dir}")

                
   
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Process the file
    process_json_file(str(input_dir), output_path)

if __name__ == "__main__":
    main()
