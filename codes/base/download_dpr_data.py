import os
import logging
from datasets import load_dataset
import requests
import gzip
from tqdm import tqdm

# URLs for DPR data
_INDEX_URL = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr"
_DATA_URL = "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"

def download_file(url, local_path):
    """Download file with progress bar"""
    if os.path.exists(local_path):
        logging.info(f"File already exists: {local_path}")
        return
    
    logging.info(f"Downloading {url} to {local_path}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
        desc=os.path.basename(local_path),
        total=total_size,
        unit='iB',
        unit_scale=True
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def setup_dpr_data(data_dir=None): 
    """Download and setup DPR data
    
    Args:
        data_dir (str): Directory to store the downloaded files
        
    Returns:
        bool: True if setup was successful
    """
    if data_dir is None:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Create path relative to the script location
        data_dir = os.path.join(script_dir, "wikipedia", "dpr")

    logging.info("Starting DPR data setup...")
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # 1. Download Wikipedia passages
        passages_path = os.path.join(data_dir, "psgs_w100.tsv.gz")
        if not os.path.exists(passages_path):
            download_file(_DATA_URL, passages_path)
        
        # 2. Download pre-computed FAISS index
        index_name = "psgs_w100.nq.exact.HNSW128_SQ8-IP-train.faiss"
        index_url = f"{_INDEX_URL}/{index_name}"
        index_path = os.path.join(data_dir, "psgs_w100.nq.exact.faiss")
        
        if not os.path.exists(index_path):
            logging.info("Downloading pre-computed FAISS index...")
            download_file(index_url, index_path)
        
        logging.info("DPR data setup completed successfully!")
        return True
        
    except Exception as e:
        logging.error(f"Error in DPR data setup: {e}")
        raise

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run DPR data setup
    setup_dpr_data() 