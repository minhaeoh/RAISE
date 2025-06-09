import os
from huggingface_hub import hf_hub_download
import argparse

def download_wiki_files(repo_id, output_dir, filename):
    """
    Download specific Wikipedia file from Hugging Face Hub
    
    Args:
        repo_id (str): Hugging Face repository ID (e.g., "username/repo-name")
        output_dir (str): Local directory to save the files
        filename (str): Specific file to download
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {filename} from {repo_id} to {output_dir}")
    
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            force_download=True,
            repo_type="dataset"
        )
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Download specific Wikipedia file from Hugging Face Hub')
    parser.add_argument('--repo_id', type=str, default="facebook/wiki_dpr",
                      help='Hugging Face repository ID (e.g., "username/repo-name")')
    parser.add_argument('--output_dir', type=str, default="/data/minhae/huggingface/datasets/facebook___wiki_dpr/psgs_w100.nq.exact/0.0.0/66fd9b80f51375c02cd9010050e781ed3e8f759e868f690c31b2686a7a0eeb5c",
                      help='Local directory to save the files')
    parser.add_argument('--filename', type=str, default="index/psgs_w100.nq.HNSW128_SQ8-IP-train.faiss",
                      help='Specific file to download')
    
    args = parser.parse_args()
    
    try:
        download_wiki_files(args.repo_id, args.output_dir, args.filename)
        print("\nFile downloaded successfully!")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main() 