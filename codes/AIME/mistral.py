from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
    local_dir="/data/minhae/models/mistral",
    local_dir_use_symlinks=False
)