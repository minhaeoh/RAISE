from datasets import load_dataset

self.ds = load_dataset("facebook/wiki_dpr", 'psgs_w100.nq.compressed', cache_dir='/home/minhae/.cache/')

