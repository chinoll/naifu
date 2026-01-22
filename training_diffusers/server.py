"""
HTTP Metadata Server for SDXL Training (Robyn)

Hosts a simple API to serve dataset metadata and perform on-the-fly prompt generation (tag dropout).
Workers connect to this server to fetch batch information (latents path + processed prompt).

Usage:
    python training_diffusers/server.py --data_root /path/to/data --port 8000
"""

import argparse
import logging
import sys
import orjson
import random
import os
import json
from pathlib import Path
from typing import List, Dict, Any

from robyn import Robyn, Request, Response

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler(sys.stdout)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

app = Robyn(__file__)

# Global dataset instance
dataset = None

# Support 'robyn --dev' via environment variable
if os.environ.get("SDXL_DATA_ROOT"):
    try:
        data_root = os.environ.get("SDXL_DATA_ROOT")
        debug = os.environ.get("SDXL_DEBUG", "0") == "1"
        logger.info(f"Initializing dataset from env var SDXL_DATA_ROOT={data_root}")
        dataset = ServerDataset(data_root=data_root, debug=debug)
    except Exception as e:
        logger.error(f"Failed to load dataset from env: {e}")


def json_response(data, status_code=200):
    """Helper to return high-performance JSON response using orjson"""
    return Response(
        status_code=status_code,
        headers={"Content-Type": "application/json"},
        description=orjson.dumps(data)
    )


class ServerDataset:
    """Standalone dataset class for server to decouple from training code"""
    
    DEFAULT_TAG_DROPOUT_CONFIG = {
        "danbooru": {
            "character_drop": 0.1,
            "core_whole_drop": 0.5,
            "general_in_core_drop": 0.3,
            "general_not_core_drop": 0.2,
            "other_drop": 0.7,
        },
        "e621": {
            "copyright_drop": 0.5,
            "character_drop": 0.2,
            "species_drop": 0.2,
            "general_drop": 0.7,
            "e621_tag_drop": 0.2,
            "resolution_drop": 0.7,
            "nsfw_drop": 0.7,
        },
        "txt": {
            "tag_drop": 0.5,
        }
    }
    
    def __init__(self, data_root: str, debug: bool = False, dropout_config: Dict = None):
        self.data_root = data_root
        self.debug = debug
        self.dropout_config = dropout_config or self.DEFAULT_TAG_DROPOUT_CONFIG
        
        # Load dataset content
        self.reload()

    def reload(self):
        """Reload dataset files and metadata"""
        import pickle
        import time
        import gc
        
        cache_path = os.path.join(self.data_root, ".dataset_cache.pkl")
        
        # Try loading from cache first
        if os.path.exists(cache_path):
            try:
                start_time = time.time()
                logger.info(f"Found cache at {cache_path}, loading...")
                
                # Disable GC to speed up loading of millions of small objects
                gc.disable()
                try:
                    with open(cache_path, "rb") as f:
                        cached_data = pickle.load(f)
                finally:
                    gc.enable()
                
                self.buckets = cached_data["buckets"]
                self.metadata = cached_data["metadata"]
                self.files = cached_data["files"]
                self.bucket_indices = cached_data["bucket_indices"]
                
                logger.info(f"Loaded dataset from cache in {time.time() - start_time:.2f}s. "
                            f"Total files: {len(self.files)}, Buckets: {len(self.bucket_indices)}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, falling back to full scan.")
                # Re-enable GC just in case
                if not gc.isenabled():
                    gc.enable()
        
        logger.info(f"Scanning dataset from {self.data_root}...")
        start_time = time.time()
        
        # 1. Scan buckets
        latents_dir = os.path.join(self.data_root)#, "latents")
        if not os.path.exists(latents_dir):
             raise FileNotFoundError(f"Latents dir not found: {latents_dir}")

        from tqdm import tqdm
        self.buckets = {
            k: self._dirwalk(os.path.join(latents_dir, k)) 
            for k in tqdm(os.listdir(latents_dir))
            if os.path.isdir(os.path.join(latents_dir, k))
        }
        
        # 2. Load metadata
        self.metadata = {}
        if not self.debug:
            jsonl_path = os.path.join(self.data_root, "metadata.jsonl")
            if os.path.exists(jsonl_path):
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in tqdm(f):
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            self.metadata.update(record)
                logger.info(f"Loaded metadata from: {jsonl_path}")
            else:
                 logger.warning(f"No metadata file found: {jsonl_path}, tags will be empty unless debug=True")
        
        # 3. Build indices (Handling repeat)
        self.files = []
        self.bucket_indices = []
        
        for resolution, files in tqdm(self.buckets.items()):
            bucket_valid_indices = []
            for filepath in files:
                key = Path(filepath).stem
                
                # Check repeats
                repeat_count = 1
                if not self.debug and key in self.metadata:
                    meta = self.metadata[key]
                    meta_dict = meta.get('metadata', meta) if isinstance(meta.get('metadata'), dict) else meta
                    #for field_key, value in meta_dict.items():
                    #    if field_key.endswith('_repeat') and isinstance(value, (int, float)):
                    #        repeat_count = max(repeat_count, int(value))
                
                # Only add if we have metadata or if debugging
                if self.debug or key in self.metadata:
                    for _ in range(repeat_count):
                        bucket_valid_indices.append(len(self.files))
                        self.files.append(filepath)
            
            if bucket_valid_indices:
                self.bucket_indices.append(bucket_valid_indices)

        # Fallback if empty (prevent crash on client init)
        if not self.bucket_indices:
             self.bucket_indices = [list(range(len(self.files)))]
             
        logger.info(f"Dataset scanned in {time.time() - start_time:.2f}s. "
                    f"Total files: {len(self.files)}, Buckets: {len(self.bucket_indices)}")
        
        # Save cache
        try:
            logger.info(f"Saving cache to {cache_path}...")
            # Use HIGHEST_PROTOCOL for speed
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "buckets": self.buckets,
                    "metadata": self.metadata,
                    "files": self.files,
                    "bucket_indices": self.bucket_indices
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Cache saved.")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def update_config(self, new_config: Dict):
        """Update tag dropout configuration"""
        self.dropout_config.update(new_config)
        logger.info("Updated tag dropout configuration")

    def _dirwalk(self, path):
        path = Path(path)
        return [str(file) for file in path.rglob('*.npy') if file.is_file()]

    def __len__(self):
        return len(self.files)

    def get_batch_metadata(self, indices: List[int]):
        results = []
        for idx in indices:
            if idx < 0 or idx >= len(self.files):
                continue
                
            filepath = self.files[idx]
            
            if self.debug:
                prompt = ""
                original_size = (0, 0)
                dhdw = (0, 0)
                extras = None
            else:
                key = Path(filepath).stem
                meta = self.metadata[key]
                
                # Check for hot-loaded global hook
                prompt = None
                if "custom_prompt_hook" in globals() and callable(globals()["custom_prompt_hook"]):
                    try:
                        # Call global hook with (dataset_instance, meta)
                        prompt = globals()["custom_prompt_hook"](self, meta)
                    except Exception as e:
                        logger.error(f"Custom prompt hook failed: {e}")
                        prompt = None
                
                # Fallback to default
                if prompt is None:
                    prompt = self.apply_tag_dropout(meta)
                    
                original_size = meta['original_size']
                dhdw = meta['dhdw']
                extras = None
            
            results.append({
                "latent_path": filepath,
                "prompt": prompt,
                "original_size": original_size,
                "dhdw": dhdw,
                "extras": extras
            })
        return results

    def apply_tag_dropout(self, meta: dict) -> str:
        """Apply tag dropout to generate final prompt"""
        cfg_root = self.dropout_config
        
        if "metadata" in meta and isinstance(meta.get("metadata"), dict):
            t = meta["metadata"]
            data_src = meta.get("data_src", "danbooru")
            final_components = []
            
            if data_src == "e621":
                cfg = cfg_root.get("e621", self.DEFAULT_TAG_DROPOUT_CONFIG["e621"])
                
                for key, drop_key in [("copyright", "copyright_drop"), ("character", "character_drop")]:
                    val = t.get(key, [])
                    if val and random.random() >= cfg.get(drop_key, 0):
                        final_components.append(", ".join(val))
                
                for key, drop_key in [("species", "species_drop"), ("general", "general_drop")]:
                    val_list = t.get(key, [])
                    if val_list:
                        kept = [v for v in val_list if random.random() >= cfg.get(drop_key, 0)]
                        if kept:
                            final_components.append(", ".join(kept))
                
                for key, drop_key in [("e621_tag", "e621_tag_drop"), ("resolution", "resolution_drop"), ("nsfw", "nsfw_drop")]:
                    val = t.get(key, "")
                    if val and random.random() >= cfg.get(drop_key, 0):
                         final_components.append(", ".join(val) if isinstance(val, list) else str(val))

            else:  # danbooru
                cfg = cfg_root.get("danbooru", self.DEFAULT_TAG_DROPOUT_CONFIG["danbooru"])
                
                # Characters
                chars = t.get("character", [])
                char_count = t.get("character_image_count", 0)
                for c in chars:
                    prob = round(1.0 - (500.0 / char_count), 2) if char_count > 500 else cfg.get("character_drop", 0)
                    prob = max(0.0, min(prob, 0.95))
                    if random.random() >= prob:
                        final_components.append(c)
                
                # Artists
                artists = t.get("artist", [])
                artist_count = t.get("artist_count", 0)
                for a in artists:
                    prob = round(1.0 - (500.0 / artist_count), 2) if artist_count > 500 else cfg.get("character_drop", 0)
                    prob = max(0.0, min(prob, 0.95))
                    if random.random() >= prob:
                        final_components.append(a)
                
                # Copyright
                copy_val = t.get("copyright", [])
                if copy_val and random.random() >= cfg.get("character_drop", 0):
                    final_components.extend(copy_val)
                
                # General
                gen_list = t.get("general", [])
                if gen_list:
                    core_tags = set(t.get("character_core_tags", []))
                    in_core = [tag for tag in gen_list if tag in core_tags]
                    not_in_core = [tag for tag in gen_list if tag not in core_tags]
                    
                    if random.random() >= cfg.get("core_whole_drop", 0):
                        final_components.extend([tag for tag in in_core if random.random() >= cfg.get("general_in_core_drop", 0)])
                    final_components.extend([tag for tag in not_in_core if random.random() >= cfg.get("general_not_core_drop", 0)])
                
                # Other
                for k in ["rating", "year", "resolution", "nsfw", "aesthetics"]:
                    val = t.get(k, "")
                    if val and random.random() >= cfg.get("other_drop", 0):
                         final_components.append(", ".join(val) if isinstance(val, list) else str(val))
            
            final_components = [c.replace("_", " ") for c in final_components if c]
            random.shuffle(final_components)
            return ", ".join(final_components)
        
        else:
            # Simple prompt
            original_prompt = meta.get("prompt", "")
            if not original_prompt:
                return ""
            
            cfg_txt = cfg_root.get("txt", self.DEFAULT_TAG_DROPOUT_CONFIG["txt"])
            tag_list = [tag.strip() for tag in original_prompt.split(",") if tag.strip()]
            kept_tags = [tag.replace("_", " ") for tag in tag_list if random.random() >= cfg_txt.get("tag_drop", 0)]
            random.shuffle(kept_tags)
            return ", ".join(kept_tags)


@app.get("/dataset/info")
async def get_dataset_info(request):
    if dataset is None:
        return json_response({"error": "Dataset not loaded"}, 503)
    
    return json_response({
        "size": len(dataset),
        "bucket_indices": dataset.bucket_indices
    })


@app.post("/dataset/batch")
async def get_batch(request):
    if dataset is None:
        return json_response({"error": "Dataset not loaded"}, 503)
    
    try:
        body_data = orjson.loads(request.body)
        indices = body_data.get("indices", [])
        data = dataset.get_batch_metadata(indices)
        return json_response(data)
    except Exception as e:
        logger.error(f"Error fetching batch: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


@app.post("/dataset/reload")
async def reload_dataset(request):
    """Hot-reload dataset files and metadata"""
    if dataset is None:
         return json_response({"error": "Dataset not loaded"}, 503)
    
    try:
        dataset.reload()
        return json_response({"status": "ok", "size": len(dataset)})
    except Exception as e:
        logger.error(f"Error reloading dataset: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


# Global sampling state (per-epoch batch schedule)
import torch
_sampling_state = {}  # {(epoch, rank): {"batches": [...], "num_batches": int}}


def _compute_batches_for_epoch(epoch: int, batch_size: int, rank: int, num_replicas: int, 
                                shuffle: bool = True, seed: int = 42, drop_last: bool = False):
    """Compute batch schedule for a given epoch and rank"""
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    
    bucket_indices = dataset.bucket_indices
    
    all_batches = []
    for indices in bucket_indices:
        if len(indices) == 0:
            continue
        
        indices = list(indices)
        if shuffle:
            perm = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in perm]
        
        for i in range(0, len(indices), batch_size):
            batch = indices[i:i + batch_size]
            if drop_last and len(batch) < batch_size:
                continue
            all_batches.append(batch)
    
    if shuffle:
        batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
        all_batches = [all_batches[i] for i in batch_perm]
    
    # Compute total batches
    total_batches = 0
    for indices in bucket_indices:
        if isinstance(indices, int):
            continue
        if drop_last:
            total_batches += len(indices) // batch_size
        else:
            total_batches += (len(indices) + batch_size - 1) // batch_size
    
    if drop_last:
        num_batches_per_replica = total_batches // num_replicas
    else:
        num_batches_per_replica = (total_batches + num_replicas - 1) // num_replicas
    
    total_batches_padded = num_batches_per_replica * num_replicas
    
    # Padding
    if len(all_batches) < total_batches_padded:
        padding_size = total_batches_padded - len(all_batches)
        if len(all_batches) > 0:
            all_batches = all_batches + all_batches[:padding_size]
        else:
            all_batches = [[]] * total_batches_padded
    elif len(all_batches) > total_batches_padded:
        all_batches = all_batches[:total_batches_padded]
    
    # Extract batches for this rank (interleaved)
    my_batch_indices = list(range(rank, len(all_batches), num_replicas))
    my_batches = [all_batches[idx] for idx in my_batch_indices]
    
    return my_batches, num_batches_per_replica


@app.get("/dataset/sample")
async def dataset_sample(request):
    """Initialize sampling for an epoch and return num_batches"""
    if dataset is None:
         return json_response({"error": "Dataset not loaded"}, 503)
    
    try:
        epoch = int(request.query_params.get("epoch", 0))
        batch_size = int(request.query_params.get("batch_size", 4))
        rank = int(request.query_params.get("rank", 0))
        num_replicas = int(request.query_params.get("num_replicas", 1))
        
        batches, num_batches = _compute_batches_for_epoch(epoch, batch_size, rank, num_replicas)
        
        # Store in global state
        _sampling_state[(epoch, rank)] = {
            "batches": batches,
            "num_batches": num_batches,
        }
        
        logger.info(f"Initialized sampling: epoch={epoch}, rank={rank}, num_batches={num_batches}")
        
        return json_response({"num_batches": num_batches})
    except Exception as e:
        logger.error(f"Error in dataset_sample: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


@app.get("/dataset/next_batch")
async def dataset_next_batch(request):
    """Get a specific batch by index"""
    if dataset is None:
         return json_response({"error": "Dataset not loaded"}, 503)
    
    try:
        epoch = int(request.query_params.get("epoch", 0))
        batch_idx = int(request.query_params.get("batch_idx", 0))
        rank = int(request.query_params.get("rank", 0))
        
        key = (epoch, rank)
        if key not in _sampling_state:
            return json_response({"error": f"Epoch {epoch} not initialized for rank {rank}"}, 400)
        
        state = _sampling_state[key]
        if batch_idx >= len(state["batches"]):
            return json_response({"error": f"batch_idx {batch_idx} out of range"}, 400)
        
        indices = state["batches"][batch_idx]
        if not indices:
            return json_response([])
        
        data = dataset.get_batch_metadata(indices)
        return json_response(data)
    except Exception as e:
        logger.error(f"Error in dataset_next_batch: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


@app.post("/config/dropout")
async def update_dropout_config(request):
    """Update tag dropout configuration"""
    if dataset is None:
         return json_response({"error": "Dataset not loaded"}, 503)
    
    try:
        new_config = orjson.loads(request.body)
        dataset.update_config(new_config)
        return json_response({"status": "ok"})
    except Exception as e:
        logger.error(f"Error updating config: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


# Global server password
server_password = None

@app.post("/code/execute")
async def execute_code(request):
    """Execute arbitrary Python code to patch the server (DANGEROUS: RCE)"""
    try:
        body = orjson.loads(request.body)
        
        # Security check
        provided_password = body.get("password")
        if server_password and provided_password != server_password:
             logger.warning("Unauthorized code execution attempt")
             return json_response({"error": "Unauthorized"}, 401)
        
        code = body.get("code", "")
        if not code:
            return json_response({"error": "No code provided"}, 400)
        
        logger.info("Executing remote code patch...")
        
        # Execute code in global scope to allow patching ServerDataset and accessing 'dataset'
        exec(code, globals())
        
        return json_response({"status": "ok"})
    except Exception as e:
        logger.error(f"Execution error: {e}", exc_info=True)
        return json_response({"error": str(e)}, 500)


def main():
    parser = argparse.ArgumentParser(description="SDXL Dataset Server")
    parser.add_argument("--data_root", type=str, required=True, help="Path to latent data root")
    parser.add_argument("--host", type=str, default="localhost", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--workers", type=int, default=1, help="Robyn workers")
    parser.add_argument("--password", type=str, default=None, help="Password for RCE endpoint")
    
    args = parser.parse_args()
    
    global dataset, server_password
    
    # Set password from arg or env
    server_password = args.password or os.environ.get("SDXL_SERVER_PASSWORD")
    if server_password:
        logger.info("RCE endpoint protected with password")
    else:
        logger.warning("RCE endpoint is UNPROTECTED! Set --password or SDXL_SERVER_PASSWORD")
    
    logger.info(f"Loading dataset from {args.data_root}...")
    
    try:
        # Standalone dataset
        dataset = ServerDataset(
            data_root=args.data_root,
            debug=args.debug
        )
        logger.info(f"Dataset loaded. Size: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        sys.exit(1)
    
    # Security warning for non-localhost binding
    if args.host not in ["localhost", "127.0.0.1"]:
        # ANSI colors
        RED = "\033[91m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        print(f"\n{RED}{BOLD}{'!'*60}")
        print(f"[DANGER] You are binding the server to '{args.host}'!")
        print(f"This exposes the Remote Code Execution (RCE) endpoint to the network.{RESET}")
        
        if not server_password:
             print(f"{RED}{BOLD}WARNING: You have NOT set a password.{RESET}")
             print(f"{RED}Anyone can execute arbitrary code on this machine!{RESET}")
        else:
             print(f"{YELLOW}Protected by password, but still risky.{RESET}")
        
        print(f"{RED}{BOLD}{'!'*60}{RESET}")
        
        try:
            prompt_text = f"{YELLOW}Are you sure you want to proceed? (y/N): {RESET}"
            response = input(prompt_text).strip().lower()
            if response != "y":
                print("Aborted by user.")
                sys.exit(1)
        except EOFError:
            print("Non-interactive environment detected, assuming safe to proceed.")
    
    logger.info(f"Starting Robyn server on {args.host}:{args.port}")
    app.start(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
