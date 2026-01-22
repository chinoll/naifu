"""
SimpleLatentDataset and SimpleBucketSampler for pre-encoded latent training.

Extracted from data/bucket.py for standalone use with Diffusers + Accelerate training.
"""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)

class SimpleBucketSampler(Sampler):
    """分布式感知的Bucket采样器
    
    支持多GPU/多节点训练，每个进程只采样属于自己的batch。
    在单机训练时自动退化为普通采样器。
    """
    
    def __init__(
        self, 
        bucket_indices: list,
        batch_size: int,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False
    ):
        # 自动检测分布式环境
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas})")
        
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._compute_num_batches()
    
    def _compute_num_batches(self):
        total_batches = 0
        for indices in self.bucket_indices:
            if isinstance(indices, int):
                continue
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        
        if total_batches == 0:
            self.num_batches = 0
            self.total_batches = 0
            return
        
        if self.drop_last:
            self.num_batches = total_batches // self.num_replicas
        else:
            self.num_batches = (total_batches + self.num_replicas - 1) // self.num_replicas
        
        self.total_batches = self.num_batches * self.num_replicas
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        all_batches = []
        for indices in self.bucket_indices:
            if len(indices) == 0:
                continue
            
            indices = list(indices)
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]
            
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]
        
        # Padding
        if len(all_batches) < self.total_batches:
            padding_size = self.total_batches - len(all_batches)
            if len(all_batches) > 0:
                all_batches = all_batches + all_batches[:padding_size]
            else:
                all_batches = [[]] * self.total_batches
        elif len(all_batches) > self.total_batches:
            all_batches = all_batches[:self.total_batches]
        
        # Interleaved distribution
        indices = list(range(self.rank, len(all_batches), self.num_replicas))
        for idx in indices:
            yield all_batches[idx]
    
    def __len__(self):
        return self.num_batches


class SimpleBucketSamplerForHttp(Sampler):
    """HTTP-based Bucket Sampler that delegates sampling to server.
    
    This sampler:
    1. Sends epoch_id, batch_size, rank to server
    2. Server computes indices and returns batch data
    3. Sampler just iterates through server responses
    
    Server API:
        GET /dataset/sample?epoch={epoch}&batch_size={bs}&rank={rank}&num_replicas={n}
        -> Returns iterator info: {"num_batches": int}
        
        GET /dataset/next_batch?epoch={epoch}&batch_idx={idx}&rank={rank}
        -> Returns batch data directly
    
    Use with EmptyLatentDataset as placeholder.
    """
    
    def __init__(
        self,
        server_url: str,
        batch_size: int,
        num_replicas: int = None,
        rank: int = None,
        timeout: float = 30.0,
        http2: bool = True,
    ):
        # Distributed setup
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_replicas})")
        
        self.server_url = server_url.rstrip('/')
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.timeout = timeout
        self.http2 = http2
        self.epoch = 0
        self._num_batches = None
        
        # JSON handling
        try:
            import orjson
            self._json_loads = orjson.loads
        except ImportError:
            self._json_loads = json.loads
        
        # Fetch initial info
        self._init_epoch()
    
    def _get_client(self):
        """Get or create HTTP client"""
        if not hasattr(self, '_client') or self._client is None:
            import httpx
            limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
            self._client = httpx.Client(
                base_url=self.server_url,
                timeout=httpx.Timeout(self.timeout),
                limits=limits,
                http2=self.http2,
            )
        return self._client
    
    def _init_epoch(self):
        """Initialize epoch and get num_batches from server"""
        client = self._get_client()
        
        try:
            response = client.get(
                "/dataset/sample",
                params={
                    "epoch": self.epoch,
                    "batch_size": self.batch_size,
                    "rank": self.rank,
                    "num_replicas": self.num_replicas,
                }
            )
            response.raise_for_status()
            info = self._json_loads(response.content)
            self._num_batches = info["num_batches"]
            logger.info(f"SimpleBucketSamplerForHttp: epoch={self.epoch}, num_batches={self._num_batches}")
        except Exception as e:
            raise RuntimeError(f"Failed to init epoch from {self.server_url}: {e}")
    
    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self._init_epoch()
    
    def __iter__(self):
        client = self._get_client()
        
        for batch_idx in range(self._num_batches):
            try:
                response = client.get(
                    "/dataset/next_batch",
                    params={
                        "epoch": self.epoch,
                        "batch_idx": batch_idx,
                        "rank": self.rank,
                    }
                )
                response.raise_for_status()
                batch_data = self._json_loads(response.content)
                
                # Load latents and collate
                latents = []
                prompts = []
                original_sizes = []
                dhdws = []
                extras_list = []
                
                for data in batch_data:
                    latent_path = data["latent_path"]
                    latent = torch.from_numpy(np.load(latent_path))
                    latents.append(latent)
                    prompts.append(data.get("prompt", ""))
                    original_sizes.append(data.get("original_size", [0, 0]))
                    dhdws.append(data.get("dhdw", [0, 0]))
                    extras_list.append(data.get("extras", None))
                
                if not latents:
                    continue
                    
                pixels = torch.stack(latents, dim=0)
                latent_h, latent_w = pixels.shape[-2:]
                target_h, target_w = (latent_h * 8, latent_w * 8)
                bs = len(latents)
                
                yield {
                    "prompts": prompts,
                    "pixels": pixels,
                    "is_latent": True,
                    "target_size_as_tuple": torch.tensor([[target_h, target_w]] * bs),
                    "original_size_as_tuple": torch.tensor(original_sizes),
                    "crop_coords_top_left": torch.tensor([[dh * 8, dw * 8] for dh, dw in dhdws]),
                    "extras": extras_list,
                }
            except Exception as e:
                logger.error(f"Error fetching batch {batch_idx}: {e}")
                continue
    
    def __len__(self):
        return self._num_batches or 0
    
    def close(self):
        if hasattr(self, '_client') and self._client:
            self._client.close()
            self._client = None
    
    def __del__(self):
        self.close()


class EmptyLatentDataset(Dataset):
    """Placeholder dataset for use with SimpleBucketSamplerForHttp.
    
    This dataset does nothing - the sampler handles all data fetching.
    """
    
    def __init__(self, size: int = 0):
        self._size = size
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        # Should never be called when using SimpleBucketSamplerForHttp
        raise NotImplementedError("EmptyLatentDataset should not be accessed directly")
    
    @staticmethod
    def passthrough_collate(batch):
        """Pass through the batch as-is (already collated by sampler)"""
        # batch is a list containing one already-collated dict from sampler
        if batch and isinstance(batch[0], dict):
            return batch[0]
        return batch


class SimpleLatentDataset(Dataset):
    """简单的Latent数据集，用于加载预处理好的latent文件
    
    数据目录结构:
        data_root/
            latents/
                resolution_1/  (例如: 512x512)
                    file1.npy
                resolution_2/  (例如: 768x512)
                    file2.npy
            metadata.jsonl
    """
    
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
    
    def __init__(
        self, 
        batch_size: int,
        rank: int = 0,
        dtype=None,
        data_root: str = None,
        img_path: str = None,
        seed: int = 42,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.data_root = data_root or img_path
        if self.data_root is None:
            raise ValueError("Must specify either 'data_root' or 'img_path'")
        
        self.batch_size = batch_size
        self.rank = rank
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.debug = debug
        
        # 读取所有分辨率的latent文件
        latents_dir = os.path.join(self.data_root, "latents")
        self.buckets = {
            k: self._dirwalk(os.path.join(latents_dir, k)) 
            for k in os.listdir(latents_dir)
            if os.path.isdir(os.path.join(latents_dir, k))
        }
        
        # 加载metadata
        if not self.debug:
            jsonl_path = os.path.join(self.data_root, "metadata.jsonl")
            if os.path.exists(jsonl_path):
                self.metadata = {}
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            self.metadata.update(record)
                logger.info(f"Loaded metadata from: {jsonl_path}")
            else:
                raise FileNotFoundError(f"No metadata file found: {jsonl_path}")
        else:
            self.metadata = {}
            logger.info("Debug mode: skipping metadata loading")
        
        # 构建文件列表和bucket索引
        self.files = []
        self.bucket_indices = []
        missing_keys = []
        
        for resolution, files in self.buckets.items():
            bucket_valid_indices = []
            for filepath in files:
                key = Path(filepath).stem
                if self.debug or key in self.metadata:
                    # 处理 repeat
                    repeat_count = 1
                    if not self.debug and key in self.metadata:
                        meta = self.metadata[key]
                        meta_dict = meta.get('metadata', meta) if isinstance(meta.get('metadata'), dict) else meta
                        for field_key, value in meta_dict.items():
                            if field_key.endswith('_repeat') and isinstance(value, (int, float)):
                                repeat_count = max(repeat_count, int(value))
                    
                    for _ in range(repeat_count):
                        bucket_valid_indices.append(len(self.files))
                        self.files.append(filepath)
                else:
                    missing_keys.append(key)
            
            if bucket_valid_indices:
                self.bucket_indices.append(bucket_valid_indices)
        
        if missing_keys and not self.debug:
            logger.warning(f"Found {len(missing_keys)} files without metadata")
        
        logger.info(f"Loaded {len(self.files)} latent files from {len(self.bucket_indices)} buckets")
    
    def _dirwalk(self, path):
        path = Path(path)
        return [str(file) for file in path.rglob('*.npy') if file.is_file()]
    
    def apply_tag_dropout(self, meta: dict) -> str:
        """应用 tag dropout 生成最终的 prompt"""
        if "metadata" in meta and isinstance(meta.get("metadata"), dict):
            t = meta["metadata"]
            data_src = meta.get("data_src", "danbooru")
            final_components = []
            
            if data_src == "e621":
                cfg = self.DEFAULT_TAG_DROPOUT_CONFIG["e621"]
                
                for key, drop_key in [("copyright", "copyright_drop"), ("character", "character_drop")]:
                    val = t.get(key, [])
                    if val and random.random() >= cfg[drop_key]:
                        final_components.append(", ".join(val))
                
                for key, drop_key in [("species", "species_drop"), ("general", "general_drop")]:
                    val_list = t.get(key, [])
                    if val_list:
                        kept = [v for v in val_list if random.random() >= cfg[drop_key]]
                        if kept:
                            final_components.append(", ".join(kept))
                
                for key, drop_key in [("e621_tag", "e621_tag_drop"), ("resolution", "resolution_drop"), ("nsfw", "nsfw_drop")]:
                    val = t.get(key, "")
                    if val and random.random() >= cfg[drop_key]:
                        final_components.append(", ".join(val) if isinstance(val, list) else str(val))
            
            else:  # danbooru
                cfg = self.DEFAULT_TAG_DROPOUT_CONFIG["danbooru"]
                
                # Characters with dynamic dropout
                chars = t.get("character", [])
                char_count = t.get("character_image_count", 0)
                for c in chars:
                    prob = round(1.0 - (500.0 / char_count), 2) if char_count > 500 else cfg["character_drop"]
                    prob = max(0.0, min(prob, 0.95))
                    if random.random() >= prob:
                        final_components.append(c)
                
                # Artists with dynamic dropout
                artists = t.get("artist", [])
                artist_count = t.get("artist_count", 0)
                for a in artists:
                    prob = round(1.0 - (500.0 / artist_count), 2) if artist_count > 500 else cfg["character_drop"]
                    prob = max(0.0, min(prob, 0.95))
                    if random.random() >= prob:
                        final_components.append(a)
                
                # Copyright
                copy_val = t.get("copyright", [])
                if copy_val and random.random() >= cfg["character_drop"]:
                    final_components.extend(copy_val)
                
                # General tags
                gen_list = t.get("general", [])
                if gen_list:
                    core_tags = set(t.get("character_core_tags", []))
                    in_core = [tag for tag in gen_list if tag in core_tags]
                    not_in_core = [tag for tag in gen_list if tag not in core_tags]
                    if random.random() >= cfg["core_whole_drop"]:
                        final_components.extend([tag for tag in in_core if random.random() >= cfg["general_in_core_drop"]])
                    final_components.extend([tag for tag in not_in_core if random.random() >= cfg["general_not_core_drop"]])
                
                # Other meta
                for k in ["rating", "year", "resolution", "nsfw", "aesthetics"]:
                    val = t.get(k, "")
                    if val and random.random() >= cfg["other_drop"]:
                        final_components.append(", ".join(val) if isinstance(val, list) else str(val))
            
            final_components = [c.replace("_", " ") for c in final_components if c]
            random.shuffle(final_components)
            return ", ".join(final_components)
        
        else:
            # Simple prompt with txt dropout
            original_prompt = meta.get("prompt", "")
            if not original_prompt:
                return ""
            
            cfg_txt = self.DEFAULT_TAG_DROPOUT_CONFIG["txt"]
            tag_list = [tag.strip() for tag in original_prompt.split(",") if tag.strip()]
            kept_tags = [tag.replace("_", " ") for tag in tag_list if random.random() >= cfg_txt["tag_drop"]]
            random.shuffle(kept_tags)
            return ", ".join(kept_tags)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        filepath = self.files[index]
        latent = torch.from_numpy(np.load(filepath))
        
        if self.debug:
            return True, latent, "", (0, 0), (0, 0), None
        else:
            key = Path(filepath).stem
            meta = self.metadata[key]
            prompt = self.apply_tag_dropout(meta)
            return True, latent, prompt, meta['original_size'], meta['dhdw'], None
    
    @staticmethod
    def collate_fn(batch):
        """Collate batch into dict format"""
        is_latents, latents, prompts, original_sizes, dhdws, extras = zip(*batch)
        
        pixels = torch.stack(latents, dim=0)
        is_latent = is_latents[0]
        
        latent_h, latent_w = pixels.shape[-2:]
        target_h, target_w = (latent_h * 8, latent_w * 8) if is_latent else (latent_h, latent_w)
        
        batch_size = len(is_latents)
        target_sizes = torch.tensor([[target_h, target_w]] * batch_size)
        original_sizes = torch.tensor(original_sizes)
        
        dhdws_list = list(dhdws)
        if is_latent:
            crop_coords = torch.tensor([[dh * 8, dw * 8] for dh, dw in dhdws_list])
        else:
            crop_coords = torch.tensor(dhdws_list)
        
        return {
            "prompts": list(prompts),
            "pixels": pixels,
            "is_latent": is_latent,
            "target_size_as_tuple": target_sizes,
            "original_size_as_tuple": original_sizes,
            "crop_coords_top_left": crop_coords,
            "extras": list(extras),
        }
    
    def init_dataloader(self, **kwargs):
        sampler = SimpleBucketSampler(
            bucket_indices=self.bucket_indices,
            batch_size=self.batch_size,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        
        return torch.utils.data.DataLoader(
            self,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            **kwargs,
        )

class SimpleLatentDatasetForHttp(Dataset):
    """基于 HTTP 的 Latent 数据集，从远程服务器获取数据
    
    使用 httpx 库实现高性能 HTTP 请求，特性：
    - 连接池复用（每个 DataLoader worker 独立的连接池）
    - HTTP/2 支持（单连接多路复用）
    - 批量请求 API
    - orjson 高速 JSON 解析
    
    服务端 API 约定:
        GET  /dataset/info     -> {"size": int, "bucket_indices": [[int, ...], ...]}
        POST /dataset/batch    -> 请求 {"indices": [int, ...]}
                               -> 返回 [{"prompt": str, "latent_path": str, 
                                         "original_size": [h, w], "dhdw": [dh, dw], ...}, ...]
    """
    
    def __init__(
        self,
        batch_size: int,
        server_url: str,  # e.g. "http://localhost:8000"
        rank: int = 0,
        seed: int = 42,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        timeout: float = 120.0,
        max_connections: int = 100,
        http2: bool = True,
        prefetch_batch_size: int = 64,
        **kwargs,
    ):
        super().__init__()
        
        self.server_url = server_url.rstrip('/')
        self.batch_size = batch_size
        self.rank = rank
        self.seed = seed
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.timeout = timeout
        self.max_connections = max_connections
        self.http2 = http2
        self.prefetch_batch_size = prefetch_batch_size
        
        # JSON loading (prefer orjson)
        try:
            import orjson
            self._json_loads = orjson.loads
            self._json_dumps = lambda x: orjson.dumps(x).decode()
        except ImportError:
            self._json_loads = json.loads
            self._json_dumps = json.dumps
        
        # Fetch dataset info
        self._fetch_dataset_info()
        
        logger.info(f"SimpleLatentDatasetForHttp initialized: "
                    f"size={self._size}, buckets={len(self.bucket_indices)}, "
                    f"server={self.server_url}")
    
    def _get_client(self):
        """Get worker-local HTTP client (lazy init per worker)"""
        import httpx
        from torch.utils.data import get_worker_info
        
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        
        if not hasattr(self, '_worker_clients'):
            self._worker_clients = {}
        
        if worker_id not in self._worker_clients:
            limits = httpx.Limits(
                max_connections=self.max_connections,
                max_keepalive_connections=self.max_connections // 2,
            )
            self._worker_clients[worker_id] = httpx.Client(
                base_url=self.server_url,
                timeout=httpx.Timeout(self.timeout),
                limits=limits,
                http2=self.http2,
            )
        
        return self._worker_clients[worker_id]
    
    def _fetch_dataset_info(self):
        """Fetch dataset size and buckets from server"""
        import httpx
        
        # Use temporary client for init
        with httpx.Client(
            base_url=self.server_url,
            timeout=httpx.Timeout(self.timeout),
            http2=self.http2,
        ) as client:
            try:
                response = client.get("/dataset/info")
                response.raise_for_status()
                info = self._json_loads(response.content)
            except Exception as e:
                raise RuntimeError(f"Failed to fetch dataset info from {self.server_url}: {e}")
        
        self._size = info["size"]
        self.bucket_indices = info.get("bucket_indices", [])
        
        # Fallback if no buckets returned
        if not self.bucket_indices:
            self.bucket_indices = [list(range(self._size))]
    
    def _fetch_batch(self, indices: list) -> list:
        """Fetch batch data from server"""
        client = self._get_client()
        
        # Ensure indices are JSON serializable (integers)
        if hasattr(indices, "tolist"):
             indices = indices.tolist()
        else:
             indices = [int(i) for i in indices]
        
        response = client.post(
            "/dataset/batch",
            content=self._json_dumps({"indices": indices}),
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        
        return self._json_loads(response.content)
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index):
        return index
        print("index",index)
        """Get single item (fallback, inefficient)"""
        data = self._fetch_batch([index])[0]
        
        latent_path = data["latent_path"]
        latent = torch.from_numpy(np.load(latent_path))
        
        prompt = data.get("prompt", "")
        original_size = tuple(data.get("original_size", (0, 0)))
        dhdw = tuple(data.get("dhdw", (0, 0)))
        extras = data.get("extras", None)
        
        return True, latent, prompt, original_size, dhdw, extras
    
    def batch_collate_fn(self, batch_indices: list):
        """Optimized collate function that fetches data in batch"""
        # Fetch batch data
        #print(batch_indices)
        batch_data = self._fetch_batch(batch_indices)
        
        latents = []
        prompts = []
        original_sizes = []
        dhdws = []
        extras_list = []
        
        for data in batch_data:
            latent_path = data["latent_path"]
            latent = torch.from_numpy(np.load(latent_path))
            latents.append(latent)
            prompts.append(data.get("prompt", ""))
            original_sizes.append(data.get("original_size", [0, 0]))
            dhdws.append(data.get("dhdw", [0, 0]))
            extras_list.append(data.get("extras", None))
        
        # Stack latents
        pixels = torch.stack(latents, dim=0)
        
        # Calculate sizes
        latent_h, latent_w = pixels.shape[-2:]
        target_h, target_w = (latent_h * 8, latent_w * 8)
        batch_size = len(batch_indices)
        target_sizes = torch.tensor([[target_h, target_w]] * batch_size)
        original_sizes = torch.tensor(original_sizes)
        
        # Crop coords
        crop_coords = torch.tensor([[dh * 8, dw * 8] for dh, dw in dhdws])
        
        return {
            "prompts": prompts,
            "pixels": pixels,
            "is_latent": True,
            "target_size_as_tuple": target_sizes,
            "original_size_as_tuple": original_sizes,
            "crop_coords_top_left": crop_coords,
            "extras": extras_list,
        }
    
    def init_dataloader(self, **kwargs):
        """Initialize DataLoader with batch sampler and efficient collate"""
        sampler = SimpleBucketSampler(
            bucket_indices=self.bucket_indices,
            batch_size=self.batch_size,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        
        # Use batch_collate_fn with batch_sampler
        return torch.utils.data.DataLoader(
            self,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.batch_collate_fn,
            pin_memory=True,
            **kwargs,
        )
    
    def close(self):
        """Close all HTTP clients"""
        if hasattr(self, '_worker_clients'):
            for client in self._worker_clients.values():
                try:
                    client.close()
                except:
                    pass
            self._worker_clients.clear()
    
    def __del__(self):
        self.close()

