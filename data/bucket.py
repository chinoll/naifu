import functools
import json
import math
import os
import numpy as np
import random
import torch
from collections import defaultdict

import cv2
from pathlib import Path
from torch.utils.data import Dataset, get_worker_info
from data.image_storage import DirectoryImageStore, Entry, LatentStore
from torchvision.transforms import Resize, InterpolationMode
from common.logging import logger
from common.utils import get_class

image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])


def is_latent_folder(path: Path):
    # iterate over all files in the folder and find if any of them is a latent
    for p in path.iterdir():
        if p.is_dir():
            continue
        if p.suffix == ".h5":
            return True

def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset: RatioDataset = worker_info.dataset  # type: ignore
    # random.seed(worker_info.seed)  # type: ignore
    dataset.init_batches()


class RatioDataset(Dataset):
    def __init__(
        self,
        batch_size: int,
        img_path: Path | str | list,
        ucg: int = 0,
        rank: int = 0,
        dtype=torch.float16,
        seed: int = 42,
        use_central_crop=True,
        **kwargs,
    ):
        self.rng = np.random.default_rng(seed)
        self.batch_size = batch_size
        self.num_workers = kwargs.get("num_workers", 4)
        self.use_central_crop = use_central_crop

        root_path = Path(img_path)
        assert root_path.exists(), f"Path {root_path} does not exist."
        
        if kwargs.get("store_cls"):
            store_class = get_class(kwargs["store_cls"])
        elif is_latent_folder(root_path):
            store_class = LatentStore
        else:
            store_class = DirectoryImageStore

        self.store = store_class(
            root_path,
            rank=rank,
            ucg=ucg,
            dtype=dtype,
            **kwargs,
        )
    
    def generate_buckets(self):
        raise NotImplementedError

    def assign_buckets(self):
        raise NotImplementedError

    def init_batches(self):
        self.assign_buckets()
        self.assign_batches()

    def init_dataloader(self, **kwargs):
        dataloader = torch.utils.data.DataLoader(
            self,
            sampler=None,
            batch_size=None,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            shuffle=True,
            pin_memory=True,
            **kwargs,
        )
        return dataloader

    def __len__(self):
        return len(self.batch_idxs)
    
    @staticmethod
    @functools.cache
    def fit_dimensions(target_ratio, min_h, min_w):
        min_area = min_h * min_w
        h = max(min_h, math.ceil(math.sqrt(min_area * target_ratio)))
        w = max(min_w, math.ceil(h / target_ratio))

        if w < min_w:
            w = min_w
            h = max(min_h, math.ceil(w * target_ratio))

        while h * w < min_area:
            increment = 8
            if target_ratio >= 1:
                h += increment
            else:
                w += increment

            w = max(min_w, math.ceil(h / target_ratio))
            h = max(min_h, math.ceil(w * target_ratio))
        return int(h), int(w)

    def assign_batches(self):
        self.batch_idxs = []
        for bucket in self.bucket_content:
            if not bucket or len(bucket) == 0:
                continue
            reminder = len(bucket) % self.batch_size
            bucket = np.array(bucket)
            self.rng.shuffle(bucket)
            if not reminder:
                self.batch_idxs.extend(bucket.reshape(-1, self.batch_size))
            else:
                self.batch_idxs.extend(bucket[:-reminder].reshape(-1, self.batch_size))
                self.batch_idxs.append(bucket[-reminder:])

        self.rng.shuffle(self.batch_idxs)

    def __getitem__(self, idx):
        img_idxs = self.batch_idxs[idx]
        return self.store.get_batch(img_idxs)
    
#        return True, latent, prompt, original_size, dhdw, extras
from torch.utils.data import Sampler
import torch.distributed as dist


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
        """
        Args:
            bucket_indices: 每个bucket包含的样本索引列表，如 [[0,1,2], [3,4,5,6]]
            batch_size: 每个batch的大小
            num_replicas: 分布式训练的总进程数，None则自动检测
            rank: 当前进程的rank，None则自动检测
            shuffle: 是否在每个epoch打乱bucket内的顺序
            seed: 随机种子
            drop_last: 是否丢弃最后不完整的batch
        """
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
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas})"
            )
        
        self.bucket_indices = bucket_indices
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # 预计算总batch数
        self._compute_num_batches()
    
    def _compute_num_batches(self):
        """计算总batch数和每个进程的batch数"""
        total_batches = 0
        for indices in self.bucket_indices:
            # 处理 Lightning Fabric 重新实例化时可能传入的不同类型
            if isinstance(indices, int):
                # 如果 indices 是 int，说明 bucket_indices 可能是一个简单的范围
                # 此时直接跳过计算，使用默认值
                continue
            if self.drop_last:
                total_batches += len(indices) // self.batch_size
            else:
                total_batches += (len(indices) + self.batch_size - 1) // self.batch_size
        
        # 如果没有有效的 bucket，设置默认值
        if total_batches == 0:
            self.num_batches = 0
            self.total_batches = 0
            return
        
        # 分布式情况下，确保每个进程有相同数量的batch
        if self.drop_last:
            self.num_batches = total_batches // self.num_replicas
        else:
            self.num_batches = (total_batches + self.num_replicas - 1) // self.num_replicas
        
        self.total_batches = self.num_batches * self.num_replicas
    
    def set_epoch(self, epoch: int):
        """设置epoch，用于改变每个epoch的shuffle顺序
        
        在分布式训练中，应该在每个epoch开始时调用此方法。
        """
        self.epoch = epoch
    
    def __iter__(self):
        # 使用epoch和seed生成随机数生成器
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # 生成所有batches
        all_batches = []
        for indices in self.bucket_indices:
            if len(indices) == 0:
                continue
            
            indices = list(indices)
            
            # 在bucket内shuffle
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in perm]
            
            # 按batch_size分组
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        
        # Shuffle所有batches的顺序
        if self.shuffle:
            batch_perm = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_perm]
        
        # Padding以确保所有进程有相同数量的batches
        if len(all_batches) < self.total_batches:
            # 重复部分batches来填充
            padding_size = self.total_batches - len(all_batches)
            if len(all_batches) > 0:
                all_batches = all_batches + all_batches[:padding_size]
            else:
                # 如果没有任何batch，创建空batch
                all_batches = [[]] * self.total_batches
        elif len(all_batches) > self.total_batches:
            all_batches = all_batches[:self.total_batches]
        
        # 分配给当前进程的batches
        # 使用交错分配方式：rank 0 取 0, num_replicas, 2*num_replicas, ...
        indices = list(range(self.rank, len(all_batches), self.num_replicas))
        
        for idx in indices:
            yield all_batches[idx]
    
    def __len__(self):
        """返回当前进程的batch数量"""
        return self.num_batches

class SimpleLatentDataset(Dataset):
    """简单的Latent数据集，用于加载预处理好的latent文件
    
    数据目录结构:
        data_root/
            latents/
                resolution_1/  (例如: 512x512)
                    file1.pt
                    file2.pt
                resolution_2/  (例如: 768x512)
                    file3.pt
            metadata.json
    
    metadata.json 格式:
        {key: {'prompt': xxx, 'original_size': xxx, 'dhdw': xxx}}
    """
    
    def __init__(
        self, 
        batch_size: int,
        rank: int = 0,
        dtype = None,
        data_root: str = None,
        img_path: str = None,  # 兼容其他 dataset 的参数名
        seed: int = 42,
        num_workers: int = 4,
        shuffle: bool = True,
        drop_last: bool = False,
        debug: bool = False,  # debug模式：忽略metadata，使用默认参数
        **kwargs,  # 忽略其他多余参数
    ):
        """
        Args:
            batch_size: 每个batch的大小
            rank: 当前进程的rank（用于分布式训练）
            dtype: 数据类型（保持接口兼容，实际不使用）
            data_root: 数据根目录路径
            img_path: 数据根目录路径（兼容其他dataset的参数名，与data_root二选一）
            seed: 随机种子
            num_workers: DataLoader的worker数量
            shuffle: 是否打乱数据
            drop_last: 是否丢弃最后不完整的batch
            **kwargs: 其他参数（忽略）
        """
        super().__init__()
        
        # 兼容 data_root 和 img_path 两种参数名
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
        
        # 读取所有分辨率的latent文件，按bucket组织
        latents_dir = os.path.join(self.data_root, "latents")
        self.buckets = {
            k: self._dirwalk(os.path.join(latents_dir, k)) 
            for k in os.listdir(latents_dir)
            if os.path.isdir(os.path.join(latents_dir, k))
        }
        
        # 加载metadata（非debug模式）
        # 支持两种格式：
        # - metadata.jsonl: 每行一条记录，格式 {"sha1": {"prompt": xxx, "original_size": xxx, "dhdw": xxx}}
        if not self.debug:
            jsonl_path = os.path.join(self.data_root, "metadata.jsonl")
            
            if os.path.exists(jsonl_path):
                # 加载 JSONL 格式
                self.metadata = {}
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            self.metadata.update(record)
                logger.info(f"Loaded metadata from JSONL: {jsonl_path}")
            else:
                raise FileNotFoundError(f"No metadata file found. Expected: {jsonl_path}")
        else:
            self.metadata = {}  # debug模式下不加载metadata
            logger.info("Debug mode: skipping metadata loading")
        
        # 构建文件列表和bucket索引
        self.files = []
        self.bucket_indices = []  # 用于SimpleBucketSampler
        missing_keys = []
        
        for resolution, files in self.buckets.items():
            bucket_valid_indices = []
            for filepath in files:
                key = Path(filepath).stem
                # debug模式下不验证key，直接添加
                if self.debug or key in self.metadata:
                    bucket_valid_indices.append(len(self.files))
                    self.files.append(filepath)
                else:
                    missing_keys.append(key)
            
            if bucket_valid_indices:
                self.bucket_indices.append(bucket_valid_indices)
        
        if missing_keys and not self.debug:
            logger.warning(f"Found {len(missing_keys)} files without metadata, skipping them")
            if len(missing_keys) <= 10:
                logger.warning(f"Missing keys: {missing_keys}")
            else:
                logger.warning(f"First 10 missing keys: {missing_keys[:10]}")
        
        logger.info(f"Loaded {len(self.files)} latent files from {len(self.bucket_indices)} buckets")
    
    def _dirwalk(self, path):
        """递归遍历目录，获取所有.pt文件"""
        logger.info(f"Reading latents from: {path}")
        path = Path(path)
        return [str(file) for file in path.rglob('*.pt') if file.is_file()]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        filepath = self.files[index]
        latent = torch.load(filepath, weights_only=True)
        
        if self.debug:
            # debug模式：使用默认值
            # original_size 和 dhdw 使用 (0, 0)，prompt 为空
            return True, latent, "", (0, 0), (0, 0), None
        else:
            key = Path(filepath).stem
            meta = self.metadata[key]
            return True, latent, meta['prompt'], meta['original_size'], meta['dhdw'], None
    
    @staticmethod
    def simple_collate_fn(batch):
        """将batch中的数据整合为统一格式，与StoreBase.get_batch返回格式一致
        
        Args:
            batch: list of (is_latent, latent, prompt, original_size, dhdw, extras)
        
        Returns:
            dict with keys: prompts, pixels, is_latent, target_size_as_tuple,
                           original_size_as_tuple, crop_coords_top_left, extras
        """
        is_latents, latents, prompts, original_sizes, dhdws, extras = zip(*batch)
        
        # Stack latents (假设同一bucket内latent尺寸相同)
        pixels = torch.stack(latents, dim=0)
        
        # 获取 is_latent（所有项应该相同）
        is_latent = is_latents[0]
        
        # 计算 target_size_as_tuple (latent shape * 8 得到像素尺寸)
        # latent shape: (C, H, W) -> target_size: (H*8, W*8)
        latent_h, latent_w = pixels.shape[-2:]
        if is_latent:
            target_h, target_w = latent_h * 8, latent_w * 8
        else:
            target_h, target_w = latent_h, latent_w
        
        batch_size = len(is_latents)
        target_sizes = torch.tensor([[target_h, target_w]] * batch_size)
        
        # original_size 转为 tensor
        original_sizes = torch.tensor(original_sizes)
        
        # crop_coords_top_left: dhdw 需要乘以 8（如果是 latent）
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
        """初始化DataLoader
        
        Returns:
            torch.utils.data.DataLoader
        """
        sampler = SimpleBucketSampler(
            bucket_indices=self.bucket_indices,
            batch_size=self.batch_size,
            rank=self.rank,  # 传递 rank 参数用于分布式
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_sampler=sampler,  # 使用batch_sampler而非sampler
            num_workers=self.num_workers,
            collate_fn=self.simple_collate_fn,
            pin_memory=True,
            **kwargs,
        )
        return dataloader

class AspectRatioDataset(RatioDataset):
    """Original implementation of AspectRatioDataset, equal to other frameworks"""
    def __init__(
        self, 
        batch_size: int, 
        img_path: Path | str | list, 
        ucg: int = 0, rank: int = 0, 
        dtype=torch.float16, 
        target_area: int = 1024 * 1024, 
        min_size: int = 512, 
        max_size: int = 2048, 
        divisible: int = 64, 
        seed: int = 42, 
        **kwargs
    ):
        super().__init__(batch_size, img_path, ucg, rank, dtype, seed, **kwargs)
        self.target_area = target_area
        self.max_size, self.min_size, self.divisible = max_size, min_size, divisible
        self.store.crop = self.crop

        self.generate_buckets()
        self.init_batches()
    
    def crop(self, entry: Entry, i: int) -> Entry:
        assert self.to_ratio is not None, "to_ratio is not initialized"
        H, W = entry.pixel.shape[-2:]
        base_ratio = H / W
        target_ratio = self.to_ratio[i]
        h, w = self.ratio_to_bucket[target_ratio]
        if not entry.is_latent:
            resize_h, resize_w = self.fit_dimensions(base_ratio, h, w)
            # interp = InterpolationMode.BILINEAR if resize_h < H else InterpolationMode.BICUBIC
            # entry.pixel = Resize(
            #     size=(resize_h, resize_w), 
            #     interpolation=interp, 
            #     antialias=None
            # )(entry.pixel)
            
            pixel = entry.pixel
            if isinstance(pixel, torch.Tensor):
                pixel = pixel.permute(1, 2, 0).cpu().numpy()
                
            interp = cv2.INTER_AREA if resize_h < H else cv2.INTER_LANCZOS4
            pixel = cv2.resize(pixel.astype(float), (resize_w, resize_h), interpolation=interp)
            entry.pixel = torch.from_numpy(pixel).permute(2, 0, 1)
        else:
            h, w = h // 8, w // 8

        H, W = entry.pixel.shape[-2:]
        if self.use_central_crop:
            dh, dw = (H - h) // 2, (W - w) // 2
        else:
            assert H >= h and W >= w, f"{H}<{h} or {W}<{w}"
            dh, dw = random.randint(0, H - h), random.randint(0, W - w)

        entry.pixel = entry.pixel[:, dh : dh + h, dw : dw + w]
        return entry, dh, dw

    def generate_buckets(self):
        assert (
            self.target_area % 4096 == 0
        ), "target area (h * w) must be divisible by 64"
        width = np.arange(self.min_size, self.max_size + 1, self.divisible)
        height = np.minimum(
            self.max_size,
            ((self.target_area // width) // self.divisible) * self.divisible,
        )
        valid_mask = height >= self.min_size

        resos = set(zip(width[valid_mask], height[valid_mask]))
        resos.update(zip(height[valid_mask], width[valid_mask]))
        resos.add(((int(np.sqrt(self.target_area)) // self.divisible) * self.divisible,) * 2)
        self.buckets_sizes = np.array(sorted(resos))
        self.bucket_ratios = self.buckets_sizes[:, 0] / self.buckets_sizes[:, 1]
        self.ratio_to_bucket = {ratio: hw for ratio, hw in zip(self.bucket_ratios, self.buckets_sizes)}

    def assign_buckets(self):
        img_res = np.array(self.store.raw_res)
        img_ratios = img_res[:, 0] / img_res[:, 1]
        self.bucket_content = [[] for _ in range(len(self.buckets_sizes))]
        self.to_ratio = {}

        # Assign images to buckets
        for idx, img_ratio in enumerate(img_ratios):
            diff = np.abs(np.log(self.bucket_ratios) - np.log(img_ratio))
            bucket_idx = np.argmin(diff)
            self.bucket_content[bucket_idx].append(idx)
            self.to_ratio[idx] = self.bucket_ratios[bucket_idx]


class AdaptiveSizeDataset(RatioDataset):
    """AdaptiveRatioDataset, a modified version of AspectRatioDataset which avoid resize from smaller images"""
    def __init__(
        self, 
        batch_size: int, 
        img_path: Path | str | list, 
        ucg: int = 0, rank: int = 0, 
        dtype=torch.float16, 
        target_area: int = 1024 * 1024, 
        divisible: int = 64, 
        seed: int = 42, 
        **kwargs
    ):
        super().__init__(batch_size, img_path, ucg, rank, dtype, seed, **kwargs)
        self.store.crop = self.crop
        self.target_area = target_area
        self.divisible = divisible

        self.generate_buckets()
        self.init_batches()
    
    def crop(self, entry: Entry, i: int) -> Entry:
        assert self.to_size is not None, "to_ratio is not initialized"
        H, W = entry.pixel.shape[-2:]
        h, w = self.to_size[i]
        bucket_width = w - w % self.divisible
        bucket_height = h - h % self.divisible
        
        if not entry.is_latent:
            resize_h, resize_w = h, w
            # entry.pixel = Resize(
            #     size=(resize_h, resize_w), 
            #     interpolation=InterpolationMode.BILINEAR, 
            #     antialias=None
            # )(entry.pixel)

            pixel = entry.pixel
            if isinstance(pixel, torch.Tensor):
                pixel = pixel.permute(1, 2, 0).cpu().numpy()
                
            interp = cv2.INTER_AREA if resize_h < H else cv2.INTER_LANCZOS4
            pixel = cv2.resize(pixel.astype(float), (resize_w, resize_h), interpolation=interp)
            entry.pixel = torch.from_numpy(pixel).permute(2, 0, 1)
        else:
            h, w = bucket_height // 8, bucket_width // 8

        H, W = entry.pixel.shape[-2:]
        if self.use_central_crop:
            dh, dw = (H - h) // 2, (W - w) // 2
        else:
            assert H >= h and W >= w, f"{H}<{h} or {W}<{w}"
            dh, dw = random.randint(0, H - h), random.randint(0, W - w)

        entry.pixel = entry.pixel[:, dh : dh + h, dw : dw + w]
        return entry, dh, dw

    def generate_buckets(self):
        pass
    
    def assign_buckets(self):
        img_res = np.array(self.store.raw_res)
        self.to_size = {}
        self.bucket_content = defaultdict(list)

        # Assign images to buckets
        for idx, (img_width, img_height) in enumerate(img_res):
            img_area = img_width * img_height

            # Check if the image needs to be resized (i.e., only allow downsizing)
            if img_area > self.target_area:
                scale_factor = math.sqrt(self.target_area / img_area)
                img_width = math.floor(img_width * scale_factor / self.divisible) * self.divisible
                img_height = math.floor(img_height * scale_factor / self.divisible) * self.divisible

            bucket_width = img_width - img_width % self.divisible
            bucket_height = img_height - img_height % self.divisible
            reso = (bucket_width, bucket_height)
            self.bucket_content[reso].append(idx)
            self.to_size[idx] = (bucket_width, bucket_height)

        self.bucket_content = [v for k, v in self.bucket_content.items()]
