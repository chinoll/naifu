import os
import hashlib
import json
import h5py as h5
import numpy as np
import torch

from tqdm.auto import tqdm
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import Callable, Generator, Optional  # type: ignore
from torchvision import transforms
from common.logging import logger

json_lib = json
try:
    import rapidjson as json_lib
except ImportError:
    pass


image_suffix = set([".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"])

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def get_class(name: str):
    import importlib

    module_name, class_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name, package=None)
    return getattr(module, class_name)


def is_img(path: Path):
    return path.suffix in image_suffix


def sha1sum(txt):
    return hashlib.sha1(txt.encode()).hexdigest()


@dataclass
class Entry:
    is_latent: bool
    pixel: torch.Tensor
    prompt: str
    original_size: tuple[int, int]  # h, w
    cropped_size: Optional[tuple[int, int]]  # h, w
    dhdw: Optional[tuple[int, int]]  # dh, dw
    extras: dict = None
    # mask: torch.Tensor | None = None


def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p


class StoreBase(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        root_path,
        rank=0,
        dtype=torch.float16,
        process_batch_fn = "data.processors.identical",
        **kwargs,
    ):
        self.rank = rank
        self.root_path = Path(root_path)
        self.dtype = dtype
        self.kwargs = kwargs
        self.process_batch_fn = process_batch_fn
            
        self.length = 0
        self.rand_list: list = []
        self.raw_res: list[tuple[int, int]] = []
        self.curr_res: list[tuple[int, int]] = []

        assert self.root_path.exists()

    def get_raw_entry(self, index) -> tuple[bool, np.ndarray, str, (int, int)]:
        raise NotImplementedError

    def fix_aspect_randomness(self, rng: np.random.Generator):
        raise NotImplementedError
    
    def crop(self, entry: Entry, index: int) -> Entry:
        return entry, 0, 0
    
    @torch.no_grad()
    def get_batch(self, indices: list[int]) -> Entry:
        entries = [self._get_entry(i) for i in indices]
        crop_pos = []
        pixels = []
        prompts = []
        original_sizes = []
        cropped_sizes = []
        extras = []

        for e, i in zip(entries, indices):
            e = self.process_batch(e)
            e, dh, dw = self.crop(e, i)
            pixels.append(e.pixel)
            original_size = torch.asarray(e.original_size)
            original_sizes.append(original_size)

            cropped_size = e.pixel.shape[-2:]
            cropped_size = (
                (cropped_size[0] * 8, cropped_size[1] * 8)
                if e.is_latent
                else cropped_size
            )
            cropped_size = torch.asarray(cropped_size)
            cropped_sizes.append(cropped_size)

            cropped_pos = (dh, dw)
            cropped_pos = (
                (cropped_pos[0] * 8, cropped_pos[1] * 8) if e.is_latent else cropped_pos
            )
            cropped_pos = (cropped_pos[0] + e.dhdw[0], cropped_pos[1] + e.dhdw[1])
            cropped_pos = torch.asarray(cropped_pos)
            crop_pos.append(cropped_pos)
            prompts.append(e.prompt)
            extras.append(e.extras)

        is_latent = entries[0].is_latent
        shape = entries[0].pixel.shape

        for e in entries[1:]:
            assert e.is_latent == is_latent
            assert (
                e.pixel.shape == shape
            ), f"{e.pixel.shape} != {shape} for the same batch"

        pixel = torch.stack(pixels, dim=0).contiguous()
        cropped_sizes = torch.stack(cropped_sizes)
        original_sizes = torch.stack(original_sizes)
        crop_pos = torch.stack(crop_pos)

        return {
            "prompts": prompts,
            "pixels": pixel,
            "is_latent": is_latent,
            "target_size_as_tuple": cropped_sizes,
            "original_size_as_tuple": original_sizes,
            "crop_coords_top_left": crop_pos,
            "extras": extras,
        }

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise NotImplementedError

    def get_batch_extras(self, path):
        return None

    def process_batch(self, inputs: Entry):
        if isinstance(self.process_batch_fn, str):
            self.process_batch_fn = get_class(self.process_batch_fn)
            
        return self.process_batch_fn(inputs)

    def _get_entry(self, index) -> Entry:
        is_latent, pixel, prompt, original_size, dhdw, extras = self.get_raw_entry(
            index
        )
        pixel = pixel.to(dtype=self.dtype)
        shape = pixel.shape
        if shape[-1] == 3 and shape[-1] < shape[0] and shape[-1] < shape[1]:
            pixel = pixel.permute(2, 0, 1)  # HWC -> CHW

        return Entry(is_latent, pixel, prompt, original_size, None, dhdw, extras)

    def repeat_entries(self, k, res, index=None):
        repeat_strategy = self.kwargs.get("repeat_strategy", None)
        if repeat_strategy is not None:
            assert index is not None
            index_new = index.copy()
            for i, ent in enumerate(index):
                for strategy, mult in repeat_strategy:
                    if strategy in str(ent):
                        k.extend([k[i]] * (mult - 1))
                        res.extend([res[i]] * (mult - 1))
                        index_new.extend([index_new[i]] * (mult - 1))
                        break
        else:
            index_new = index
        return k, res, index_new

class NewLatentStore(StoreBase):
    def __init__(self, *args, **kwargs):
        logger.debug(f"loading latent codes from {self.root_path}")
        self.metadata = json_lib.load(open(os.path.join(self.root_path,"metadata.json")))
        latent_files = list(dirwalk(os.path.join(self.root_path,"latents"),lambda p: p.suffix == ".pt" and p.stem in metadata))
        self.latent_map = {}
        for p in latent_files:
            self.latent_map[p.stem] = p
        
        self.keys = self.data_repeat()
        self.length = len(self.keys)
        logger.debug(f"Loaded {self.length} latent codes from {self.root_path}")

    def get_raw_entry(self, index):
        key = self.keys[index]
        metadata = self.metadata[key]

    def apply_tag_dropout(self, entry, config=None):
            if config is None:
                config = {
                    "parquet": {
                        "character_drop": 0.1,
                        "core_whole_drop": 0.5,
                        "general_in_core_drop": 0.3,
                        "general_not_core_drop": 0.2,
                        "other_drop": 0.7,
                    },
                    "txt": {
                        "tag_drop": 0.5,
                    }
                }
    
            if entry.get("from_parquet") and "tags" in entry:
                t = entry["tags"]
                # logger.warning(f"{t}")
                source_type = entry.get("source_type", "dan") 
                final_components = []
    
                if source_type == "e621":
                    #  Artists: 全部丢弃
                    #  Year: 全部丢弃
                    #  Copyrights: 50% 概率全部丢弃
                    val_copy = t.get("copyright") or ""
                    if val_copy and random.random() >= 0.5:
                        final_components.append(val_copy)
                    # Characters: 20% 概率全部丢弃
                    val_char = t.get("character") or ""
                    if val_char and random.random() >= 0.2:
                        final_components.append(val_char)
                    #Species: 随机丢弃 20% tag
                    species_str = t.get("species") or ""
                    if species_str:
                        s_list = [s.strip() for s in species_str.split(",") if s.strip()]
                        kept_s = [s for s in s_list if random.random() >= 0.2]
                        if kept_s: final_components.append(", ".join(kept_s))
                    # General随机丢弃 70% tag
                    gen_str = t.get("general") or ""
                    if gen_str:
                        g_list = [g.strip() for g in gen_str.split(",") if g.strip()]
                        kept_g = [g for g in g_list if random.random() >= 0.7]
                        if kept_g: final_components.append(", ".join(kept_g))
                    # e621 tag: 20% 丢弃
                    val_e621 = t.get("e621_tag") or ""
                    if val_e621 and random.random() >= 0.2:
                        final_components.append(val_e621)
                    # 分辨率: 70% 概率丢弃
                    val_res = t.get("resolution") or ""
                    if val_res and random.random() >= 0.7:
                        final_components.append(val_res)
                    # NSFW  70% 丢弃
                    val_nsfw = t.get("nsfw") or ""
                    if val_nsfw and random.random() >= 0.7:
                        final_components.append(val_nsfw)
                    # logger.warning(f"e621_final:{",".join(final_components)}")
                    
                else:
                    cfg = config["parquet"]
                    
                    char_raw = t.get("character") or ""
                    chars = [c.strip() for c in str(char_raw).split(",") if c.strip()]
                    kept_chars = []
                    for c in chars:
                        count = self.real_char_counts.get(c, 0)
                        if count > 500:
                            prob = round(1.0 - (500.0 / count), 2)
                            prob = max(0.0, min(prob, 0.95))
                        else:
                            prob = cfg["character_drop"]
                        if random.random() >= prob:
                            kept_chars.append(c)
                    if kept_chars:
                        final_components.append(", ".join(kept_chars))
    
                    artist_raw = t.get("artist") or ""
                    artists = [a.strip() for a in str(artist_raw).split(",") if a.strip()]
                    kept_artists = []
                    for a in artists:
                        count = self.real_artist_counts.get(a, 0)
                        if count > 500:
                            prob = round(1.0 - (500.0 / count), 2)
                            prob = max(0.0, min(prob, 0.95))
                        else:
                            prob = cfg["character_drop"]
                        if random.random() >= prob:
                            kept_artists.append(a)
                    if kept_artists:
                        final_components.append(", ".join(kept_artists))
    
                    copy_val = t.get("copyright") or ""
                    if copy_val and random.random() >= cfg["character_drop"]:
                        final_components.append(copy_val)
    
                    general_str = t.get("general") or ""
                    if general_str:
                        gen_list = [tag.strip() for tag in general_str.split(",") if tag.strip()]
                        core_tags_str = entry.get("character_core_tags") or ""
                        core_set = set([tag.strip() for tag in str(core_tags_str).split(",") if tag.strip()])
                        in_core = [tag for tag in gen_list if tag in core_set]
                        not_in_core = [tag for tag in gen_list if tag not in core_set]
                        kept_general = []
                        if random.random() >= cfg["core_whole_drop"]:
                            kept_general.extend([tag for tag in in_core if random.random() >= cfg["general_in_core_drop"]])
                        kept_general.extend([tag for tag in not_in_core if random.random() >= cfg["general_not_core_drop"]])
                        if kept_general:
                            final_components.append(", ".join(kept_general))
    
                    meta_keys = ["rating", "year", "resolution", "nsfw", "aesthetics"]
                    for k in meta_keys:
                        val = t.get(k, "")
                        if val and random.random() >= cfg["other_drop"]:
                            final_components.append(str(val))
                    # logger.warning(f"dan_final:{",".join(final_components)}")
    
                final_components = [c.replace("_", " ") for c in final_components if c]
                random.shuffle(final_components)
                
                return ", ".join(final_components)
    
            else:
                original_prompt = entry.get("prompt", "")
                if not original_prompt:
                    return ""
                
                cfg_txt = config["txt"]
                tag_list = [tag.strip() for tag in original_prompt.split(",") if tag.strip()]
                kept_tags = [tag for tag in tag_list if random.random() >= cfg_txt["tag_drop"]]
                
                kept_tags = [t.replace("_", " ") for t in kept_tags]
                random.shuffle(kept_tags)
                
                final_txt_prompt = ", ".join(kept_tags)
                if random.random() < 0.01:
                    logger.warning(f"Fallback to TXT for {entry['path'].name}. Original: {original_prompt[:50]}...")
                return final_txt_prompt
            

    def data_repeat(self):
        keys = []
        for key in self.metadata:
            temp = [key] * temp["metadata"]["artist_repeat"]["character_repeat"]
            keys.extend(temp)
    def get_raw_entry(self,index):
        key = self.keys[index]
        latent = torch.load(self.latent_map[key])
        metadata = self.metadata[key]
        return True,latent,self.apply_tag_dropout(metadata),metadata['origin_size'],metadata['dhdw'],None
class LatentStore(StoreBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        prompt_mapping = next(dirwalk(self.root_path, lambda p: p.suffix == ".json"))
        prompt_mapping = json_lib.loads(Path(prompt_mapping).read_text())

        self.h5_paths = list(
            dirwalk(
                self.root_path,
                lambda p: p.suffix == ".h5" and "prompt_cache" not in p.stem,
            )
        )
        
        self.h5_keymap = {}
        self.h5_filehandles = {}
        self.paths = []
        self.keys = []
        progress = tqdm(
            total=len(prompt_mapping),
            desc=f"Loading latents",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        )

        has_h5_loc = "h5_path" in next(iter(prompt_mapping.values()))
        for idx, h5_path in enumerate(self.h5_paths):
            fs = h5.File(h5_path, "r", libver="latest")
            h5_name = h5_path.name
            
            for k in fs.keys():
                hashkey = k[:-8]  # ".latents"
                if hashkey not in prompt_mapping:
                    logger.warning(f"Key {k} not found in prompt_mapping")
                    continue
                
                it = prompt_mapping[hashkey]
                if not it["train_use"] or (has_h5_loc and it["h5_path"] != h5_name):
                    continue
                
                height, width, fp = it["train_height"], it["train_width"], it["file_path"]
                self.paths.append(fp)
                self.keys.append(k)
                self.raw_res.append((height, width))
                self.h5_keymap[k] = (h5_path, it, (height, width))
                progress.update(1)
                
        progress.close()
        self.length = len(self.keys)
        self.scale_factor = 0.13025
        logger.debug(f"Loaded {self.length} latent codes from {self.root_path}")

        self.keys, self.raw_res, self.paths = self.repeat_entries(self.keys, self.raw_res, index=self.paths)
        new_length = len(self.keys)
        if new_length != self.length:
            self.length = new_length
            logger.debug(f"Using {self.length} entries after applied repeat strategy")

    def setup_filehandles(self):
        self.h5_filehandles = {}
        for h5_path in self.h5_paths:
            self.h5_filehandles[h5_path] = h5.File(h5_path, "r", libver="latest")

    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, (int, int)]:
        if len(self.h5_filehandles) == 0:
            self.setup_filehandles()
            
        latent_key = self.keys[index]
        h5_path, entry, original_size = self.h5_keymap[latent_key]
        
        # modify here if you want to use a different format
        prompt = entry["train_caption"]
        latent = torch.asarray(self.h5_filehandles[h5_path][latent_key][:]).float()
        dhdw = self.h5_filehandles[h5_path][latent_key].attrs.get("dhdw", (0, 0))
    
        # if scaled, we need to unscale the latent (training process will scale it back)
        scaled = self.h5_filehandles[h5_path][latent_key].attrs.get("scale", True)
        if scaled:
            latent = 1.0 / self.scale_factor * latent

        extras = self.get_batch_extras(self.paths[index])
        return True, latent, prompt, original_size, dhdw, extras


class DirectoryImageStore(StoreBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        label_ext = self.kwargs.get("label_ext", ".txt")
        self.paths = list(dirwalk(self.root_path, is_img))
        self.length = len(self.paths)
        self.transforms = IMAGE_TRANSFORMS
        logger.debug(f"Found {self.length} images in {self.root_path}")

        remove_paths = []
        for p in tqdm(
            self.paths,
            desc="Loading image sizes",
            leave=False,
            ascii=True,
        ):
            try:
                w, h = Image.open(p).size
                self.raw_res.append((h, w))
            except Exception as e:
                print(f"\033[33mSkipped: error processing {p}: {e}\033[0m")
                remove_paths.append(p)

        remove_paths = set(remove_paths)
        self.paths = [p for p in self.paths if p not in remove_paths]
        self.length = len(self.raw_res)

        self.length = len(self.paths)
        self.prompts: list[str] = []
        for path in tqdm(
            self.paths,
            desc="Loading prompts",
            disable=self.rank != 0,
            leave=False,
            ascii=True,
        ):
            p = path.with_suffix(label_ext)
            try:
                with open(p, "r") as f:
                    self.prompts.append(f.read())
            except Exception as e:
                logger.warning(f"Skipped: error processing {p}: {e}")
                self.prompts.append("")
                
        self.prompts, self.raw_res, self.paths = self.repeat_entries(
            self.prompts, self.raw_res, index=self.paths
        )
        new_length = len(self.paths)
        if new_length != self.length:
            self.length = new_length
            logger.debug(f"Using {self.length} entries after applied repeat strategy")

    def get_raw_entry(self, index) -> tuple[bool, torch.tensor, str, (int, int)]:
        p = self.paths[index]
        prompt = self.prompts[index]
        _img = Image.open(p)
        if _img.mode == "RGB":
            img = np.array(_img)
        elif _img.mode == "RGBA":
            # transparent images
            baimg = Image.new('RGB', _img.size, (255, 255, 255))
            baimg.paste(_img, (0, 0), _img)
            img = np.array(baimg)
        else:
            img = np.array(_img.convert("RGB"))

        img = self.transforms(img)
        h, w = img.shape[-2:]
        dhdw = (0, 0)
        extras = self.get_batch_extras(p)
        return False, img, prompt, (h, w), dhdw, extras
