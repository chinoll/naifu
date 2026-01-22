"""
SDXL Training with Diffusers + Accelerate

A minimal, clean implementation for SDXL fine-tuning using:
- 🤗 Diffusers for model components
- 🤗 Accelerate for distributed training (including DeepSpeed ZeRO)
- SimpleLatentDataset for pre-encoded latents

Usage:
    # Single GPU
    python train_sdxl.py --config config/train_sdxl.yaml
    
    # Multi-GPU with Accelerate
    accelerate launch train_sdxl.py --config config/train_sdxl.yaml
    
    # With DeepSpeed
    accelerate launch --config_file accelerate_deepspeed.yaml train_sdxl.py --config config/train_sdxl.yaml
"""

import os
import sys
import math
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    UNet2DConditionModel,
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from accelerate.state import AcceleratorState
from transformers import CLIPTextModel, CLIPTextModelWithProjection, AutoTokenizer
from safetensors.torch import load_file, save_file

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from training_diffusers.dataset import SimpleLatentDataset, SimpleLatentDatasetForHttp

logger = get_logger(__name__)


# ============================================================================
# SNR Utils (Min-SNR weighting)
# ============================================================================

def cache_snr_values(noise_scheduler, device):
    """Cache SNR values for min-SNR weighting"""
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2
    noise_scheduler.all_snr = all_snr.to(device)


def apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=False):
    """Apply min-SNR weighting to loss"""
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    min_snr_gamma = torch.minimum(snr, torch.full_like(snr, gamma))
    if v_prediction:
        snr_weight = torch.div(min_snr_gamma, snr + 1).float().to(loss.device)
    else:
        snr_weight = torch.div(min_snr_gamma, snr).float().to(loss.device)
    return loss * snr_weight

# ============================================================================
# Model Loading
# ============================================================================

def load_models(model_path: str):
    """Load SDXL models from HuggingFace format or single file
    
    Args:
        model_path: Can be:
            - HuggingFace Hub ID: "stabilityai/stable-diffusion-xl-base-1.0"
            - Local Diffusers directory: "/path/to/diffusers_model/"
            - Single file: "/path/to/model.safetensors" or "/path/to/model.ckpt"
    
    Returns:
        tuple: (unet, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler)
    """
    is_single_file = model_path.endswith((".safetensors", ".ckpt"))
    
    if is_single_file:
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float32,
        )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
        )
    
    # Extract components
    unet = pipe.unet
    vae = pipe.vae
    text_encoder_1 = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    tokenizer_1 = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    noise_scheduler = pipe.scheduler
    
    # Ensure DDPMScheduler for training
    if not isinstance(noise_scheduler, DDPMScheduler):
        noise_scheduler = DDPMScheduler.from_config(noise_scheduler.config)
    
    del pipe
    torch.cuda.empty_cache()
    
    return unet, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler


# ============================================================================
# Text Encoding
# ============================================================================

def encode_prompt(
    prompts: list[str],
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    device,
    max_length: int = 77,
):
    """Encode prompts using both CLIP text encoders for SDXL"""
    if max_length <= 77:
        # Tokenize
        tokens_1 = tokenizer_1(
            prompts, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        
        tokens_2 = tokenizer_2(
            prompts, padding="max_length", max_length=max_length,
            truncation=True, return_tensors="pt"
        ).input_ids.to(device)
        
        # Encode with text_encoder_1 (CLIP-ViT-L)
        with torch.no_grad():
            encoder_output_1 = text_encoder_1(tokens_1, output_hidden_states=True)
            hidden_states_1 = encoder_output_1.hidden_states[-2]  # penultimate layer
        
        # Encode with text_encoder_2 (CLIP-ViT-bigG)
        with torch.no_grad():
            encoder_output_2 = text_encoder_2(tokens_2, output_hidden_states=True)
            hidden_states_2 = encoder_output_2.hidden_states[-2]
            pooled_output = encoder_output_2.text_embeds
        
        # Concatenate hidden states [batch, seq_len, 768 + 1280]
        prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
        
        return prompt_embeds, pooled_output

    # Long prompt encoding support
    tokens_1 = tokenizer_1(
        prompts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)
    
    tokens_2 = tokenizer_2(
        prompts, padding="max_length", max_length=max_length,
        truncation=True, return_tensors="pt"
    ).input_ids.to(device)

    def process_encoder(tokens, encoder, tokenizer):
        batch_size = tokens.shape[0]
        chunk_size = 75 # 77 - 2
        num_chunks = math.ceil((max_length - 2) / chunk_size)
        
        # Reshape to list of chunks [batch, 77]
        input_ids_chunks = []
        
        for i in range(num_chunks):
            start = 1 + i * chunk_size
            end = min(1 + (i + 1) * chunk_size, max_length - 1)
            
            chunk = tokens[:, start:end]
            
            # Pad
            if chunk.shape[1] < chunk_size:
                pad = torch.full((batch_size, chunk_size - chunk.shape[1]), tokenizer.pad_token_id, device=device, dtype=tokens.dtype)
                chunk = torch.cat([chunk, pad], dim=1)
                
            # Add BOS/EOS
            bos = torch.full((batch_size, 1), tokenizer.bos_token_id, device=device, dtype=tokens.dtype)
            eos = torch.full((batch_size, 1), tokenizer.eos_token_id, device=device, dtype=tokens.dtype)
            
            chunk_full = torch.cat([bos, chunk, eos], dim=1)
            input_ids_chunks.append(chunk_full)
            
        # Concat chunks -> [batch * num_chunks, 77]
        input_ids_all = torch.cat(input_ids_chunks, dim=0)
        
        with torch.no_grad():
            output = encoder(input_ids_all, output_hidden_states=True)
            
        if hasattr(output, "text_embeds"):
            # TE2
            hidden_states = output.hidden_states[-2]
            pooled = output.text_embeds
        else:
            # TE1
            hidden_states = output.hidden_states[-2]
            pooled = None
            
        # Unpack hidden states [batch * num_chunks, 77, dim] -> [num_chunks, batch, 77, dim]
        hidden_states = hidden_states.view(num_chunks, batch_size, 77, -1)
        
        states_list = []
        # BOS
        states_list.append(hidden_states[0, :, 0:1, :])
        
        for i in range(num_chunks):
            states_list.append(hidden_states[i, :, 1:76, :]) # Take 75 tokens
            
        states = torch.cat(states_list, dim=1) # [batch, 1 + num_chunks * 75, dim]
        
        # Truncate to max_length
        if states.shape[1] > max_length:
            states = states[:, :max_length, :]
            
        # Handle pooled
        if pooled is not None:
             # Pooled is [batch * num_chunks, dim] -> [num_chunks, batch, dim]
             pooled = pooled.view(num_chunks, batch_size, -1)
             pooled = pooled[0] # Use first chunk for pooled representation
             
        return states, pooled

    hidden_states_1, _ = process_encoder(tokens_1, text_encoder_1, tokenizer_1)
    hidden_states_2, pooled_output = process_encoder(tokens_2, text_encoder_2, tokenizer_2)
    
    prompt_embeds = torch.cat([hidden_states_1, hidden_states_2], dim=-1)
    return prompt_embeds, pooled_output


def compute_time_ids(original_size, crop_coords, target_size, device, dtype):
    """Compute SDXL time embeddings"""
    # original_size: (batch, 2), crop_coords: (batch, 2), target_size: (batch, 2)
    add_time_ids = torch.cat([original_size, crop_coords, target_size], dim=-1)
    return add_time_ids.to(device=device, dtype=dtype)


# ============================================================================
# Training Step
# ============================================================================

def training_step(
    batch,
    unet,
    vae,
    text_encoder_1,
    text_encoder_2,
    tokenizer_1,
    tokenizer_2,
    noise_scheduler,
    config,
    accelerator,
    weight_dtype,
):
    """Single training step"""
    advanced = config.get("advanced", {})
    
    # Get latents
    if batch["is_latent"]:
        latents = batch["pixels"].to(weight_dtype)
        # Normalize if needed (VAE scaling)
        latents = latents * vae.config.scaling_factor
    else:
        # Encode images to latents
        with torch.no_grad():
            latents = vae.encode(batch["pixels"].to(vae.dtype)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
    bsz = latents.shape[0]
    # Encode prompts
    prompt_embeds, pooled_embeds = encode_prompt(
        batch["prompts"],
        tokenizer_1, tokenizer_2,
        text_encoder_1, text_encoder_2,
        device=accelerator.device,
        max_length=config.dataset.get("max_token_length", 77),
    )
    prompt_embeds = prompt_embeds.to(weight_dtype)
    pooled_embeds = pooled_embeds.to(weight_dtype)
    
    # CFG training: randomly drop conditioning (condition dropout)
    cfg_dropout_rate = advanced.get("condition_dropout_rate", 0.0)
    if cfg_dropout_rate > 0.0:
        # Create mask for dropping entire batch samples' conditioning
        drop_mask = torch.rand(bsz, device=accelerator.device) < cfg_dropout_rate
        if drop_mask.any():
            # Zero out embeddings for dropped samples
            prompt_embeds[drop_mask] = 0.0
            pooled_embeds[drop_mask] = 0.0
    
    # Sample noise
    noise = torch.randn_like(latents)
    if advanced.get("offset_noise", False):
        offset = torch.randn(latents.shape[0], latents.shape[1], 1, 1, device=latents.device)
        noise = noise + advanced.get("offset_noise_val", 0.0375) * offset
    
    # Sample timesteps
    bsz = latents.shape[0]
    timestep_start = advanced.get("timestep_start", 0)
    timestep_end = advanced.get("timestep_end", 1000)
    
    if advanced.get("timestep_sampler_type") == "logit_normal":
        mu = advanced.get("timestep_sampler_mean", 0)
        sigma = advanced.get("timestep_sampler_std", 1)
        t = torch.sigmoid(mu + sigma * torch.randn(size=(bsz,), device=latents.device))
        timesteps = (t * (timestep_end - timestep_start) + timestep_start).long()
    else:
        timesteps = torch.randint(timestep_start, timestep_end, (bsz,), device=latents.device, dtype=torch.long)
    
    # Add noise
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    
    # Time embeddings for SDXL
    add_time_ids = compute_time_ids(
        batch["original_size_as_tuple"].float(),
        batch["crop_coords_top_left"].float(),
        batch["target_size_as_tuple"].float(),
        device=accelerator.device,
        dtype=weight_dtype,
    )
    
    # Predict noise
    model_pred = unet(
        noisy_latents.to(weight_dtype),
        timesteps,
        encoder_hidden_states=prompt_embeds,
        added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
        return_dict=False,
    )[0]
    
    # Get target
    if advanced.get("v_parameterization", False):
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise
    
    # Compute loss
    if advanced.get("min_snr", False):
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=[1, 2, 3])
        loss = apply_snr_weight(loss, timesteps, noise_scheduler, advanced.get("min_snr_val", 5), advanced.get("v_parameterization", False))
        loss = loss.mean()
    else:
        loss = F.mse_loss(model_pred.float(), target.float())
    
    return loss


# ============================================================================
# Sampling
# ============================================================================

@torch.inference_mode()
def generate_samples(
    unet, vae, text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, noise_scheduler,
    prompts, config, accelerator, epoch, step, save_dir, max_length=77,
):
    """Generate sample images for validation"""
    from diffusers import EulerDiscreteScheduler
    
    if not accelerator.is_main_process:
        return
    
    unet.eval()
    sampling_cfg = config.sampling
    device = accelerator.device
    dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
    
    # Use Euler scheduler for sampling
    sampling_scheduler = EulerDiscreteScheduler.from_config(noise_scheduler.config)
    
    height = sampling_cfg.get("height", 1024)
    width = sampling_cfg.get("width", 1024)
    num_steps = sampling_cfg.get("steps", 25)
    guidance_scale = sampling_cfg.get("guidance_scale", 7.0)
    
    generator = torch.Generator(device="cpu").manual_seed(sampling_cfg.get("seed", 42))
    os.makedirs(save_dir, exist_ok=True)
    
    for idx, prompt in enumerate(prompts):
        # Encode prompt
        prompt_embeds, pooled_embeds = encode_prompt(
            [prompt, ""],  # prompt + negative
            tokenizer_1, tokenizer_2,
            text_encoder_1, text_encoder_2,
            device=device,
            max_length=max_length,
        )
        prompt_embeds = prompt_embeds.to(dtype)
        pooled_embeds = pooled_embeds.to(dtype)
        
        # Time IDs
        add_time_ids = torch.tensor([[height, width, 0, 0, height, width]], device=device, dtype=dtype)
        add_time_ids = add_time_ids.repeat(2, 1)
        
        # Initial latents
        latents = torch.randn((1, 4, height // 8, width // 8), generator=generator, device="cpu")
        latents = latents.to(device=device, dtype=dtype) * sampling_scheduler.init_noise_sigma
        
        sampling_scheduler.set_timesteps(num_steps, device=device)
        
        for t in sampling_scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)
            latent_input = sampling_scheduler.scale_model_input(latent_input, t)
            
            noise_pred = unet(
                latent_input,
                t,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
                return_dict=False,
            )[0]
            
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            
            latents = sampling_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        # Decode
        latents = latents / vae.config.scaling_factor
        with torch.autocast(device.type, enabled=False):
            image = vae.decode(latents, return_dict=False)[0]
        
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")[0]
        
        from PIL import Image
        Image.fromarray(image).save(f"{save_dir}/sample_e{epoch}_s{step}_{idx}.png")
        logger.info(f"Saved sample: {save_dir}/sample_e{epoch}_s{step}_{idx}.png")
    
    unet.train()


# ============================================================================
# Checkpoint Saving
# ============================================================================

def save_checkpoint(
    unet, text_encoder_1, text_encoder_2, vae,
    tokenizer_1, tokenizer_2, noise_scheduler,
    accelerator, config, epoch, step, output_dir,
):
    """Save checkpoint: Accelerate State (for resume) + Single File (for inference)"""
    
    save_dir = Path(output_dir) / f"checkpoint-e{epoch}_s{step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Accelerate State (Optimizer, LR Scheduler, etc.) - Must be called by ALL processes
    accelerator.save_state(save_dir)
    
    # 2. Main Process Only: Save Single File
    if accelerator.is_main_process:
        # Save single file (.safetensors)
        try:
            unwrapped_unet = accelerator.unwrap_model(unet)
            pipeline = StableDiffusionXLPipeline(
                vae=accelerator.unwrap_model(vae),
                text_encoder=accelerator.unwrap_model(text_encoder_1),
                text_encoder_2=accelerator.unwrap_model(text_encoder_2),
                tokenizer=None,
                tokenizer_2=None,
                unet=unwrapped_unet,
                scheduler=None,
            )
            pipeline.save_single_file(save_dir / "model.safetensors")
            logger.info(f"Saved checkpoint and single-file to {save_dir}")
        except Exception as e:
            logger.warning(f"Failed to save single file: {e}")

    # Ensure all processes wait for saving to complete
    accelerator.wait_for_everyone()


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    cfg = config.trainer
    
    # Accelerator setup
    project_config = ProjectConfiguration(
        project_dir=cfg.checkpoint_dir,
        logging_dir=cfg.get("logging_dir", "logs"),
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.get("accumulate_grad_batches", 1),
        mixed_precision=config.get("accelerate", {}).get("mixed_precision", "bf16"),
        log_with="wandb" if cfg.get("wandb_id") else None,
        project_config=project_config,
    )
    # Logging
    AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.trainer.batch_size
    if accelerator.is_main_process:
        if cfg.get("wandb_id"):
            accelerator.init_trackers(project_name=cfg.wandb_id, config=OmegaConf.to_container(config))
    
    set_seed(cfg.seed)
    weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16
    
    # Load models
    logger.info(f"Loading models from {cfg.model_path}")
    unet, vae, te1, te2, tok1, tok2, scheduler = load_models(cfg.model_path)
    
    # Cache SNR values for min-SNR weighting
    if config.advanced.get("min_snr", False):
        cache_snr_values(scheduler, accelerator.device)
    
    # Freeze VAE
    vae.requires_grad_(False)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # Freeze text encoders if not training
    if not config.advanced.get("train_text_encoder_1", False):
        te1.requires_grad_(False)
        te1.to(accelerator.device, dtype=weight_dtype)
    if not config.advanced.get("train_text_encoder_2", False):
        te2.requires_grad_(False)
        te2.to(accelerator.device, dtype=weight_dtype)
    
    # Enable gradient checkpointing
    if config.advanced.get("gradient_checkpointing", True):
        unet.enable_gradient_checkpointing()
    
    # Enable xformers memory efficient attention
    if config.advanced.get("use_xformers", False):
        try:
            unet.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers memory efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable xformers: {e}")
    
    # Dataset
    if config.dataset.get("server_url"):
        logger.info(f"Using HTTP dataset: {config.dataset.server_url}")
        dataset = SimpleLatentDatasetForHttp(
            batch_size=cfg.batch_size,
            rank=accelerator.process_index,
            **config.dataset,
        )
    else:
        logger.info(f"Using local dataset: {config.dataset.get('data_root', config.dataset.get('img_path'))}")
        dataset = SimpleLatentDataset(
            batch_size=cfg.batch_size,
            rank=accelerator.process_index,
            **config.dataset,
        )
    dataloader = dataset.init_dataloader()
    print("dataloader length",len(dataloader))
    print(len(dataloader))
    
    # Optimizer - support torch.optim, bitsandbytes, etc.
    trainable_params = list(unet.parameters())
    param_groups = [{"params": trainable_params, "lr": config.optimizer.params.lr}]
    
    if config.advanced.get("train_text_encoder_1", False):
        lr = config.advanced.get("text_encoder_1_lr", config.optimizer.params.lr)
        param_groups.append({"params": list(te1.parameters()), "lr": lr})
    if config.advanced.get("train_text_encoder_2", False):
        lr = config.advanced.get("text_encoder_2_lr", config.optimizer.params.lr)
        param_groups.append({"params": list(te2.parameters()), "lr": lr})
    
    optimizer_name = config.optimizer.name
    optimizer_params = {k: v for k, v in config.optimizer.params.items() if k != "lr"}
    
    if "bitsandbytes" in optimizer_name:
        import bitsandbytes as bnb
        optimizer_cls = getattr(bnb.optim, optimizer_name.split(".")[-1])
    elif "torch.optim" in optimizer_name:
        optimizer_cls = getattr(torch.optim, optimizer_name.split(".")[-1])
    else:
        import importlib
        module_path, class_name = optimizer_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        optimizer_cls = getattr(module, class_name)
    
    optimizer = optimizer_cls(param_groups, **optimizer_params)
    
    # Learning rate scheduler
    lr_scheduler = None
    if config.get("scheduler"):
        from transformers import get_scheduler
        num_training_steps = cfg.max_epochs * len(dataloader) // cfg.get("accumulate_grad_batches", 1)
        lr_scheduler = get_scheduler(
            config.scheduler.name.split(".")[-1].replace("get_", "").replace("_schedule_with_warmup", "").replace("_schedule", ""),
            optimizer=optimizer,
            num_warmup_steps=config.scheduler.params.get("num_warmup_steps", 0),
            num_training_steps=num_training_steps,
        )
    
    # Prepare with Accelerate
    # no accelerator.prepare dataloader
    unet, optimizer = accelerator.prepare(unet, optimizer)
    if config.advanced.get("train_text_encoder_1", False):
        te1 = accelerator.prepare(te1)
    if config.advanced.get("train_text_encoder_2", False):
        te2 = accelerator.prepare(te2)
    if lr_scheduler:
        lr_scheduler = accelerator.prepare(lr_scheduler)
    
    # Training state
    global_step = 0
    start_epoch = 0
    
    # Resume
    if args.resume:
        dirs = sorted(Path(cfg.checkpoint_dir).glob("checkpoint-*"), key=lambda x: x.stat().st_mtime)
        if dirs:
            accelerator.load_state(str(dirs[-1]))
            logger.info(f"Resumed from {dirs[-1]}")
    
    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num epochs = {cfg.max_epochs}")
    logger.info(f"  Batch size = {cfg.batch_size}")
    logger.info(f"  Gradient accumulation steps = {cfg.get('accumulate_grad_batches', 1)}")
    
    progress_bar = tqdm(range(cfg.max_epochs * len(dataloader)), disable=not accelerator.is_main_process)
    grad_norm = None
    for epoch in range(start_epoch, cfg.max_epochs):
        unet.train()
        
        for batch in dataloader:
            for key in batch.keys():
                if type(batch[key]) == torch.Tensor:
                    batch[key] = batch[key].to(unet.device)
            with accelerator.accumulate(unet):
                loss = training_step(
                    batch, unet, vae, te1, te2, tok1, tok2, scheduler,
                    config, accelerator, weight_dtype,
                )
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    if cfg.get("gradient_clip_val", 0) > 0:
                        grad_norm = accelerator.clip_grad_norm_(unet.parameters(), cfg.gradient_clip_val)
                
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=loss.item(), epoch=epoch, step=global_step)
                
                # Logging
                if accelerator.is_main_process:
                    if grad_norm is None:
                        accelerator.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=global_step)
                    else:
                        accelerator.log({"loss":loss.item(),"lr":optimizer.param_groups[0]["lr"],"grad_norm":grad_norm},step=global_step)

                # Checkpoint by steps
                if cfg.get("checkpoint_steps", -1) > 0 and global_step % cfg.checkpoint_steps == 0:
                    save_checkpoint(unet, te1, te2, vae, tok1, tok2, scheduler, accelerator, config, epoch, global_step, cfg.checkpoint_dir)
                
                # Sampling by steps
                sampling_cfg = config.get("sampling", {})
                if sampling_cfg.get("enabled") and sampling_cfg.get("every_n_steps", -1) > 0:
                    if global_step % sampling_cfg.every_n_steps == 0:
                        generate_samples(
                            unet, vae, te1, te2, tok1, tok2, scheduler,
                            sampling_cfg.prompts, config, accelerator, epoch, global_step,
                            sampling_cfg.get("save_dir", "samples"),
                            max_length=config.dataset.get("max_token_length", 77),
                        )
        
        # End of epoch
        # Checkpoint by epochs
        if cfg.get("checkpoint_freq", -1) > 0 and (epoch + 1) % cfg.checkpoint_freq == 0:
            save_checkpoint(unet, te1, te2, vae, tok1, tok2, scheduler, accelerator, config, epoch, global_step, cfg.checkpoint_dir)
        
        # Sampling by epochs  
        if sampling_cfg.get("enabled") and sampling_cfg.get("every_n_epochs", -1) > 0:
            if (epoch + 1) % sampling_cfg.every_n_epochs == 0:
                generate_samples(
                    unet, vae, te1, te2, tok1, tok2, scheduler,
                    sampling_cfg.prompts, config, accelerator, epoch, global_step,
                    sampling_cfg.get("save_dir", "samples"),
                    max_length=config.dataset.get("max_token_length", 77),
                )
    
    accelerator.end_training()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
