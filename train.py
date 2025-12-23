import importlib
import os
import shutil
import copy
import argparse
import random
import time
import datetime
import numpy as np
import cv2
import torch
import torchvision
import yaml
import math
from contextlib import nullcontext
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from utils import PerceptualLoss

# 사용자 라이브러리 (setup, metric_utils 등)
from setup import init_config
# WandB는 선택사항이므로 try-except 처리하거나 설치 필요
try:
    import wandb
except ImportError:
    wandb = None

# --- [1] 초기 설정 및 DDP Setup ---
config, config_name = init_config()
config_name = os.path.basename(config_name).split('.')[0]

# TF32 설정 (Ampere GPU 최적화)
torch.backends.cuda.matmul.allow_tf32 = config.train.get("use_tf32", True)
torch.backends.cudnn.allow_tf32 = config.train.get("use_tf32", True)

rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))
local_rank = int(os.environ.get("LOCAL_RANK", 0))

device = "cuda:{}".format(local_rank)
torch.cuda.set_device(device)
torch.cuda.empty_cache()

# Seed 설정
seed = 1111 + rank
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.distributed.init_process_group(backend='nccl')
torch.distributed.barrier()

# Logger 설정 (Rank 0만 출력)
def log_info(msg):
    if rank == 0:
        print(f"[Rank {rank}] {msg}")

log_info(f"Initialized DDP. World Size: {world_size}")

# --- [2] 데이터셋 및 로더 설정 ---
dataset_name = config.train.get("dataset_name", "data.dataset.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
Dataset = importlib.import_module(module).__dict__[class_name]
dataset = Dataset(config)

# 학습용 Sampler (Shuffle=True)
datasampler = DistributedSampler(dataset, shuffle=True, rank=rank, num_replicas=world_size)

batch_size_per_gpu = config.train.batch_size_per_gpu
dataloader = DataLoader(
    dataset, 
    batch_size=batch_size_per_gpu,
    shuffle=False, # Sampler가 있으므로 False
    num_workers=config.train.num_workers,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    prefetch_factor=config.train.get("prefetch_factor", 2),
    sampler=datasampler
)

# --- [1] 가중치 초기화 함수 (Weight Initialization) ---
# "initialize model weights using a zero-mean normal distribution with a std of 0.02"
def init_weights(m):
    if isinstance(m, nn.Linear):
        # Linear Layer 가중치 초기화
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # "Bias terms are omitted" -> 모델 정의 시 bias=False여야 하지만, 
        # 혹시 남아있다면 0으로 초기화하거나 무시
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    # LayerNorm은 보통 1.0(weight), 0.0(bias)으로 둡니다 (여기서는 언급 없으므로 default 유지)

# --- [2] Optimizer 설정 (Parameter Grouping) ---
# "Weight decay of 0.05 ... except the weights of LayerNorm"
def create_optimizer(model, config):
    # 파라미터를 두 그룹으로 분리
    decay_params = []
    no_decay_params = []
    
    # 제외할 레이어 타입 (LayerNorm)
    no_decay_classes = (nn.LayerNorm,)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            
            # LayerNorm의 파라미터이거나, 바이어스(Bias)인 경우 Weight Decay 제외
            # (Bias는 omitted라고 했지만, LayerNorm에는 bias가 있을 수 있으므로 포함)
            if isinstance(module, no_decay_classes) or param_name.endswith("bias"):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # 중복 방지를 위한 검증 (선택 사항)
    # param_dict = {id(p): p for p in model.parameters()}
    # assert len(decay_params) + len(no_decay_params) == len(param_dict.keys())

    optim_groups = [
        {
            "params": decay_params,
            "weight_decay": config.train.weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,  # LayerNorm 등은 0.0 적용
        },
    ]

    # AdamW 생성 (beta2 = 0.095 주의)
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=config.train.lr,
        betas=(config.train.beta1, config.train.beta2), # (0.9, 0.095)
        eps=1e-8
    )
    return optimizer

# --- [3] Scheduler 설정 (Warmup + Cosine) ---
# "Cosine learning rate schedule with ... warmup of 2500 iterations"
def create_scheduler(optimizer, num_training_steps, warmup_steps):
    def lr_lambda(current_step):
        # 1. Warmup 구간
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # 2. Cosine Decay 구간
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler

# --- [3] 모델 및 Optimizer 설정 ---
module, class_name = config.model.class_name.rsplit(".", 1)
ILRM = importlib.import_module(module).__dict__[class_name]
model = ILRM(config).to(device)

# SyncBatchNorm (배치 사이즈가 작을 때 유용)
if config.train.get("use_sync_bn", False):
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

# DDP Wrapper
# find_unused_parameters=True는 모델 구조에 따라 필요할 수 있음 (안전하게 True)
model = DDP(model, device_ids=[local_rank])

# Optimizer & Scheduler
optimizer = create_optimizer(model, config)

# 스텝 계산
train_steps = config.train.train_steps # 총 학습 스텝 수
grad_accum_steps = config.train.get("grad_accum_steps", 1) # 그래디언트 누적 스텝
param_update_steps = train_steps # 실제 업데이트 횟수 기준이라면

# Scheduler 
scheduler = create_scheduler(optimizer, config.train.train_steps, config.train.warmup_steps)

# AMP Scaler
enable_amp = config.train.get("use_amp", False)
amp_dtype = config.train.get("amp_dtype", "fp16")
amp_dtype_mapping = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
scaler = torch.cuda.amp.GradScaler(enabled=(enable_amp and amp_dtype == "fp16"))

# --- [4] Checkpoint Resume (자동 복구) ---
start_step = 0
ckpt_path = config.train.get("resume_ckpt", None)
checkpoint_dir = config.train.out_dir
os.makedirs(checkpoint_dir, exist_ok=True)

# 만약 resume_ckpt가 지정되지 않았으면, 최신 체크포인트 자동 검색
if ckpt_path is None and config.train.get("auto_resume", True):
    # checkpoint_dir에서 가장 최신 파일 찾기 로직 (간략화)
    ckpts = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if ckpts:
        ckpts.sort()
        ckpt_path = os.path.join(checkpoint_dir, ckpts[-1])

if ckpt_path:
    log_info(f"Resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(model, DDP):
        status = model.module.load_state_dict(checkpoint['model'], strict=False)
    else:
        status = model.load_state_dict(checkpoint['model'], strict=False)
    log_info(f"Loaded model state: {status}")

    if not config.train.get("reset_training_state", False):
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_step = checkpoint['step']
else:
    model.apply(init_weights)


# WandB Init (Rank 0)
if rank == 0 and config.train.get("use_wandb", False) and wandb:
    wandb.init(
        entity=config.train.wandb_entity,
        project=config.train.wandb_project,
        name=config_name,
        config=config,
        resume="allow",
        id=checkpoint.get("wandb_id", None) if ckpt_path else wandb.util.generate_id()
    )

# --- [5] 학습 루프 (Step-based) ---
log_info("Start Training...")
model.train()

# Iterator 생성
dataloader_iter = iter(dataloader)
step = start_step

# 현재 Epoch 계산 (Sampler 셔플용)
len_dataset = len(dataset)
total_batch_size = batch_size_per_gpu * world_size * grad_accum_steps
cur_epoch = (step * grad_accum_steps * batch_size_per_gpu * world_size) // len_dataset
datasampler.set_epoch(cur_epoch)

perceptual_loss_fn = PerceptualLoss(device, config)

while step < train_steps:
    step_start_time = time.time()
    
    # Gradient Accumulation Loop
    accum_loss = 0.0
    for accum_step in range(grad_accum_steps):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            cur_epoch += 1
            datasampler.set_epoch(cur_epoch)
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # 데이터 이동
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # DDP 최적화: 마지막 accum_step이 아니면 그래디언트 동기화(AllReduce)를 하지 않음
        # 이를 통해 통신 비용을 줄임
        do_sync = (accum_step == grad_accum_steps - 1)
        
        # context manager: no_sync()는 sync를 끔. do_sync가 True면 context를 안 씀(nullcontext)
        context = model.no_sync() if not do_sync else nullcontext()
        
        with context:
            with torch.autocast(enabled=enable_amp, device_type="cuda", dtype=amp_dtype_mapping[amp_dtype]):
                # Forward
                batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in batch.items()}
                input_data_dict = {key: value[:, :config.data.num_input_frames] if type(value) == torch.Tensor else value for key, value in batch.items()}
                target_data_dict = {key: value[:, config.data.num_input_frames:] if type(value) == torch.Tensor else None for key, value in batch.items()}
                ret_dict = model(input_data_dict, target_data_dict, 
                       save_video=config.inference.get("save_video")) 
                
                # Compute Loss
                target_render = ret_dict['render'].reshape(-1, ret_dict['render'].shape[-3], ret_dict['render'].shape[-2], ret_dict['render'].shape[-1])
                target_gt = target_data_dict['image'].reshape(-1, target_data_dict['image'].shape[-3], target_data_dict['image'].shape[-2], target_data_dict['image'].shape[-1])
                l2_loss = nn.functional.mse_loss(target_render, target_gt)
                perceptual_loss = perceptual_loss_fn(target_render, target_gt)
                total_loss = l2_loss + config.train.get("perceptual_loss_weight", 0.0) * perceptual_loss
                
                # Normalize loss for accumulation
                total_loss = total_loss / grad_accum_steps
            
            # Backward
            scaler.scale(total_loss).backward()
            accum_loss += total_loss.item()

    # --- Optimizer Step ---
    # Unscale Gradients (Clipping을 위해 미리 unscale)
    scaler.unscale_(optimizer)
    
    # Gradient Clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config.train.get("grad_clip_norm", 1.0)
    )
    
    # Skip Step if NaN/Inf
    if not torch.isfinite(grad_norm):
        log_info(f"Skipping step {step}: Gradient norm is {grad_norm}")
        optimizer.zero_grad()
    else:
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
    
    step_time = time.time() - step_start_time
    
    # --- Logging & Checkpoint ---
    if rank == 0:
        if step % config.train.print_every == 0:
            print(f"Step {step}/{train_steps} | Epoch {cur_epoch} | Loss: {accum_loss:.4f} | Grad: {grad_norm:.2f} | Time: {step_time:.3f}s")
            
        if step % config.train.get("wandb_every", 100) == 0 and wandb:
            wandb.log({
                "train/loss": accum_loss,
                "train/grad_norm": grad_norm,
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/step_time": step_time,
                "train/epoch": cur_epoch
            }, step=step)
            
        if step % config.train.save_interval == 0 and step > 0:
            save_path = os.path.join(checkpoint_dir, f"ckpt_{step:06d}.pt")
            torch.save({
                'step': step,
                'model': model.module.state_dict(), # module 주의
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config,
                'wandb_id': wandb.run.id if wandb else None
            }, save_path)
            log_info(f"Saved checkpoint to {save_path}")

    step += 1

torch.distributed.barrier()
destroy_process_group()