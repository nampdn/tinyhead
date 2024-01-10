import os
import sys
import math
import glob
import time
from functools import partial
from pathlib import Path
from typing import Tuple, Optional

import lightning as L
from lightning.fabric.strategies import FSDPStrategy

import torch
from torch.utils.data import DataLoader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.models.config_mamba import MambaConfig

from mockup_data import MockNextTokenPredictionDataset

out_dir = "out/training"
eval_interval = 2000
eval_iters = 200
log_interval = 1

# Hyperparameters
learning_rate = 5e-5
batch_size = 125
micro_batch_size = 5
max_iters = 600_000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 1e-5
block_size = 128


def main(
    devices: int = 4,
    train_data_dir: Path = "data/train",
    val_data_dir: Optional[Path] = None,
) -> None:
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    print(auto_wrap_policy)
    strategy = FSDPStrategy(
        auto_wrap_policy=auto_wrap_policy, activation_checkpointing=Block, limit_all_gathers=True
    )
    fabric = L.Fabric(accelerator="cuda", devices=4, precision="bf16-mixed")

    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    
    model = MambaLMHeadModel.from_pretrained("./vinamamba-130m")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False
    )
    
    model, optimizer = fabric.setup(model, optimizer)

    process_batch_size = batch_size // devices
    gradient_accumulation_iters = process_batch_size // micro_batch_size

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size = micro_batch_size,
        block_size = block_size + 1,
        fabric = fabric,
        train_data_dir = train_data_dir,
        val_data_dir = val_data_dir,
    )
    
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    train(fabric, model, optimizer, train_dataloader, None, gradient_accumulation_iters, devices)

def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader],
    grad_accum_steps: int,
    devices: int
) -> None:
    step_count = 0
    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    vocab_size = 46303
    prev_t1 = time.time()

    for iter_num, train_data in enumerate(train_dataloader):
        t0 = time.time()

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids = train_data[:, 0 : block_size].contiguous()
        targets = train_data[:, 1 : block_size + 1].contiguous()
        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        with fabric.no_backward_sync(model, enabled=is_accumulating):
            outputs = model(input_ids, num_last_tokens=block_size)
            
            logits = outputs[0]
            # shift_logits = logits[..., :-1, :].contiguous()
            # shift_labels = input_ids[..., 1:].contiguous()

            # print(shift_logits.shape, shift_labels.shape)
            
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            fabric.backward(loss / grad_accum_steps)

        t1 = time.time()

        dt = t1 - t0

        tokens += micro_batch_size * block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num & log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            fabric.log_dict(
                {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}
            )

            fabric.print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms, speed: {tokens_sec_str} toks/s/device"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > max_iters:
            break

def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric: L.Fabric,
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    seed: int = 12345
) -> DataLoader:
    num_samples = 1_000_000
    vocab_size = 46303
    dataset = MockNextTokenPredictionDataset(num_samples=num_samples, vocab_size=vocab_size, sequence_length=block_size)
   
    # Create a DataLoader
    batch_size = 32
    shuffle = True
    num_workers = 4  # Adjust this according to your machine's capabilities
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader, None


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    if it > lr_decay_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

if __name__ ==  "__main__":
    torch.set_float32_matmul_precision("high")
    from jsonargparse.cli import CLI

    CLI(main)
