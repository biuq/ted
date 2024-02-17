# Credits: https://github.com/karpathy/ng-video-lecture

from dataclasses import dataclass

import math
import random
import os
import signal
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from ted.config import TrainConfig
from ted.data import load_dataset
from ted.model import TED, ModelConfig
from ted.vocab import Vocab, tokenize

experiment_name = "tiny_stories"

interrupted = False
def signal_handler(signal, frame):
    global interrupted
    interrupted = True
    print("Interrupt caught, will finish after this epoch")
signal.signal(signal.SIGINT, signal_handler)

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIRECTORY)
VOCAB_PATH = os.path.join(ROOT_DIRECTORY, experiment_name, 'vocab.json')
WEIGHTS_PATH_JSON = os.path.join(ROOT_DIRECTORY, experiment_name, 'weights.json')
WEIGHTS_PATH_PICKLE = os.path.join(ROOT_DIRECTORY, experiment_name, 'weights.pth')
MODEL_CONFIG_PATH_JSON = os.path.join(ROOT_DIRECTORY, experiment_name, 'model.json')
TRAIN_CONFIG_PATH = os.path.join(ROOT_DIRECTORY, experiment_name, 'train.json')

train_cfg = TrainConfig().load(TRAIN_CONFIG_PATH)
generator = torch.manual_seed(train_cfg.random_seed)
random.seed(train_cfg.random_seed)

print('Loading data...')

train_text_data, val_text_data = load_dataset(experiment_name)
if train_cfg.load_vocab:
    vocab = Vocab.load(VOCAB_PATH)
else:
    vocab = tokenize(train_text_data)
vocab_size = len(vocab)

if not train_cfg.load_vocab:
    vocab.save(VOCAB_PATH)

train_data = torch.tensor(vocab.encode(train_text_data), dtype=torch.long)
val_data = torch.tensor(vocab.encode(val_text_data), dtype=torch.long)

def train():
    model_cfg = ModelConfig().load(MODEL_CONFIG_PATH_JSON)
    if train_cfg.load_weights:
        model_cfg = ModelConfig().load(MODEL_CONFIG_PATH_JSON)
    else:
        model_cfg.vocab_size = vocab_size
        model_cfg.save(MODEL_CONFIG_PATH_JSON)
    model = TED(model_cfg)
    if train_cfg.load_weights:
        model.load_pickle(WEIGHTS_PATH_PICKLE)
    model = model.to(train_cfg.device)

    total_params = 0
    trainable_params = 0
    total_params += sum(param.numel() for param in model.parameters())
    trainable_params += sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'Total number of parameters: {total_params / 1000000:.6f} M')
    print(f'Number of trainable parameters: {trainable_params / 1000000:.6f} M')
    print('')
    
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    lr = calc_lr(0)
    update_lr(optim, lr)
    
    losses = estimate_loss(model, generator)
    print(f"step {0}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {lr:.4E}")

    for iter in range(1, train_cfg.max_iters + 1):
        xb, yb = get_batch('train', train_cfg.train_sequence_length, train_cfg.batch_size, generator)

        logits = model.forward(xb)
        loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten(0, 1))
        loss.backward()

        nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optim.step()
        optim.zero_grad()

        lr = calc_lr(iter)
        update_lr(optim, lr)

        if iter % train_cfg.eval_interval == 0 or iter == train_cfg.max_iters:
            losses = estimate_loss(model, generator)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, lr {calc_lr(iter):.4E}")

        if iter % train_cfg.checkpoint_interval == 0 or iter == train_cfg.max_iters:
            print('Checkpoint...')
            model.save_json(WEIGHTS_PATH_JSON)
            model.save_pickle(WEIGHTS_PATH_PICKLE)

        if interrupted:
            break

def linear_progress(i: int, n: int):
    return i / n

def cos_decay(i: int, n: int):
    return (math.cos(linear_progress(i, n) * math.pi) + 1) / 2

def lr_schedule(start: float, peak: float, end: float, warmup_iters: int, total_iters: int, i: int):
    if i < warmup_iters:
        return start + (peak - start) * linear_progress(i, warmup_iters)
    return (peak - end) * cos_decay(i - warmup_iters, total_iters - warmup_iters) + end

def update_lr(optim: torch.optim.Optimizer, lr: float):
    for param_group in optim.param_groups:
        param_group['lr'] = lr
        
def calc_lr(i: int):
    return lr_schedule(
        train_cfg.start_learning_rate,
        train_cfg.peak_learning_rate, 
        train_cfg.end_learning_rate, 
        train_cfg.warmup_iters(), 
        train_cfg.max_iters,
        i
    )

def get_batch(split, sequence_length, batch_size, generator):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - sequence_length, (batch_size,), generator=generator)
    x = torch.stack([data[i:i+sequence_length] for i in ix])
    y = torch.stack([data[i+1:i+sequence_length+1] for i in ix])
    x, y = x.to(train_cfg.device), y.to(train_cfg.device)
    return x, y

@torch.inference_mode()
def estimate_loss(model: nn.Module, generator):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(train_cfg.eval_iters)
        for k in range(train_cfg.eval_iters):
            X, Y = get_batch(split, train_cfg.train_sequence_length, train_cfg.batch_size, generator)
            logits = model.forward(X)
            losses[k] = F.cross_entropy(logits.flatten(0, 1), Y.flatten(0, 1))
        out[split] = losses.mean()
    model.train()
    return out

train()
