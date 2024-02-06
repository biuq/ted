from dataclasses import dataclass

import torch
import os
import sys
from ted.config import GenConfig

ROOT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIRECTORY)

from ted.prob import sample
from ted.model import TED, ContextConfig, ModelConfig
from ted.vocab import Vocab

experiment_name = "tiny_stories"
prompt = "Once upon a time"

VOCAB_PATH = os.path.join(ROOT_DIRECTORY, experiment_name, 'vocab.json')
WEIGHTS_PATH_PICKLE = os.path.join(ROOT_DIRECTORY, experiment_name, 'weights.pth')
MODEL_CONFIG_PATH_JSON = os.path.join(ROOT_DIRECTORY, experiment_name, 'model.json')
GEN_CONFIG_PATH_JSON = os.path.join(ROOT_DIRECTORY, experiment_name, 'gen.json')
CONTEXT_CONFIG_PATH_JSON = os.path.join(ROOT_DIRECTORY, experiment_name, 'context.json')

gen_cfg = GenConfig().load(GEN_CONFIG_PATH_JSON)
context_config = ContextConfig().load(CONTEXT_CONFIG_PATH_JSON)
vocab = Vocab.load(VOCAB_PATH)
model_cfg = ModelConfig().load(MODEL_CONFIG_PATH_JSON)

model = TED(model_cfg)
model.set_context_config(context_config)
model_state = torch.load(WEIGHTS_PATH_PICKLE)
model.load_state_dict(model_state)
model = model.to(gen_cfg.device)
model.eval()

total_params = 0
total_params += sum(param.numel() for param in model.parameters())

print(f'Total number of parameters: {total_params / 1000000:.6f} M')
print('')

print(prompt, end="", flush=True)
chars = prompt

with torch.inference_mode():
    for _ in range(gen_cfg.num_tokens_to_generate):
        tokens = torch.tensor([vocab.encode(chars)], dtype=torch.long, device='cpu')
        logits = model.forward(tokens)
        torch.set_printoptions(profile="full")
        sampled = sample(logits[0], top_k=gen_cfg.top_k, top_p=gen_cfg.top_p, temperature=gen_cfg.temperature)[-1]
        d_chars = vocab.decode(sampled.tolist())
        print(d_chars, end="", flush=True)
        chars = d_chars
