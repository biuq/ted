from dataclasses import asdict, dataclass
import json
import math
from typing import Self

def load_json(path):
    with open(path, mode='rt', encoding='utf-8') as f:
        return json.load(f)  

@dataclass
class Config:
    def save(self, path: str) -> None:
        with open(path, mode='wt', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=4)
            
    def load(self: Self, path: str) -> Self:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.__dict__.update(data)
        return self

@dataclass
class ModelConfig(Config):
    num_layers: int = 4
    dim: int = 128
    hidden_dim: int = 128
    norm_eps: float = 1e-6
    vocab_size: int = 256

@dataclass
class ContextConfig(Config):
    top_k: int = 0
    relevance_score: float = 1E-6
    min_tokens_to_keep: int = 1

@dataclass
class TrainConfig(Config):
    device: str = 'cuda'
    random_seed: int = 1337
    train_sequence_length: int = 2048
    batch_size: int = 4
    max_iters: int = 10000
    eval_interval: int = 200
    eval_iters: int = 100
    checkpoint_interval: int = 1000
    start_learning_rate: int = 1e-5
    peak_learning_rate: int = 1e-3
    end_learning_rate: int = 1e-6
    warmup_percentage = 0.1
    load_vocab: bool = False
    load_weights: bool = False

    def warmup_iters(self):
        return math.ceil(self.max_iters * self.warmup_percentage)

@dataclass
class GenConfig(Config):
    device: str = 'cpu'
    num_tokens_to_generate: int = 2 ** 12
    top_k: int = 0
    top_p: float = 0.9
    temperature: float = 0.7
