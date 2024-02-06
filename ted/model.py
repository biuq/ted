from dataclasses import dataclass, asdict
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import base64

from ted.config import ContextConfig, ModelConfig

class TED(nn.Module):
    """
    TED - Trainable Exponential Decay(s)
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for _ in range(cfg.num_layers):
            self.layers.append(TEDLayer(
                dim=cfg.dim,
                hidden_dim=cfg.hidden_dim,
                norm_eps=cfg.norm_eps
            ))
        self.out_proj_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.out_proj = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        x = F.embedding(x, self.out_proj.weight)  # https://paperswithcode.com/method/weight-tying
        
        layer: TEDLayer
        for layer in self.layers:
            x = layer.forward(x)

        return self.out_proj.forward(self.out_proj_norm(x))

    def reset(self):
        layer: TEDLayer
        for layer in self.layers:
            layer.reset()
    
    def set_context_config(self, config: ContextConfig):
        """
        Setting context config enables the context usage during inference.
        """
        layer: TEDLayer
        for layer in self.layers:
            layer.decay.context_config = config

    def save_json(self, path: str):
        state = self.state_dict()
        json_state = dict()
        for k in state:
            json_state[k] = {
                'size': state[k].size(),
                'data': base64.b64encode(state[k].cpu().numpy().tobytes()).decode('utf-8')
            }
        with open(path, mode='wt', encoding='utf-8') as f:
            json.dump(json_state, f)

    def save_pickle(self, path: str):
        torch.save(self.state_dict(), path)

    def load_pickle(self, path: str):
        self.load_state_dict(torch.load(path))

class TEDLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float = 1E-6
    ) -> None:
        super().__init__()
        self.decay_norm = RMSNorm(dim, norm_eps)
        self.decay = ExponentialDecay(dim)
        self.ffn_norm = RMSNorm(dim, norm_eps)
        self.ffn = FeedForward(dim, hidden_dim)

    def forward(self, x: torch.Tensor):
        x = x + self.decay.forward(self.decay_norm(x))
        x = x + self.ffn.forward(self.ffn_norm(x))
        return x
    
    def reset(self):
        self.decay.reset()

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.to_hidden = nn.Linear(dim, hidden_dim, bias=False)
        self.to_hidden_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.to_dim = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x: torch.Tensor):
        h = self.to_hidden.forward(x)
        g = self.to_hidden_gate.forward(x)
        return self.to_dim(h * F.silu(g))

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class ExponentialDecay(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lambda_matrix = nn.Linear(dim, 1, bias=False)
        self.quantity_matrix = nn.Linear(dim, dim, bias=False)
        self.gate_matrix = nn.Linear(dim, dim, bias=False)
        self.output_matrix = nn.Linear(dim, dim, bias=False)

        self.context: ExponentialDecayContext | None = None
        self.context_config: ContextConfig | None = None
        self.time: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor):
        steps = tokens.size(-2)

        if self.time is None or self.time.size(-1) != steps or (self.time.is_inference() and self.training):
            self.time = torch.ones(steps, steps, device=tokens.device, dtype=torch.float32).tril_(-1).cumsum_(-2)

        time = self.time
        negative_lambdas = self.lambda_matrix.forward(tokens).sigmoid().log().swapaxes(-1, -2) # the negative of decay constant λ - lambda
        decay = (negative_lambdas * time).exp().tril() # e^(-λ*t)
        quantities = self.quantity_matrix.forward(tokens)
        output = decay @ quantities

        # We don't need context during training,
        # since all the context is in the input batch,
        # so don't set context config for training.
        if self.context_config is not None:
            output = self.use_context(output, time, negative_lambdas, quantities)

        return self.output_matrix.forward(F.silu(output)) * self.gate_matrix.forward(tokens)

    def use_context(
        self,
        output: torch.Tensor,
        time: torch.Tensor, 
        negative_lambdas: torch.Tensor, 
        quantities: torch.Tensor, 
    ):
        if self.context is not None:
            self.context.tick(time)
            output = output + self.context.influence()
        else:
            self.context = ExponentialDecayContext(time, negative_lambdas, quantities)
        
        self.context.add(time, negative_lambdas, quantities)
        self.context.keep_top_k(self.context_config.top_k, self.context_config.min_tokens_to_keep)
        self.context.keep_above_threshold(self.context_config.relevance_score, self.context_config.min_tokens_to_keep)
        
        return output

    def reset(self):
        self.context = None

class ExponentialDecayContext(nn.Module):
    def __init__(self, time: torch.Tensor, negative_lambdas: torch.Tensor, quantities: torch.Tensor) -> None:
        super().__init__()
        self.time = time[-1:, :]
        self.negative_lambdas = negative_lambdas
        self.tokens = quantities
        self.magnitudes = torch.linalg.vector_norm(quantities, dim=-1, keepdim=True).swapaxes(-1, -2)

    def tick(self, steps: int | torch.Tensor):
        if isinstance(steps, torch.Tensor):
            steps = steps.size(0)
        if steps < 0:
            raise ValueError("negative steps are currently not supported")
        if steps == 0:
            return self.time
        self.time = self.time + torch.linspace(1, steps, steps, device=self.time.device, dtype=torch.float32)

    def decay_factor(self, steps: int | torch.Tensor = 0):
        return (self.negative_lambdas * self.time).exp()
    
    def influence(self):
        return self.decay_factor() @ self.tokens
    
    def add(self, time: torch.Tensor, negative_lambdas: torch.Tensor, quantities: torch.Tensor):
        self.time = torch.cat([self.time, time[-1:, :]], dim=-1)[-1:, :]
        self.negative_lambdas = torch.cat([self.negative_lambdas, negative_lambdas], dim=-1)
        self.tokens = torch.cat([self.tokens, quantities], dim=-2)
        self.magnitudes = torch.cat([self.magnitudes, torch.linalg.vector_norm(quantities, dim=-1, keepdim=True).swapaxes(-1, -2)], dim=-1)

    def keep_top_k(self, top_k: int, min_tokens_to_keep: int):
        if top_k < 1:
            return
        
        decay_factor = self.decay_factor()[..., -1:, :]        
        
        top_k = max(top_k, min_tokens_to_keep)
        top_k = min(decay_factor.size(-1), top_k)
        
        relevance_score = self.magnitudes * decay_factor
        _, indices = torch.topk(relevance_score, top_k, dim=-1, largest=True, sorted=False)
        
        if decay_factor.size(0) == 1:
            indices = indices.squeeze(0).squeeze(0)
            self.time = self.time[:, indices]
            self.negative_lambdas = self.negative_lambdas[:, :, indices]
            self.tokens = self.tokens[:, indices, :]
            self.magnitudes = self.magnitudes[:, :, indices]
        else:
            raise ValueError('batched mode is currently not supported')

    def keep_above_threshold(self, threshold: float, min_tokens_to_keep: int):
        if threshold < 0 or threshold > 1:
            return
        
        decay = self.decay_factor()[..., -1:, :]
        
        relevance_score = self.magnitudes * decay
        mask = relevance_score >= threshold
        indices = torch.where(mask)
        
        if indices[2].numel() < min_tokens_to_keep:
            return self.keep_top_k(min_tokens_to_keep, min_tokens_to_keep)
        
        if decay.size(0) == 1:
            indices = indices[2]
            self.time = self.time[:, indices]
            self.negative_lambdas = self.negative_lambdas[:, :, indices]
            self.tokens = self.tokens[:, indices, :]
            self.magnitudes = self.magnitudes[:, :, indices]
        else:
            raise ValueError('batched mode is currently not supported')
